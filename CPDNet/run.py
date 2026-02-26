# =======================
# Import required libraries and modules
# =======================
import argparse          # For parsing command-line arguments
import subprocess        # For running external processes (if needed)
import tensorflow as tf  # Main deep learning framework
import numpy as np       # Numerical operations and array handling
from datetime import datetime  # For date and time operations
import json              # For reading/writing JSON files
import os                # For file and directory operations
import sys               # For system-specific parameters and functions
import datetime          # For date and time operations (duplicate import, but sometimes used for full module)
import time              # For timing and performance measurement
import collections       # For namedtuple and other collection utilities

# Set up base directory for module imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import project-specific modules
import P2PNET            # Main model definition
import ioUtil            # Data loading and utility functions


# DEFAULT SETTINGS
# =======================
# Argument Parsing and Dataset Loading
# =======================
# Set up command-line arguments for experiment configuration
parser = argparse.ArgumentParser()

# Data file arguments
parser.add_argument('--train_hdf5', default='data_hdf5/spine_train.hdf5', help='Path to training HDF5 file')
parser.add_argument('--test_hdf5', default='data_hdf5/spine_test.hdf5', help='Path to test HDF5 file')

# Domain and metadata arguments
parser.add_argument('--domain_A', default='skeleton', help='Name of domain A (e.g., skeleton)')
parser.add_argument('--domain_B', default='surface',  help='Name of domain B (e.g., surface)')
parser.add_argument('--meta_A', default='meta',  help='Metadata key for domain A')
parser.add_argument('--pre_T', default='pre_T',  help='Translation data key for domain A')
parser.add_argument('--pre_R', default='pre_R',  help='Rotation data key for domain A')
parser.add_argument('--post_T', default='post_T',  help='Translation data key for domain B')
parser.add_argument('--post_R', default='post_R',  help='Rotation data key for domain B')

# Training and model hyperparameters
parser.add_argument('--mode', type=str, default='train', help='Mode: train or test')
parser.add_argument('--modeCode', type=str, default='create', help='Mode code (custom use)')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use [default: 0]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch size during training [default: 6]')
parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to run [default: 500]')
parser.add_argument('--decayEpoch',  type=int, default=200, help='Epochs before decaying learning rate')
parser.add_argument('--learning_rate',  type=float, default=0.005, help='Initial learning rate')

# Loss weights and advanced hyperparameters
parser.add_argument('--shapeWeight', type=float, default=1, help='Shape loss weight')
parser.add_argument('--densityWeight', type=float, default=0, help='Density loss weight')
parser.add_argument('--regularWeight', type=float, default=0, help='Regularization loss weight')
parser.add_argument('--localWeight', type=float, default=0, help='Local regularization weight')
parser.add_argument('--metaWeight', type=float, default=2, help='Meta-data loss weight')
parser.add_argument('--nnk', type=int, default=24, help='Number of nearest neighbors for density loss')
parser.add_argument('--range_max', type=float, default=150, help='Max length of point displacement')
parser.add_argument('--radiusScal', type=float, default=1, help='Scaling factor for radii in PointNet++')
parser.add_argument('--noiseLength', type=int, default=20, help='Length of point-wise noise vector')
parser.add_argument('--metaLength', type=int, default=6, help='Length of meta-data vector')

# Checkpoint and output arguments
parser.add_argument('--checkpoint', default=None, help='Path to checkpoint file for restoring model')
parser.add_argument('--point_num', type=int, default=108, help='Number of points per sample (set automatically)')
parser.add_argument('--point_num_meta', type=int, default=108, help='Number of meta points per sample (set automatically)')
parser.add_argument('--vertebra_num', type=int, default=102, help='Number of vertebrae (set automatically)')
parser.add_argument('--example_num', type=int, default=None, help='Number of examples (set automatically)')
parser.add_argument('--output_dir', type=str,  default=None, help='Output directory (set automatically)')

# Parse all arguments and store in FLAGS
FLAGS = parser.parse_args()

# =======================
# Load datasets using ioUtil helper functions
# =======================
Train_examples = ioUtil.load_examples(
    FLAGS.train_hdf5, FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A,
    FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')
Eval_examples  = ioUtil.load_eval_examples(
    FLAGS.test_hdf5, FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A,
    FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')
Test_examples  = ioUtil.load_test_examples(
    FLAGS.test_hdf5, FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A,
    FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')

# =======================
# Set up key training constants based on loaded data
# =======================
FLAGS.point_num = Train_examples.pointsets_A.shape[1]  # Number of points per sample
FLAGS.vertebra_num = Train_examples.pointsets_T_A.shape[1]  # Number of vertebrae per sample
POINT_NUM = FLAGS.point_num

Example_NUM = Train_examples.pointsets_A.shape[0]  # Total number of training examples
FLAGS.example_num =  Example_NUM

TRAINING_EPOCHES = FLAGS.epoch  # Number of training epochs

batch_size = FLAGS.batch_size  # Batch size for training

# =======================
# Data consistency check
# =======================
if Train_examples.pointsets_B.shape[1] != POINT_NUM \
    or Eval_examples.pointsets_A.shape[1] != POINT_NUM \
    or Eval_examples.pointsets_B.shape[1] != POINT_NUM :
    print('point number inconsistent in the data set.')
    exit()


# =======================
# Create output folders for results, models, and summaries
# =======================
# Determine output directory name based on training file and domains
datapath, basefname = os.path.split(FLAGS.train_hdf5)
output_dir = 'output_' + basefname[0:basefname.index('_')] + '_' + FLAGS.domain_A + '-' + FLAGS.domain_B
# Create main output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Create subdirectory for storing trained model checkpoints
MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

# Create subdirectory for TensorBoard summaries
SUMMARIES_FOLDER = os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

# =======================
# (Optional) Save test input point clouds for visualization/debugging
# Uncomment the following lines to export ground truth point clouds as .ply files
# ioUtil.output_point_cloud_ply(Eval_examples.pointsets_A, Eval_examples.names, output_dir, 'gt_' + FLAGS.domain_A)
# ioUtil.output_point_cloud_ply(Eval_examples.pointsets_B, Eval_examples.names, output_dir, 'gt_' + FLAGS.domain_B)

# =======================
# Print all parsed arguments for reproducibility and logging
# =======================
for k, v in FLAGS._get_kwargs():
    print(k + ' = ' + str(v))

def train():
    """
    Main training and evaluation loop for the P2PNET model.
    
    This function sets up the TensorFlow computation graph, initializes the model, handles checkpointing,
    manages TensorBoard summaries, and orchestrates the training and evaluation process over multiple epochs.
    It includes inner functions for training and evaluation of a single epoch, and manages session configuration
    for GPU usage and reproducibility.
    """
    # Create a new TensorFlow computation graph
    with tf.Graph().as_default() as g:
        # Assign computation to the specified GPU (or CPU)
        with tf.device('/cpu:' + str(FLAGS.gpu)):
            # Build the P2PNET model using parsed arguments
            model = P2PNET.create_model(FLAGS)

        # =======================
        # Initialization and Session Configuration
        # =======================
        # Set up saver for model checkpoints (keep up to 25)
        saver = tf.train.Saver( max_to_keep=25 )
        # Configure TensorFlow session for flexible GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Allow dynamic GPU memory allocation
        config.allow_soft_placement = True      # Allow TF to fall back to CPU if needed
        config.gpu_options.per_process_gpu_memory_fraction = 0.90  # Limit GPU memory usage
        config.log_device_placement=True        # Log device placement for debugging
        sess = tf.Session(config=config)

        # Initialize global and local variables
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)

        # =======================
        # Restore from checkpoint if provided
        # =======================
        Start_epoch_number = 1
        if FLAGS.checkpoint is not None:
            print('load checkpoint: ' + FLAGS.checkpoint)
            saver.restore(sess, FLAGS.checkpoint )
            # Extract epoch number from checkpoint filename
            fname = os.path.basename( FLAGS.checkpoint )
            Start_epoch_number = int( fname[6:-5] )  +  1
            print( 'Start_epoch_number = ' + str(Start_epoch_number) )

        # =======================
        # Set up TensorBoard summary writers for training and testing
        # =======================
        now = datetime.datetime.now()
        var_date= now.strftime("%Y-%m-%d %H:%M")
        train_writer = tf.summary.FileWriter(
            "%s/train_%.4f_%.4f_%.4f_%d_%d_%.4f_%d_%s" % (
                SUMMARIES_FOLDER, FLAGS.localWeight, FLAGS.regularWeight, FLAGS.densityWeight,
                FLAGS.batch_size,FLAGS.radiusScal, FLAGS.learning_rate,FLAGS.metaWeight,var_date),
            sess.graph)
        test_writer = tf.summary.FileWriter(
            "%s/test_%.4f_%.4f_%.4f_%d_%d_%.4f_%d_%s" % (
                SUMMARIES_FOLDER, FLAGS.localWeight, FLAGS.regularWeight, FLAGS.densityWeight,
                FLAGS.batch_size,FLAGS.radiusScal, FLAGS.learning_rate,FLAGS.metaWeight,var_date))

        # Save experiment arguments to file for reproducibility
        fcmd = open(os.path.join(output_dir, 'arguments.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        # =======================
        # Inner function: Train for one epoch
        # =======================
        def train_one_epoch(epoch_num):
            """
            Run one epoch of training: shuffle data, run batches, accumulate and log losses.
            """
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            start_time = time.time()
            is_training = True
            # Shuffle training examples for this epoch
            Train_examples_shuffled = ioUtil.shuffle_examples(Train_examples)
            # Extract shuffled data arrays
            pointsets_A = Train_examples_shuffled.pointsets_A
            pointsets_B = Train_examples_shuffled.pointsets_B
            pointsets_meta_A = Train_examples_shuffled.pointsets_meta_A
            pointsets_T_A = Train_examples_shuffled.pointsets_T_A
            pointsets_R_A = Train_examples_shuffled.pointsets_R_A
            pointsets_T_B = Train_examples_shuffled.pointsets_T_B
            pointsets_R_B = Train_examples_shuffled.pointsets_R_B
            names = Train_examples_shuffled.names

            num_data = pointsets_A.shape[0]
            num_batch = num_data // batch_size

            # Initialize accumulators for losses
            total_total_loss = 0.0
            total_data_loss_A = 0.0
            total_shape_loss_A = 0.0
            total_density_loss_A = 0.0
            total_data_loss_B = 0.0
            total_shape_loss_B = 0.0
            total_density_loss_B = 0.0
            total_rigid_loss_A = 0.0
            total_rigid_loss_B = 0.0
            total_reg_loss = 0.0

            # Iterate over all batches
            for j in range(num_batch):
                begidx = j * batch_size
                endidx = (j + 1) * batch_size
                # Prepare feed dictionary for this batch
                feed_dict = {
                    model.pointSet_A_ph: pointsets_A[begidx: endidx, ...],
                    model.pointSet_B_ph: pointsets_B[begidx: endidx, ...],
                    model.pointSet_meta_A_ph: pointsets_meta_A[begidx: endidx, ...],
                    model.pointSet_T_A_ph: pointsets_T_A[begidx: endidx, ...],
                    model.pointSet_R_A_ph: pointsets_R_A[begidx: endidx, ...],
                    model.pointSet_T_B_ph: pointsets_T_B[begidx: endidx, ...],
                    model.pointSet_R_B_ph: pointsets_R_B[begidx: endidx, ...],
                    model.is_training_ph: is_training,
                }
                # Specify which tensors to fetch during session run
                fetches = {
                    "train": model.total_train,
                    "shapeLoss_A": model.shapeLoss_A,
                    "densityLoss_A": model.densityLoss_A,
                    "shapeLoss_B": model.shapeLoss_B,
                    "densityLoss_B": model.densityLoss_B,
                    "total_loss": model.total_loss,
                    "data_loss_A": model.data_loss_A,
                    "data_loss_B": model.data_loss_B,
                    "rigid_loss_A": model.rigid_loss_A,
                    "rigid_loss_B": model.rigid_loss_B,
                    "regul_loss": model.regul_loss,
                    "learning_rate": model.learning_rate,
                    "global_step": model.global_step,
                    "Predicted_A": model.Predicted_A,
                    "Predicted_B": model.Predicted_B,
                    "displace_B2A":model.list_dis_B2A_all,
                    "displace_A2B": model.list_dis_A2B_all,
                    "A":model.pointSet_A_ph,
                    "B":model.pointSet_B_ph,
                }
                # Run the session and get results for this batch
                results = sess.run(fetches, feed_dict=feed_dict)
                # (Optional) Extract predictions and displacements for further analysis
                A=results["A"]
                B=results["B"]
                Predicted_A = results["Predicted_A"]
                Predicted_B = results["Predicted_B"]
                displace_B2A = results["displace_B2A"]
                displace_A2B = results["displace_A2B"]

                # Accumulate losses for reporting
                total_total_loss += results["total_loss"]
                total_data_loss_A += results["data_loss_A"]
                total_shape_loss_A += results["shapeLoss_A"]
                total_density_loss_A += results["densityLoss_A"]
                total_data_loss_B += results["data_loss_B"]
                total_shape_loss_B += results["shapeLoss_B"]
                total_density_loss_B += results["densityLoss_B"]
                total_rigid_loss_A += results["rigid_loss_A"]
                total_rigid_loss_B += results["rigid_loss_B"]
                total_reg_loss += results["regul_loss"]

            # Average losses over all batches
            total_total_loss /= num_batch
            total_data_loss_A      /= num_batch
            total_shape_loss_A  /= num_batch
            total_density_loss_A   /= num_batch
            total_data_loss_B      /= num_batch
            total_shape_loss_B  /= num_batch
            total_density_loss_B   /= num_batch
            total_rigid_loss_A /= num_batch
            total_rigid_loss_B /= num_batch
            total_reg_loss         /= num_batch

            # Write training summaries for TensorBoard
            training_sum = sess.run( model.training_sum_ops, \
                                    feed_dict={model.train_dataloss_A_ph: total_data_loss_A, \
                                               model.train_dataloss_B_ph: total_data_loss_B, \
                                               model.train_totalloss_ph: total_total_loss, \
                                               model.train_rigidloss_A_ph: total_rigid_loss_A, \
                                               model.train_rigidloss_B_ph: total_rigid_loss_B, \
                                               model.train_regul_ph: total_reg_loss, \
                                               })
            train_writer.add_summary(training_sum, epoch_num)

            # Print losses for monitoring
            print(  '\tData_loss_A = %.4f,' % total_data_loss_A    + \
                    '    shape = %.4f,' % total_shape_loss_A + \
                    '    density = %.4f' % total_density_loss_A )
            print(  '\tData_loss_B = %.4f,' % total_data_loss_B    + \
                    '    shape = %.4f,' % total_shape_loss_B + \
                    '    density = %.4f' % total_density_loss_B )
            print(  '\tReg_loss: %.4f\n' % total_reg_loss)
            print('\tRigid_loss_A: %.4f\n' % total_rigid_loss_A)
            print('\tRigid_loss_B: %.4f\n' % total_rigid_loss_B)

            # Print timing and throughput information
            elapsed_time = time.time() - start_time
            print( '\tply/sec:' + str( round(num_data/elapsed_time) ) )
            print( '\tduration of this epoch:' + str(round(elapsed_time/60) ) + ' min' )
            print( '\testimated finishing time:' + str(round(elapsed_time/60.0 * (TRAINING_EPOCHES-epoch_num-1)) ) + ' min' )

        # =======================
        # Inner function: Evaluate for one epoch
        # =======================
        def eval_one_epoch(epoch_num, mustSavePly=False):
            """
            Evaluate the model for one epoch on either the validation or test set.
    
            This function runs the model in evaluation mode, processes the dataset in batches, accumulates losses and metrics,
            and optionally saves predicted point clouds for visualization. It also writes TensorBoard summaries and prints
            evaluation statistics for monitoring.
    
            Args:
                epoch_num (int): The current epoch number (used for logging and saving results).
                mustSavePly (bool): If True, always save predicted point clouds as .ply files (for test mode or specific epochs).
            """
            is_training = False  # Disable training-specific behaviors (e.g., dropout)
            # Shuffle evaluation and test examples for this epoch
            Eval_examples_shuffled = ioUtil.shuffle_examples(Eval_examples)
            Test_examples_shuffled = ioUtil.shuffle_examples(Test_examples)

            # Select which dataset to evaluate based on mode
            if FLAGS.mode=='train':
                # Use validation set during training
                pointsets_A = Eval_examples.pointsets_A
                pointsets_B = Eval_examples.pointsets_B
                pointsets_meta_A = Eval_examples.pointsets_meta_A
                pointsets_T_A = Eval_examples.pointsets_T_A
                pointsets_R_A = Eval_examples.pointsets_R_A
                pointsets_T_B = Eval_examples.pointsets_T_B
                pointsets_R_B = Eval_examples.pointsets_R_B
                names = Eval_examples.names
            else:
                # Use test set during testing
                pointsets_A = Test_examples.pointsets_A
                pointsets_B = Test_examples.pointsets_B
                pointsets_meta_A = Test_examples.pointsets_meta_A
                pointsets_T_A = Test_examples.pointsets_T_A
                pointsets_R_A = Test_examples.pointsets_R_A
                pointsets_T_B = Test_examples.pointsets_T_B
                pointsets_R_B = Test_examples.pointsets_R_B
                names = Test_examples.names

            num_data = pointsets_A.shape[0]
            num_batch = num_data // batch_size

            # Initialize accumulators for losses and metrics
            total_total_loss = 0.0
            total_data_loss_A = 0.0
            total_shape_loss_A = 0.0
            total_density_loss_A = 0.0
            total_data_loss_B = 0.0
            total_shape_loss_B = 0.0
            total_density_loss_B = 0.0
            total_rigid_loss_A = 0.0
            total_rigid_loss_B = 0.0
            total_reg_loss = 0.0

            # Iterate over all batches in the evaluation set
            for j in range(num_batch):
                begidx = j * batch_size
                endidx = (j + 1) * batch_size
                # Prepare feed dictionary for this batch
                feed_dict = {
                    model.pointSet_A_ph: pointsets_A[begidx: endidx, ...],
                    model.pointSet_B_ph: pointsets_B[begidx: endidx, ...],
                    model.pointSet_meta_A_ph: pointsets_meta_A[begidx: endidx, ...],
                    model.pointSet_T_A_ph: pointsets_T_A[begidx: endidx, ...],
                    model.pointSet_R_A_ph: pointsets_R_A[begidx: endidx, ...],
                    model.pointSet_T_B_ph: pointsets_T_B[begidx: endidx, ...],
                    model.pointSet_R_B_ph: pointsets_R_B[begidx: endidx, ...],
                    model.is_training_ph: is_training,
                }
                # Specify which tensors to fetch during session run
                fetches = {
                    "shapeLoss_A": model.shapeLoss_A,
                    "densityLoss_A": model.densityLoss_A,
                    "shapeLoss_B": model.shapeLoss_B,
                    "densityLoss_B": model.densityLoss_B,
                    "total_loss": model.total_loss,
                    "data_loss_A": model.data_loss_A,
                    "data_loss_B": model.data_loss_B,
                    "rigid_loss_A": model.rigid_loss_A,
                    "rigid_loss_B": model.rigid_loss_B,
                    "regul_loss": model.regul_loss,
                    "Predicted_A": model.Predicted_A,
                    "Predicted_B": model.Predicted_B,
                    "displace_B2A": model.list_dis_B2A_all,
                    "displace_A2B": model.list_dis_A2B_all,
                    "Accuracy": model.Accuracy,
                }

                # Run the session and get results for this batch
                results = sess.run(fetches, feed_dict=feed_dict)

                # Accumulate losses and metrics for reporting
                total_total_loss += results["total_loss"]
                total_data_loss_A += results["data_loss_A"]
                total_shape_loss_A += results["shapeLoss_A"]
                total_density_loss_A += results["densityLoss_A"]
                total_data_loss_B += results["data_loss_B"]
                total_shape_loss_B += results["shapeLoss_B"]
                total_density_loss_B += results["densityLoss_B"]
                total_rigid_loss_A += results["rigid_loss_A"]
                total_rigid_loss_B += results["rigid_loss_B"]
                total_reg_loss += results["regul_loss"]
                Accuracy = results["Accuracy"]

                # Optionally save predicted point clouds as .ply files for visualization/debugging
                if FLAGS.mode=='test' and (epoch_num == 200 or epoch_num == 300  or epoch_num == 400 or epoch_num == 500 or epoch_num == 600 or epoch_num == 700 or  epoch_num == 800 or epoch_num == 900 or epoch_num == 1000 or mustSavePly):
                    # Save predicted point sets for this batch
                    nametosave = names[begidx: endidx, ...]
                    Predicted_A_xyz = np.squeeze(np.array(results["Predicted_A"]))
                    Predicted_B_xyz = np.squeeze(np.array(results["Predicted_B"]))
                    ioUtil.output_point_cloud_ply(Predicted_A_xyz, nametosave, output_dir,
                                                  'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X1')
                    ioUtil.output_point_cloud_ply(Predicted_B_xyz, nametosave, output_dir,
                                                  'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X1')

            # Average losses and metrics over all batches
            total_total_loss /= num_batch
            total_data_loss_A      /= num_batch
            total_shape_loss_A  /= num_batch
            total_density_loss_A   /= num_batch
            total_data_loss_B      /= num_batch
            total_shape_loss_B  /= num_batch
            total_density_loss_B   /= num_batch
            total_rigid_loss_A /= num_batch
            total_rigid_loss_B /= num_batch
            total_reg_loss         /= num_batch

            # Write evaluation summaries for TensorBoard
            testing_sum = sess.run( model.testing_sum_ops, \
                                    feed_dict={model.test_dataloss_A_ph: total_data_loss_A, \
                                               model.test_dataloss_B_ph: total_data_loss_B, \
                                               model.test_totalloss_ph: total_total_loss, \
                                               model.test_regul_ph: total_reg_loss, \
                                               model.test_rigidloss_A_ph: total_rigid_loss_A, \
                                               model.test_rigidloss_B_ph: total_rigid_loss_B, \
                                               })
            test_writer.add_summary(testing_sum, epoch_num)

            # Print evaluation statistics for monitoring
            print('\tData_loss_A = %.4f,' % total_data_loss_A  + \
                  '    shape = %.4f,' % total_shape_loss_A + \
                  '    density = %.4f' % total_density_loss_A)
            print('\tData_loss_B = %.4f,' % total_data_loss_B + \
                  '    shape = %.4f,' % total_shape_loss_B + \
                  '    density = %.4f' % total_density_loss_B)
            print('\tRigid_loss_A: %.4f\n' % total_rigid_loss_A)
            print('\tRigid_loss_B: %.4f\n' % total_rigid_loss_B)
            print('\tReg_loss: %.4f\n' % total_reg_loss)
            print('\tAccuracy: %.4f\n' % Accuracy)

        ################## end  of test function #################### end  of test function ##########

        # =======================
        # Ensure model checkpoint directory exists
        # =======================
        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        # =======================
        # Main training or testing loop
        # =======================
        if FLAGS.mode=='train':
            # Training mode: iterate over epochs
            for epoch in range(Start_epoch_number,  TRAINING_EPOCHES+1):
                print( '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))
                # Train for one epoch
                train_one_epoch(epoch)

                # Every 20 epochs, save a checkpoint and evaluate on the test set
                if epoch % 20 == 0:
                    # Save model checkpoint
                    cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch) + '.ckpt'))
                    print( 'Successfully store the checkpoint model into ' + cp_filename)
                    # Evaluate model on test dataset and print results
                    print('\n<<< Testing on the test dataset...')
                    eval_one_epoch(epoch, mustSavePly=False)
        else:
            # Testing mode: evaluate once on the test set and save predictions
            print( '\n<<< Testing on the test dataset ...')
            eval_one_epoch(Start_epoch_number, mustSavePly=True)
            
    train_writer.close()
    test_writer.close()
if __name__ == '__main__':
    train()
