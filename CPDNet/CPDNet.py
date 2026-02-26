# =======================
# Import required libraries and modules
# =======================
import os                    # OS path and directory operations
import sys                   # System-specific parameters and functions
import collections           # For namedtuple and other collection utilities
from numpy import linalg as LA   # Linear algebra operations from NumPy
import cv2                   # OpenCV for image processing (if used)
# TensorFlow backend and ops for advanced tensor manipulations
from tensorflow.python.keras import backend
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow import linalg as _linalg

# =======================
# Set up Python path for custom module imports
# =======================
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + "/pointnet_plusplus/utils")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/3d_interpolation")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/grouping")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/sampling")

# =======================
# Import main libraries and project modules
# =======================
import tensorflow as tf      # Main deep learning framework
import numpy as np           # Numerical operations and array handling
import ioUtil                # Project-specific I/O utilities
# import tensorflow_probability as tfp  # (Optional) TensorFlow Probability, if needed
import tf_util               # Project-specific TensorFlow utilities
from pointnet_util import pointnet_sa_module, pointnet_fp_module  # PointNet++ modules
# Autograd for automatic differentiation (used for custom gradients)
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad
# Set NumPy print options for better readability
np.set_printoptions(precision=4)

# =======================
# Define a namedtuple to organize all model-related tensors, placeholders, and operations
# =======================
# The Model namedtuple acts as a container for all key components of the network, including:
#   - Input placeholders for point sets, meta-data, transformations, and training mode
#   - Output tensors for predictions and displacements
#   - Loss components (data, shape, density, rigid, regularization, total)
#   - Training operations and learning rate
#   - TensorBoard summary ops for training and testing
#   - Placeholders for logging and summary writing
# This structure allows for clean, organized access to all relevant tensors and operations throughout the training and evaluation code.
Model = collections.namedtuple("Model", \
    "pointSet_A_ph, pointSet_B_ph, pointSet_meta_A_ph, pointSet_T_A_ph, pointSet_R_A_ph, pointSet_T_B_ph, pointSet_R_B_ph, "
    "is_training_ph, "
    "Predicted_A, Predicted_B, displace_BA, displace_AB, list_dis_A2B_all, list_dis_B2A_all, Accuracy, "
    "data_loss_A, shapeLoss_A, densityLoss_A, "
    "data_loss_B, shapeLoss_B, densityLoss_B, "
    "total_loss, "
    "rigid_loss_A, rigid_loss_B, "
    "regul_loss, "
    "data_train, rigid_train, total_train, "
    "learning_rate, global_step, bn_decay, "
    "training_sum_ops, testing_sum_ops, "
    "train_dataloss_A_ph, train_dataloss_B_ph, train_totalloss_ph, train_regul_ph, train_rigidloss_A_ph, train_rigidloss_B_ph, "
    "test_dataloss_A_ph, test_dataloss_B_ph, test_totalloss_ph, test_regul_ph, test_rigidloss_A_ph, test_rigidloss_B_ph"
)

def create_model( FLAGS  ):

    ############################################################
    # Section: Hyper-parameters
    # Set up global step, learning rate schedule, batch normalization decay,
    # and load training/testing data. These control optimization and regularization.
    ##############################################################

    # Global step variable for tracking training progress
    global_step = tf.train.get_or_create_global_step()

    # Exponential decay for learning rate
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,  # base learning rate
        global_step * FLAGS.batch_size,  # steps
        FLAGS.example_num * FLAGS.decayEpoch,  # decay step size
        0.5,  # decay rate
        staircase=True
    )
    # Prevent learning rate from dropping below 1e-4
    learning_rate = tf.maximum(learning_rate, 1e-4)

    # Batch normalization momentum decay
    bn_momentum = tf.train.exponential_decay(
        0.001,
        global_step * FLAGS.batch_size,
        FLAGS.example_num * FLAGS.decayEpoch * 2,
        0.5,
        staircase=True
    )
    # Cap batch normalization decay at 0.99
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)

    # Load training and testing examples from HDF5 files
    Train_examples = ioUtil.load_examples(
        FLAGS.train_hdf5, FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A,
        FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')
    Test_examples = ioUtil.load_examples(
        FLAGS.test_hdf5, FLAGS.domain_A, FLAGS.domain_B, FLAGS.meta_A,
        FLAGS.pre_T, FLAGS.pre_R, FLAGS.post_T, FLAGS.post_R, 'names')

    ##############################################################
    # Section: Create the network
    # Define all TensorFlow placeholders for input, output, and intermediate tensors.
    # These are used to feed data and control the model during training and inference.
    ##############################################################

    # Placeholders for point sets, metadata, and transformation targets
    pointSet_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Input points from domain A
    pointSet_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Input points from domain B
    pointSet_meta_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num_meta, 6))  # Metadata for points
    pointSet_T_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))  # Target translations for A
    #pointSet_R_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3, 3))  # (Unused)
    #pointSet_R_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))  # (Unused)
    pointSet_R_A_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 24, 3))  # Target rotations for A
    pointSet_T_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))  # Target translations for B
    #pointSet_R_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3 , 3))  # (Unused)
    #pointSet_R_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.vertebra_num, 3))  # (Unused)
    pointSet_R_B_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 24, 3))  # Target rotations for B

    # Placeholders for predicted outputs and displacements
    Predicted_A = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Predicted points for A
    Predicted_B = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Predicted points for B

    displace_A2B = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Displacement from A to B
    displace_B2A = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Displacement from B to A

    dis_A2B = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Auxiliary displacement
    dis_B2A = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Auxiliary displacement
    Accuracy = tf.constant(0.0, dtype=tf.float32)  # Accuracy metric

    noise_vertebra = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))  # Noise for augmentation

    is_training_ph = tf.placeholder(tf.bool, shape=())  # Training flag

    # Noise tensors for stochastic augmentation
    noise1 = None
    noise2 = None
    if FLAGS.noiseLength > 0:
        noise1 = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1, dtype=tf.float32)
        noise2 = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1, dtype=tf.float32)

    # Displacement prediction networks for A->B and B->A
    list_dis = []
    list_dis_A2B = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
    list_dis_A2B_all = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
    with tf.variable_scope("p2pnet_A2B") as scope:
        displace_A2B = get_displacements_transformation(pointSet_A_ph, is_training_ph, noise1, pointSet_meta_A_ph, FLAGS, bn_decay)
        dis = displace_A2B

    list_dis = []
    list_dis_B2A = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
    list_dis_B2A_all = tf.zeros((FLAGS.batch_size, FLAGS.point_num, 3), dtype=np.float32)
    with tf.variable_scope("p2pnet_B2A") as scope:
        displace_B2A = get_displacements_transformation(pointSet_B_ph, is_training_ph, noise1, pointSet_meta_A_ph, FLAGS, bn_decay)
        dis = displace_B2A

    # Final predicted shapes (after applying displacements)
    Predicted_A = pointSet_B_ph + displace_B2A
    Predicted_B = pointSet_A_ph + displace_A2B

    # Rigid transformation losses for predicted shapes
    rigid_loss_A = rigid_transform_3D(Predicted_A, pointSet_T_A_ph, pointSet_R_A_ph, FLAGS)  # Loss for pre-op prediction
    rigid_loss_A = rigid_loss_A
    rigid_loss_B = rigid_transform_3D(Predicted_B, pointSet_T_B_ph, pointSet_R_B_ph, FLAGS)  # Loss for post-op prediction
    rigid_loss_B = rigid_loss_B


    # Losses and accuracy metrics
    shapeLoss_A, densityLoss_A = get_Geometric_Loss_PointWise(Predicted_A, pointSet_A_ph, FLAGS)  # Shape/density loss for A
    Accuracy = get_Accuracy_Rec(Predicted_B, pointSet_B_ph, FLAGS)  # Accuracy metric for B
    shapeLoss_B, densityLoss_B = get_Geometric_Loss_PointWise(Predicted_B, pointSet_B_ph, FLAGS)  # Shape/density loss for B

    # Regularization loss (optional)
    if FLAGS.regularWeight > 0:
        regul_loss = get_Regularizing_Loss(pointSet_A_ph, pointSet_B_ph, Predicted_A, Predicted_B)
    else:
        regul_loss = tf.constant(0.0, dtype=tf.float32)

    RigidLoss = rigid_loss_B

    # Weighted sum of losses for training
    data_loss_A = shapeLoss_A * FLAGS.shapeWeight + densityLoss_A * FLAGS.densityWeight
    data_loss_B = shapeLoss_B * FLAGS.shapeWeight + densityLoss_B * FLAGS.densityWeight
    DataLoss = data_loss_B

    TotalLoss = DataLoss
    # Alternative loss combinations (commented out)
    # TotalLoss = DataLoss + RigidLoss * FLAGS.localWeight
    # TotalLoss = DataLoss + regul_loss * FLAGS.regularWeight + RigidLoss * FLAGS.localWeight
    # TotalLoss = DataLoss + regul_loss * FLAGS.regularWeight

    # ------------------- Training Operations -------------------
    train_variables = tf.trainable_variables()  # All trainable variables
    trainer = tf.train.AdamOptimizer(learning_rate)  # Adam optimizer

    # Training ops for different loss objectives
    #val_list = lambda: Model.Predicted_A  # Example for custom variable list
    #rigid_train_op = trainer.minimize(RigidLoss, var_list=[val_list])
    rigid_train_op = trainer.minimize(RigidLoss, var_list=train_variables, global_step=global_step)
    data_train_op = trainer.minimize(DataLoss, var_list=train_variables, global_step=global_step)
    total_train_op = trainer.minimize(TotalLoss, var_list=train_variables, global_step=global_step)

    rigid_train = rigid_train_op
    data_train = data_train_op
    total_train = total_train_op

    ##############################################################
    # Section: Create summarizers
    # Define TensorFlow summary ops for monitoring training and testing metrics in TensorBoard.
    ##############################################################

    # Placeholders for summary metrics (losses, regularization, etc.)
    train_totalloss_ph = tf.placeholder(tf.float32, shape=())
    train_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
    train_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
    train_regul_ph = tf.placeholder(tf.float32, shape=())
    train_rigidloss_A_ph = tf.placeholder(tf.float32, shape=())
    train_rigidloss_B_ph = tf.placeholder(tf.float32, shape=())

    test_totalloss_ph = tf.placeholder(tf.float32, shape=())
    test_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
    test_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
    test_regul_ph = tf.placeholder(tf.float32, shape=())
    test_rigidloss_A_ph = tf.placeholder(tf.float32, shape=())
    test_rigidloss_B_ph = tf.placeholder(tf.float32, shape=())



    # Scalar summary ops for TensorBoard visualization
    lr_sum_op = tf.summary.scalar('learning rate', learning_rate)
    global_step_sum_op = tf.summary.scalar('batch_number', global_step)

    train_totalloss_sum_op = tf.summary.scalar('train_totalloss', train_totalloss_ph)
    train_dataloss_A_sum_op = tf.summary.scalar('train_dataloss_A', train_dataloss_A_ph)
    train_dataloss_B_sum_op = tf.summary.scalar('train_dataloss_B', train_dataloss_B_ph)
    train_regul_sum_op = tf.summary.scalar('train_regul', train_regul_ph)
    train_rigidloss_A_sum_op = tf.summary.scalar('train_rigidloss_A', train_rigidloss_A_ph)
    train_rigidloss_B_sum_op = tf.summary.scalar('train_rigidloss_B', train_rigidloss_B_ph)

    test_totalloss_sum_op = tf.summary.scalar('test_totalloss', test_totalloss_ph)
    test_dataloss_A_sum_op = tf.summary.scalar('test_dataloss_A', test_dataloss_A_ph)
    test_dataloss_B_sum_op = tf.summary.scalar('test_dataloss_B', test_dataloss_B_ph)
    test_regul_sum_op = tf.summary.scalar('test_regul', test_regul_ph)
    test_rigidloss_A_sum_op = tf.summary.scalar('test_rigidloss_A', test_rigidloss_A_ph)
    test_rigidloss_B_sum_op = tf.summary.scalar('test_rigidloss_B', test_rigidloss_B_ph)


    # Merge all training and testing summaries for easy logging
    training_sum_ops = tf.summary.merge([
        lr_sum_op, train_totalloss_sum_op, train_dataloss_A_sum_op, train_dataloss_B_sum_op,
        train_regul_sum_op, train_rigidloss_A_sum_op, train_rigidloss_B_sum_op
    ])

    testing_sum_ops = tf.summary.merge([
        test_totalloss_sum_op, test_dataloss_A_sum_op, test_dataloss_B_sum_op,
        test_regul_sum_op, test_rigidloss_A_sum_op, test_rigidloss_B_sum_op
    ])

    return Model(
        pointSet_A_ph=pointSet_A_ph,  pointSet_B_ph=pointSet_B_ph,
        Predicted_A=Predicted_A,Predicted_B=Predicted_B, displace_BA=displace_B2A, displace_AB=displace_A2B, list_dis_A2B_all=list_dis_A2B_all, list_dis_B2A_all=list_dis_B2A_all, Accuracy=Accuracy,
        pointSet_T_A_ph=pointSet_T_A_ph, pointSet_R_A_ph=pointSet_R_A_ph,
        pointSet_T_B_ph=pointSet_T_B_ph, pointSet_R_B_ph=pointSet_R_B_ph,
        pointSet_meta_A_ph=pointSet_meta_A_ph,
        is_training_ph=is_training_ph,
        data_loss_A=data_loss_A,   shapeLoss_A=shapeLoss_A,     densityLoss_A=densityLoss_A,
        data_loss_B=data_loss_B,   shapeLoss_B=shapeLoss_B,     densityLoss_B=densityLoss_B,
        total_loss=TotalLoss,
        rigid_loss_A=rigid_loss_A,
        rigid_loss_B=rigid_loss_B,
        regul_loss=regul_loss,
        data_train=data_train,    rigid_train=rigid_train,   total_train=total_train,
        learning_rate=learning_rate, global_step=global_step, bn_decay=bn_decay,
        training_sum_ops=training_sum_ops, testing_sum_ops=testing_sum_ops,
        train_dataloss_A_ph=train_dataloss_A_ph, train_dataloss_B_ph=train_dataloss_B_ph, train_totalloss_ph=train_totalloss_ph,train_regul_ph=train_regul_ph, train_rigidloss_A_ph=train_rigidloss_A_ph, train_rigidloss_B_ph=train_rigidloss_B_ph,
        test_dataloss_A_ph=test_dataloss_A_ph, test_dataloss_B_ph=test_dataloss_B_ph, test_totalloss_ph=test_totalloss_ph, test_regul_ph=test_regul_ph, test_rigidloss_A_ph=test_rigidloss_A_ph, test_rigidloss_B_ph=test_rigidloss_B_ph
    )

def get_displacements_transformation(input_vertebra, is_training, noise, meta_A, FLAGS, bn_decay=None):
    """
    Predicts point-wise displacements using a simple PointNet-based network.
    Args:
        input_vertebra: Tensor of shape [batch_size, num_points, 3], input point cloud.
        is_training: Boolean Tensor, training mode flag.
        noise: Optional Tensor for noise augmentation, shape [batch_size, num_points, noiseLength].
        meta_A: Metadata tensor (unused here).
        FLAGS: Namespace of hyperparameters.
        bn_decay: Batch normalization decay value.
    Returns:
        displacements: Tensor of predicted displacements, shape [batch_size, num_points, 3].
    """

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num

    # Start with input vertebra points
    l0_vertebra = input_vertebra

    # If noise augmentation is enabled, concatenate noise to input features
    if noise is not None:
        l0_vertebra = tf.concat(axis=2, values=[l0_vertebra, noise])

    # First layer: 1D convolution to extract 128 features per point
    net = tf_util.conv1d(l0_vertebra, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    # Second layer: 1D convolution to reduce to 64 features per point
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # Third layer: 1D convolution to output 3 values per point (displacement vector)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    # Apply sigmoid to constrain output, scale to [-range_max, range_max]
    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    # Return predicted displacements
    return displacements

def get_displacements(input_points, input_vertebra_T, is_training, noise, meta_A, FLAGS, bn_decay=None):
    """
    Predicts point-wise displacements using a PointNet++ architecture with set abstraction and feature propagation.
    Args:
        input_points: Tensor of shape [batch_size, num_points, 3], input point cloud.
        input_vertebra_T: Target vertebra transformation (unused here).
        is_training: Boolean Tensor, training mode flag.
        noise: Optional Tensor for noise augmentation, shape [batch_size, num_points, noiseLength].
        meta_A: Metadata tensor (unused here).
        FLAGS: Namespace of hyperparameters.
        bn_decay: Batch normalization decay value.
    Returns:
        displacements: Tensor of predicted displacements, shape [batch_size, num_points, 3].
    """

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num

    # Input point cloud
    point_cloud = input_points

    # Initial coordinates for set abstraction
    l0_xyz = point_cloud
    l0_points = None

    # Set Abstraction layers: progressively downsample and extract features
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(
        l0_xyz, l0_points,
        npoint=200, radius=0.1 * FLAGS.radiusScal, nsample=30,
        mlp=[64, 64, 128], mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='layer1')

    l2_xyz, l2_points, l2_indices = pointnet_sa_module(
        l1_xyz, l1_points,
        npoint=100, radius=0.2 * FLAGS.radiusScal, nsample=30,
        mlp=[128, 128, 256], mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='layer2')

    l3_xyz, l3_points, l3_indices = pointnet_sa_module(
        l2_xyz, l2_points,
        npoint=50, radius=0.4 * FLAGS.radiusScal, nsample=30,
        mlp=[256, 256, 512], mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Global feature extraction (group_all=True)
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(
        l3_xyz, l3_points,
        npoint=None, radius=None, nsample=None,
        mlp=[512, 512, 1024], mlp2=None, group_all=True,
        is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers: upsample and interpolate features back to original points
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512, 512], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512, 256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay, scope='fa_layer4')

    # If noise augmentation is enabled, concatenate noise to features
    if noise is not None:
        l0_points = tf.concat(axis=2, values=[l0_points, noise])

    # Fully connected layers (1D conv) to predict displacements
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    # Apply sigmoid to constrain output, scale to [-range_max, range_max]
    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    # Return predicted displacements
    return displacements

def get_Accuracy_Rec(x, y, FLAGS, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        # calculate shape loss
        diff = x-y
        square_diff = tf.square(diff)
        minRow = tf.reduce_sum(square_diff, axis=2)
        dis = tf.sqrt(minRow)
        minColl = tf.reduce_mean(dis, axis=1)
        Accuracy =tf.reduce_mean(minColl)

        return Accuracy

def get_Geometric_Loss_PointWise(predictedPts, targetpoints, FLAGS):
    # --- Shape Loss Calculation ---
    # Compute the difference between predicted and target points (per point)
    diff = predictedPts - targetpoints

    # Square the differences for each coordinate (x, y, z)
    square_diff = tf.square(diff)

    # Sum squared differences across coordinates to get squared Euclidean distance for each point
    minRow = tf.reduce_sum(square_diff, axis=2)

    # Take the square root to get Euclidean distance for each point
    dis = tf.sqrt(minRow)
    # Alternative: add small epsilon for numerical stability (commented out)
    # dis = tf.sqrt(1e-6 + minRow)

    # Average distances across all points in each batch
    minColl = tf.reduce_mean(dis, axis=1)

    # Average across all batches to get final shape loss
    shapeLoss = tf.reduce_mean(minColl)

    ######## 
    # square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    # dist = tf.sqrt(square_dist)
    #
    # # calculate density loss
    # square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    # dist2 = tf.sqrt(square_dist2)
    # knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk)
    # knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    # densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

    # calculate rigid transformation loss
    # square_dist3 = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    # dist3 = tf.sqrt(square_dist3)
    # minRow3 = tf.reduce_min(dist3, axis=2)
    # minCol3 = tf.reduce_min(dist3, axis=1)
    # shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)
    ########
    
    #data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    densityLoss=tf.constant(0.0, dtype=tf.float32)
    return  shapeLoss, densityLoss

def get_Geometric_Loss(predictedPts, targetpoints, FLAGS):

    # --- Shape Loss Calculation ---
    # Compute pairwise squared Euclidean distances between target and predicted points
    square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)

    # Take square root to get Euclidean distances
    dist = tf.sqrt(square_dist)

    # For each target point, find the minimum distance to any predicted point (per batch)
    minRow = tf.reduce_min(dist, axis=2)

    # For each predicted point, find the minimum distance to any target point (per batch)
    minCol = tf.reduce_min(dist, axis=1)

    # Average minimum distances across all points and batches
    shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)
  
    ########
    # calculate density loss
    # square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    # dist2 = tf.sqrt(square_dist2)
    # knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk)
    # knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    # densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

    # calculate rigid transformation loss
    #square_dist3 = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    #dist3 = tf.sqrt(square_dist3)
    #minRow3 = tf.reduce_min(dist3, axis=2)
    #minCol3 = tf.reduce_min(dist3, axis=1)
    #shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)
    ########
    
    densityLoss = tf.constant(0.0, dtype=tf.float32)
    # data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    # return data_loss, shapeLoss, densityLoss
    return  shapeLoss, densityLoss

def get_Regularizing_Loss(pointSet_A_ph, pointSet_B_ph, Predicted_A, Predicted_B):
    """
    Computes a regularization loss to encourage consistency between predicted and original point sets.
    Args:
        pointSet_A_ph: Tensor, original points from domain A.
        pointSet_B_ph: Tensor, original points from domain B.
        Predicted_A: Tensor, predicted points for domain A.
        Predicted_B: Tensor, predicted points for domain B.
    Returns:
        RegularLoss: Scalar Tensor, regularization loss value.
    """

    # Concatenate original A and predicted B for comparison
    displacements_A = tf.concat(axis=2, values=[pointSet_A_ph, Predicted_B])
    # Concatenate predicted A and original B for comparison
    displacements_B = tf.concat(axis=2, values=[Predicted_A, pointSet_B_ph])

    # Compute pairwise squared Euclidean distances between the two sets
    square_dist = pairwise_l2_norm2_batch(displacements_A, displacements_B)
    dist = tf.sqrt(square_dist)
    
    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)

    # Average minimum distances across all points and batches, then average both directions
    RegularLoss = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol)) / 2

    # Return the regularization loss
    return RegularLoss


def pairwise_l2_norm2_batch(x, y, scope=None):
    """
    Computes pairwise squared Euclidean distances between two batches of point sets.
    Args:
        x: Tensor of shape [batch_size, num_points_x, dim], first set of points.
        y: Tensor of shape [batch_size, num_points_y, dim], second set of points.
        scope: Optional TensorFlow scope.
    Returns:
        square_dist: Tensor of shape [batch_size, num_points_x, num_points_y],
            pairwise squared distances between points in x and y for each batch.
    """
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        # Get number of points in each set
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        # Expand x to shape [batch_size, num_points_x, dim, num_points_y]
        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y]))

        # Expand y to shape [batch_size, num_points_y, dim, num_points_x], then transpose
        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        # Compute difference between each pair of points
        diff = tf.subtract(xx, yy)
        # Add small epsilon for numerical stability, then square differences
        square_diff = tf.square(1e-4 + diff)

        # Sum squared differences across coordinates to get squared Euclidean distance
        square_dist = tf.reduce_sum(square_diff, 2)

        # Return pairwise squared distances
        return square_dist

def pairwise_l2_norm2_batch_high_dim(x, y, scope=None):
    """
    Computes pairwise squared Euclidean distances between two batches of high-dimensional point sets.
    Args:
        x: Tensor of shape [batch_size, num_points_x, dim1, dim2], first set of points (high-dimensional).
        y: Tensor of shape [batch_size, num_points_y, dim1, dim2], second set of points (high-dimensional).
        scope: Optional TensorFlow scope.
    Returns:
        square_dist: Tensor of shape [batch_size, num_points_x, num_points_y],
            pairwise squared distances between points in x and y for each batch.
    """
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        # Get number of points in each set
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        # Expand x to shape [batch_size, num_points_x, dim1, dim2, num_points_y]
        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, 1, nump_y]))

        # Expand y to shape [batch_size, num_points_y, dim1, dim2, num_points_x], then transpose
        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 4, 2, 3, 1])

        # Compute difference between each pair of high-dimensional points
        diff = tf.subtract(xx, yy)
        # Square the differences for each coordinate
        square_diff = tf.square(diff)

        # Sum squared differences across dim1 and dim2 to get squared Euclidean distance
        square_dist = tf.reduce_sum(square_diff, [2, 3])

        # Return pairwise squared distances
        return square_dist

def f(x):

    return x

def rigid_transform_3D(Predicted_shape, target_T, target_R, FLAGS, scope=None):
    """
    Computes rigid transformation loss between predicted and target shapes.
    This loss measures how well the predicted vertebrae align with target translations and rotations.
    Args:
        Predicted_shape: Tensor, predicted vertebra shape [batch_size, num_points, 3].
        target_T: Tensor, target translations [batch_size, 23, 3].
        target_R: Tensor, target rotations [batch_size, 24, 3].
        FLAGS: Namespace of hyperparameters.
        scope: Optional TensorFlow scope.
    Returns:
        rigidLoss: Scalar Tensor, sum of translation and rotation alignment losses.
    """
    rigidLoss = tf.constant(2.0, dtype=tf.float32)  # Default loss value
    list_translations = []  # Store translations for each batch
    list_rotations = []    # Store rotations for each batch
    batch_size = FLAGS.batch_size
    translations = tf.zeros((batch_size, 23, 3), dtype=np.float32)
    rotations = tf.zeros((batch_size, 23, 3), dtype=np.float32)
    # Loop over each batch
    for batch_id in range(0, batch_size, 1):
        # Extract predicted shape for current batch
        slice1_ = tf.slice(Predicted_shape, [batch_id, 0, 0], [1, -1, -1])
        c = tf.zeros((23, 3), dtype=np.float32)
        vecZ = tf.zeros((3), dtype=np.float32)
        vecY = tf.zeros((3), dtype=np.float32)
        vecX = tf.zeros((3), dtype=np.float32)
        vec = tf.zeros((batch_size, 24, 3), dtype=np.float32)
        trans = tf.zeros((23, 3), dtype=np.float32)
        rotate = tf.zeros((23, 3), dtype=np.float32)
        row = 102  # Number of points per vertebra (fixed for this dataset)
        if row == 102:
            j = 0
            list_c = []   # Store translation vectors
            list_xyz = [] # Store axes for each vertebra
            list_vec = [] # Store all axes
            # Compute translation vectors for each vertebra
            for i in range(0, 97, 6):
                T_1 = (slice1_[0, i + 6, :] - slice1_[0, i, :])
                T_2 = (slice1_[0, i + 7, :] - slice1_[0, i + 1, :])
                T_3 = (slice1_[0, i + 8, :] - slice1_[0, i + 2, :])
                T_4 = (slice1_[0, i + 9, :] - slice1_[0, i + 3, :])
                T_5 = (slice1_[0, i + 10, :] - slice1_[0, i + 4, :])
                T_6 = (slice1_[0, i + 11, :] - slice1_[0, i + 5, :])
                list_c.append(T_1)
                list_c.append(T_2)
                list_c.append(T_3)
                list_c.append(T_4)
                list_c.append(T_5)
                list_c.append(T_6)

            # Compute axes (Z, Y, X) for each vertebra
            for i in range(0, 108, 6):
                list_xyz = []
                # Z axis: difference between two points
                vecZ_x = slice1_[0, i + 5, 0] - slice1_[0, i + 4, 0]
                vecZ_y = slice1_[0, i + 5, 1] - slice1_[0, i + 4, 1]
                vecZ_z = slice1_[0, i + 5, 2] - slice1_[0, i + 4, 2]
                vecZ_t = tf.concat([[vecZ_x], [vecZ_y], [vecZ_z]], 0)
                # Normalize Z axis
                if tf.norm(vecZ_t, axis=None) == 0:
                    vecZ_norm_t = vecZ_t
                else:
                    vecZ_norm_t = vecZ_t / tf.norm(vecZ_t, axis=None)
                # Y axis: average difference between two pairs
                vecY_x = (slice1_[0, i + 1, 0] - slice1_[0, i + 0, 0] + slice1_[0, i + 3, 0] - slice1_[0, i + 2, 0]) / 2
                vecY_y = (slice1_[0, i + 1, 1] - slice1_[0, i + 0, 1] + slice1_[0, i + 3, 1] - slice1_[0, i + 2, 1]) / 2
                vecY_z = (slice1_[0, i + 1, 2] - slice1_[0, i + 0, 2] + slice1_[0, i + 3, 2] - slice1_[0, i + 2, 2]) / 2
                vecY_t = tf.concat([[vecY_x], [vecY_y], [vecY_z]], 0)
                # Orthogonalize Y axis to Z axis
                vecY_norm_t = gramschimdit3(vecZ_norm_t, vecY_t)
                # X axis: fixed vector, then orthogonalize to Z and Y
                vecX_t = tf.constant([1, 1, 1], dtype=tf.float32)
                vecZY_t = tf.stack((vecZ_norm_t, vecY_norm_t))
                vecX_norm_t = gramschimdit3(vecZY_t, vecX_t)
                j = j + 1
                list_xyz.append(vecZ_norm_t)
                list_xyz.append(vecY_norm_t)
                list_xyz.append(vecX_norm_t)
                list_vec.append(list_xyz)

            centers_t = tf.stack(list_c)   # All translation vectors
            vec_t = tf.stack(list_vec)     # All axes for vertebrae

            # Compute rotation matrices for each vertebra
            list_trans = []
            list_rotate = []
            for index in range(0, 17, 1):
                R = challismethod(tf.reshape(vec_t[index, :, :], [3, 3]), tf.reshape(vec_t[index + 1, :, :], [3, 3]))
                list_rotate.append(R)

            trans = tf.stack(centers_t)
            rotate = tf.stack(list_rotate)
            list_translations.append(trans)
            list_rotations.append(rotate)

    # Stack translations and rotations for all batches
    translations = tf.stack(list_translations)
    rotations = tf.stack(list_rotations)

    # Compute translation loss: difference between predicted and target translations
    diff = target_T - translations
    square_diff = tf.square(diff)
    minRow = tf.reduce_sum(square_diff, axis=2)
    dis = tf.sqrt(minRow)
    minColl = tf.reduce_mean(dis, axis=1)
    translationLoss = tf.reduce_mean(minColl)

    # Compute rotation loss: difference between predicted and target rotations
    diff = target_R[:, 6:23, :] - rotations
    square_diff = tf.square(diff)
    minRow = tf.reduce_sum(square_diff, axis=2)
    dis = tf.sqrt(1e-4 + minRow)  # Add epsilon for stability
    minColl = tf.reduce_mean(dis, axis=1)
    rotationLoss = tf.reduce_mean(minColl)

    # Calculate rigid loss as sum of translation and rotation losses
    rigidLoss = rotationLoss + translationLoss
    # rigidLoss = translationLoss  # Alternative: only translation loss
    return rigidLoss


def challismethod(X, Y, scope=None):
    """
    Computes the rotation vector aligning two sets of axes using SVD (Challis method).
    This method is used to find the optimal rotation between two 3x3 matrices (axes).
    Args:
        X: Tensor of shape [3, 3], source axes matrix.
        Y: Tensor of shape [3, 3], target axes matrix.
        scope: Optional TensorFlow scope.
    Returns:
        R_vec: Tensor, rotation vector representing the optimal rotation from X to Y.
    """
    rowsX = X.shape[0]
    colsX = X.shape[1]
    rowsY = Y.shape[0]
    colsY = Y.shape[1]
    # Transpose X for multiplication
    X_transpose = tf.transpose(X)
    # Compute cross-covariance matrix (Challis method)
    C = 1 / colsX.value * Y * X_transpose
    # Perform SVD on the covariance matrix
    with tf.device('/cpu:0'):
        U, S, V = tf.linalg.svd(C)
    V_transpose = tf.transpose(V)
    # Compute determinant to ensure right-handed coordinate system
    det = tf.linalg.det(U * V_transpose)
    # Construct correction matrix (if needed to compute rotation matrix)
    M = tf.stack((tf.constant([1, 0, 0], dtype=tf.float32),
                 tf.constant([0, 1, 0], dtype=tf.float32),
                 [0, 0, det]), axis=1)
    # Compute rotation matrix (matlab: U * M * V_transpose, here: U * V_transpose)
    R_mat = U * V_transpose
    ##R_mat = U * M *  V_transpose
    # Convert rotation matrix to rotation vector (axis-angle)
    R_vec = mat2vec(R_mat)
    return R_vec
    # Alternative: return R_mat for rotation matrix

def gramschimdit3(axes, axeinit):
    """
    Orthogonalizes a vector to a set of axes using the Gram-Schmidt process (TensorFlow version).
    This function projects axeinit onto the subspace orthogonal to the given axes.
    Args:
        axes: Tensor of shape [n, d] or [1, d], set of axes to orthogonalize against.
        axeinit: Tensor of shape [d], the vector to be orthogonalized.
    Returns:
        axe: Tensor of shape [d], the orthogonalized and normalized vector.
    """
    axe = axeinit
    if axes.shape[0] == 1:
        # If only one axis, subtract its projection from axeinit
        a = tf.reduce_sum(tf.multiply(axeinit, axes))  # Dot product
        b = tf.power(tf.norm(axes), 2)                 # Squared norm
        axe = axe - a / b
        axe = tf.multiply(axe, axes)                   # Project out the axis
    else:
        # For multiple axes, iteratively subtract projections
        for i in range(axes.shape[0]):
            a = tf.reduce_sum(tf.multiply(axeinit, axes[i]))  # Dot product
            b = tf.pow(tf.norm(axes[i]), 2)                   # Squared norm
            axe = axe - a / b
            axe = tf.multiply(axe, axes[i])                   # Project out each axis
    # Normalize the resulting vector
    return axe / tf.norm(axe)

def gramschimdit2(vectors):
    """
    Performs Gram-Schmidt orthogonalization on a list of vectors (NumPy version).
    This function constructs an orthonormal basis from the input vectors.
    Args:
        vectors: Iterable of 1D NumPy arrays, the vectors to orthogonalize.
    Returns:
        np.ndarray: Orthonormal basis vectors as a 2D NumPy array.
    """
    basis = []  # List to store orthonormal basis vectors
    for v in vectors:
        # Subtract projections onto all previous basis vectors
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        # Only add if the vector is not (numerically) zero
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))  # Normalize and add to basis
    return np.array(basis)

def gramschimdit(X):
    """
    Performs Gram-Schmidt orthogonalization on a list of vectors (pure Python version).
    This function constructs an orthogonal basis from the input vectors using explicit projection subtraction.
    Args:
        X: List of 1D vectors (lists or arrays), the vectors to orthogonalize.
    Returns:
        Y: List of orthogonalized vectors (not normalized).
    """
    Y = []  # List to store orthogonalized vectors
    for i in range(len(X)):
        temp_vec = X[i]
        # Subtract projections onto all previous basis vectors
        for inY in Y:
            proj_vec = proj(inY, X[i])  # Projection of X[i] onto inY
            # Subtract projection from temp_vec
            temp_vec = list(map(lambda x, y: x - y, temp_vec, proj_vec))
        Y.append(temp_vec)  # Add orthogonalized vector to basis
    return Y

def gs_cofficient(v1, v2):
    """
    Computes the Gram-Schmidt coefficient for projecting v2 onto v1.
    This coefficient determines the scalar multiple of v1 that best approximates v2 in the direction of v1.
    Args:
        v1: 1D array-like, the vector to project onto.
        v2: 1D array-like, the vector being projected.
    Returns:
        float: The scalar projection coefficient (dot(v2, v1) / dot(v1, v1)).
    """
    # Compute the dot product of v2 and v1 (numerator)
    numerator = np.dot(v2, v1)
    # Compute the dot product of v1 with itself (denominator)
    denominator = np.dot(v1, v1)
    # Return the projection coefficient
    return numerator / denominator

def multiply(cofficient, v):
    """
    Multiplies each element of a vector by a scalar coefficient.
    This is used in the Gram-Schmidt process to scale a vector.
    Args:
        cofficient: Scalar value to multiply each element by.
        v: 1D array-like, the vector to be scaled.
    Returns:
        list: A new list where each element is v[i] * cofficient.
    """
    # Use map to multiply each element of v by the coefficient
    a = list(map((lambda x: x * cofficient), v))
    return a

def proj(v1, v2):
    """
    Projects vector v2 onto vector v1 using the Gram-Schmidt process.
    This computes the component of v2 that lies in the direction of v1.
    Args:
        v1: 1D array-like, the vector to project onto.
        v2: 1D array-like, the vector being projected.
    Returns:
        list: The projection of v2 onto v1 as a scaled version of v1.
    """
    # Multiply v1 by the Gram-Schmidt coefficient to get the projection
    return multiply(gs_cofficient(v1, v2), v1)

def mat2vec(v1):
    """
    Converts a 3x3 rotation matrix to a rotation vector (axis-angle representation).
    This function extracts the axis and angle of rotation from a rotation matrix using the Rodrigues' formula.
    Args:
        v1: Tensor of shape [3, 3], the rotation matrix.
    Returns:
        Tensor of shape [3], the rotation vector (axis * angle).
    """
    # Compute the cosine of the rotation angle from the trace of the matrix
    input = (tf.linalg.trace(v1) - 1) / 2  # Equivalent to (np.trace(v1) - 1) / 2
    # Clamp input if needed for numerical stability (commented out)
    # if input < -1:
    #     input = -1
    # elif input > 1:
    #     input = 1
    # Compute the rotation angle (in radians)
    angle = tf.acos(input)  # Equivalent to np.arccos(input)
    if angle != 0:
        # Compute the skew-symmetric part to extract the rotation axis
        matrix = (v1 - tf.transpose(v1)) / (2 * tf.sin(angle))  # Equivalent to (v1 - v1.T) / (2 * sin(angle))
    else:
        # If angle is zero, return a zero matrix (no rotation)
        matrix = tf.zeros((3, 3), dtype=tf.float32)

    # Extract the rotation axis from the skew-symmetric matrix
    axe = tf.stack((matrix[2, 1], matrix[0, 2], matrix[1, 0]), axis=0)  # [x, y, z] components
    # The rotation vector is axis * angle
    vec = axe * angle
    return vec


@ops.RegisterGradient("Svd")
def _SvdGrad(op, grad_s, grad_u, grad_v):
    """
    Computes the gradient for the singular value decomposition (SVD) operation in TensorFlow.
    This function is registered as the gradient for the SVD op and is used for backpropagation through SVD layers.
    It handles both the compute_uv=True and compute_uv=False cases, and supports full and reduced SVD.
    Args:
        op: The SVD operation for which the gradient is computed.
        grad_s: Gradient with respect to the singular values.
        grad_u: Gradient with respect to the left singular vectors.
        grad_v: Gradient with respect to the right singular vectors.
    Returns:
        grad_a: The gradient with respect to the input matrix a.
    """

    # The derivation for the compute_uv=False case, and most of
    # the derivation for the full_matrices=True case, are in Giles' paper.
    # See also: https://j-towns.github.io/papers/svd-derivative.pdf
    # For complex SVD: https://re-ra.xyz/misc/complexsvd.pdf
    a = op.inputs[0]
    a_shape = a.get_shape().with_rank_at_least(2)
    grad_s = math_ops.cast(grad_s, a.dtype)
    grad_s_mat = array_ops.matrix_diag(grad_s)

    # Handle the case where only singular values are computed (no u, v)
    if not op.get_attr("compute_uv"):
        s, u, v = linalg_ops.svd(a, compute_uv=True)
        grad_a = math_ops.matmul(u, math_ops.matmul(grad_s_mat, v, adjoint_b=True))
        grad_a.set_shape(a_shape)
        return grad_a

    # Get SVD attributes and shapes
    full_matrices = op.get_attr("full_matrices")
    grad_u_shape = grad_u.get_shape().with_rank_at_least(2)
    grad_v_shape = grad_v.get_shape().with_rank_at_least(2)
    m = a_shape.dims[-2].merge_with(grad_u_shape[-2])
    n = a_shape.dims[-1].merge_with(grad_v_shape[-2])
    batch_shape = a_shape[:-2].merge_with(grad_u_shape[:-2]).merge_with(
        grad_v_shape[:-2])
    a_shape = batch_shape.concatenate([m, n])

    m = a_shape.dims[-2].value
    n = a_shape.dims[-1].value
    # Check for unknown matrix shapes
    if m is None or n is None:
        raise NotImplementedError(
            "SVD gradient has not been implemented for input with unknown "
            "inner matrix shape.")

    # Get SVD outputs
    s = op.outputs[0]
    u = op.outputs[1]
    v = op.outputs[2]
    s = math_ops.cast(s, a.dtype)

    use_adjoint = False
    if m > n:
        # For tall matrices, swap u and v and their gradients, and transpose at the end
        use_adjoint = True
        m, n = n, m
        u, v = v, u
        grad_u, grad_v = grad_v, grad_u

    with ops.control_dependencies([grad_s, grad_u, grad_v]):
        if full_matrices and abs(m - n) > 1:
            raise NotImplementedError(
                "svd gradient is not implemented for abs(m - n) > 1 "
                "when full_matrices is True")
        s_mat = array_ops.matrix_diag(s)
        s2 = math_ops.square(s)

        # The f matrix handles the case of repeated or zero singular values
        def safe_reciprocal(x, epsilon=1E-20):
            # Avoid division by zero for degenerate singular values
            return x * math_ops.reciprocal(x * x + epsilon)

        s_shape = array_ops.shape(s)
        f = array_ops.matrix_set_diag(
            safe_reciprocal(
                array_ops.expand_dims(s2, -2) - array_ops.expand_dims(s2, -1)
            ), array_ops.zeros_like(s))
        s_inv_mat = array_ops.matrix_diag(safe_reciprocal(s))

        v1 = v[..., :, :m]
        grad_v1 = grad_v[..., :, :m]

        # Compute intermediate products for the gradient
        u_gu = math_ops.matmul(u, grad_u, adjoint_a=True)
        v_gv = math_ops.matmul(v1, grad_v1, adjoint_a=True)

        f_u = f * u_gu
        f_v = f * v_gv

        # First term: gradient with respect to singular values and orthogonality constraints
        term1_nouv = (
                grad_s_mat + math_ops.matmul(f_u + _adjoint(f_u), s_mat) +
                math_ops.matmul(s_mat, f_v + _adjoint(f_v)))

        term1 = math_ops.matmul(u, math_ops.matmul(term1_nouv, v1, adjoint_b=True))

        if m == n:
            grad_a_before_transpose = term1
        else:
            # Second term: handles the case where m != n (rectangular matrices)
            gv1t = array_ops.matrix_transpose(grad_v1, conjugate=True)
            gv1t_v1 = math_ops.matmul(gv1t, v1)
            term2_nous = gv1t - math_ops.matmul(gv1t_v1, v1, adjoint_b=True)

            if full_matrices:
                v2 = v[..., :, m:n]
                grad_v2 = grad_v[..., :, m:n]

                v1t_gv2 = math_ops.matmul(v1, grad_v2, adjoint_a=True)
                term2_nous -= math_ops.matmul(v1t_gv2, v2, adjoint_b=True)

            u_s_inv = math_ops.matmul(u, s_inv_mat)
            term2 = math_ops.matmul(u_s_inv, term2_nous)

            grad_a_before_transpose = term1 + term2

        # Third term: handles complex-valued SVD
        if a.dtype.is_complex:
            eye = _linalg.eye(s_shape[-1], batch_shape=s_shape[:-1], dtype=a.dtype)
            l = eye * v_gv
            term3_nouv = math_ops.matmul(s_inv_mat, _adjoint(l) - l)
            term3 = 1 / 2. * math_ops.matmul(u, math_ops.matmul(term3_nouv, v1, adjoint_b=True))

            grad_a_before_transpose += term3

        # If adjoint is needed, transpose the result
        if use_adjoint:
            grad_a = array_ops.matrix_transpose(grad_a_before_transpose, conjugate=True)
        else:
            grad_a = grad_a_before_transpose

        grad_a.set_shape(a_shape)
        return grad_a


def _LeftShift(x):
    """
    Shifts the next-to-last dimension of a tensor to the left by one position, filling the rightmost position with zeros.
    This is typically used for manipulating batched matrices or tensors where shifting is needed along a specific axis.
    Args:
        x: Tensor of at least rank 3. The shift is applied to the second-to-last dimension.
    Returns:
        Tensor with the same shape as x, but with the next-to-last dimension shifted left by one and zeros padded on the right.
    Example:
        If x has shape [batch, N, D], the operation shifts elements along the N dimension to the left.
    """
    # Get the rank (number of dimensions) of the input tensor
    rank = array_ops.rank(x)
    # Create a zeros tensor for padding, shape: (rank-2, 2)
    zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
    # Define the padding: no pad for leading dims, pad 1 zero to the right of next-to-last dim, none for last dim
    pad = array_ops.concat([zeros, array_ops.constant([[0, 1], [0, 0]])], axis=0)
    # Shift the next-to-last dimension left by slicing x[..., 1:, :], then pad zeros on the right
    return array_ops.pad(x[..., 1:, :], pad)


def _RightShift(x):
    """
    Shifts the next-to-last dimension of a tensor to the right by one position, filling the leftmost position with zeros.
    This is typically used for manipulating batched matrices or tensors where shifting is needed along a specific axis.
    Args:
        x: Tensor of at least rank 3. The shift is applied to the second-to-last dimension.
    Returns:
        Tensor with the same shape as x, but with the next-to-last dimension shifted right by one and zeros padded on the left.
    Example:
        If x has shape [batch, N, D], the operation shifts elements along the N dimension to the right.
    """
    # Get the rank (number of dimensions) of the input tensor
    rank = array_ops.rank(x)
    # Create a zeros tensor for padding, shape: (rank-2, 2)
    zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
    # Define the padding: no pad for leading dims, pad 1 zero to the left of next-to-last dim, none for last dim
    pad = array_ops.concat([zeros, array_ops.constant([[1, 0], [0, 0]])], axis=0)
    # Shift the next-to-last dimension right by slicing x[..., :-1, :], then pad zeros on the left
    return array_ops.pad(x[..., :-1, :], pad)

def _adjoint(x):
    """
    Returns the adjoint (conjugate transpose) of a tensor matrix.
    For real-valued tensors, this is simply the transpose of the last two dimensions.
    For complex tensors, this would also take the complex conjugate (not shown here).
    Args:
        x: Tensor of at least rank 2, representing a matrix or batch of matrices.
    Returns:
        Tensor with the last two dimensions transposed (adjoint matrix).
    Note:
        The function currently uses tf.transpose(x), which transposes all dimensions.
        For batched matrices, tf.transpose(x, perm=[...]) should be used to only swap the last two dims.
    """
    # Transpose the input tensor. For 2D, this is the matrix transpose. For higher rank, all axes are reversed.
    adj_X = tf.transpose(x)  # For batched matrices, use tf.transpose(x, perm=[0,2,1]) to swap only last two dims
    # Alternative adjoint logic for square matrices (commented):
    # if tf.linalg.det(x) == 0:
    #     adj_X = tf.linalg.det(x)
    # else:
    #     adj_X = tf.linalg.inv(x) * tf.linalg.det(x)
    return adj_X