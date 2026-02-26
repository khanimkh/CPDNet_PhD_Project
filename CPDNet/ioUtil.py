# =======================
# ioUtil.py: Utility functions for data loading, shuffling, and point cloud export
# =======================
# This module provides helper functions to:
#   - Load training, evaluation, and test datasets from HDF5 files
#   - Shuffle dataset examples for training
#   - Export point clouds to PLY format for visualization
# The Examples namedtuple is used to organize all relevant data fields for each dataset.

import os                   # File and directory operations
import sys                  # System-specific parameters and functions
import numpy as np          # Numerical operations and array handling
import h5py                 # HDF5 file reading
import collections          # For namedtuple

# Set up base directory for module imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Named tuple to store all relevant fields for a dataset
Examples = collections.namedtuple("Examples", "names, pointsets_A, pointsets_B, pointsets_meta_A, pointsets_T_A, pointsets_R_A, pointsets_T_B, pointsets_R_B")

# =======================
# Shuffle the order of examples in a dataset for training
# =======================
def shuffle_examples(data):
    """
    Randomly shuffle the order of examples in the dataset.
    Args:
        data (Examples): Namedtuple containing all dataset arrays.
    Returns:
        Examples: Shuffled dataset.
    """
    idx = np.arange(data.names.shape[0])
    np.random.shuffle(idx)
    return Examples(
        names=data.names[idx, ...],
        pointsets_A=data.pointsets_A[idx, ...],
        pointsets_B=data.pointsets_B[idx, ...],
        pointsets_meta_A=data.pointsets_meta_A[idx, ...],
        pointsets_T_A=data.pointsets_T_A[idx, ...],
        pointsets_R_A=data.pointsets_R_A[idx, ...],
        pointsets_T_B=data.pointsets_T_B[idx, ...],
        pointsets_R_B=data.pointsets_R_B[idx, ...],
    )

# =======================
# Load all examples from an HDF5 file (for training)
# =======================
def load_examples(h5_filename, fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname):
    """
    Load all examples from an HDF5 file for training.
    Args:
        h5_filename (str): Path to HDF5 file.
        fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname (str): Keys for data fields.
    Returns:
        Examples: Loaded dataset.
    """
    f = h5py.File(h5_filename)
    pointsets_A = f[fieldname_A][:, :, :]
    pointsets_B = f[fieldname_B][:, :, :]
    pointsets_meta_A = f[meta_A][:, :, :]
    pointsets_T_A = f[pre_T][:, :, :]
    pointsets_R_A = f[pre_R][:, :, :]
    pointsets_T_B = f[post_T][:, :, :]
    pointsets_R_B = f[post_R][:, :, :]
    names = f[fieldname_modelname][:]
    return Examples(
        names=names,
        pointsets_A=pointsets_A,
        pointsets_B=pointsets_B,
        pointsets_meta_A=pointsets_meta_A,
        pointsets_T_A=pointsets_T_A,
        pointsets_R_A=pointsets_R_A,
        pointsets_T_B=pointsets_T_B,
        pointsets_R_B=pointsets_R_B,
    )

# =======================
# Load a subset of examples for evaluation (validation set)
# =======================
def load_eval_examples(h5_filename, fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname):
    """
    Load a subset of examples from an HDF5 file for evaluation (validation set).
    Args:
        h5_filename (str): Path to HDF5 file.
        ... (other keys as above)
    Returns:
        Examples: Loaded validation dataset (typically a slice of the full set).
    """
    f = h5py.File(h5_filename)
    pointsets_A = f[fieldname_A][10:20, :, :]
    pointsets_B = f[fieldname_B][10:20, :, :]
    pointsets_meta_A = f[meta_A][10:20, :, :]
    pointsets_T_A = f[pre_T][10:20, :, :]
    pointsets_R_A = f[pre_R][10:20, :, :]
    pointsets_T_B = f[post_T][10:20, :, :]
    pointsets_R_B = f[post_R][10:20, :, :]
    names = f[fieldname_modelname][10:20]
    return Examples(
        names=names,
        pointsets_A=pointsets_A,
        pointsets_B=pointsets_B,
        pointsets_meta_A=pointsets_meta_A,
        pointsets_T_A=pointsets_T_A,
        pointsets_R_A=pointsets_R_A,
        pointsets_T_B=pointsets_T_B,
        pointsets_R_B=pointsets_R_B,
    )

# =======================
# Load a subset of examples for testing (test set)
# =======================
def load_test_examples(h5_filename, fieldname_A, fieldname_B, meta_A, pre_T, pre_R, post_T, post_R, fieldname_modelname):
    """
    Load a subset of examples from an HDF5 file for testing (test set).
    Args:
        h5_filename (str): Path to HDF5 file.
        ... (other keys as above)
    Returns:
        Examples: Loaded test dataset (typically a slice of the full set).
    """
    f = h5py.File(h5_filename)
    pointsets_A = f[fieldname_A][10:20, :, :]
    pointsets_B = f[fieldname_B][10:20, :, :]
    pointsets_meta_A = f[meta_A][10:20, :, :]
    pointsets_T_A = f[pre_T][10:20, :, :]
    pointsets_R_A = f[pre_R][10:20, :, :]
    pointsets_T_B = f[post_T][10:20, :, :]
    pointsets_R_B = f[post_R][10:20, :, :]
    names = f[fieldname_modelname][10:20]
    return Examples(
        names=names,
        pointsets_A=pointsets_A,
        pointsets_B=pointsets_B,
        pointsets_meta_A=pointsets_meta_A,
        pointsets_T_A=pointsets_T_A,
        pointsets_R_A=pointsets_R_A,
        pointsets_T_B=pointsets_T_B,
        pointsets_R_B=pointsets_R_B,
    )

# =======================
# Export point clouds to PLY files for visualization
# =======================
def output_point_cloud_ply(xyzs, names, output_dir, foldername):
    """
    Export a batch of point clouds to PLY files for visualization or debugging.
    Args:
        xyzs (np.ndarray): Array of point clouds (batch or single).
        names (list/array): List of names for each point cloud.
        output_dir (str): Output directory for results.
        foldername (str): Subfolder name for this export.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plydir = output_dir + '/' + foldername
    if not os.path.exists(plydir):
        os.mkdir(plydir)
    numFiles = len(names)
    for fid in range(numFiles):
        print('write: ' + plydir + '/' + names[fid] + '.ply')
        with open(plydir + '/' + names[fid] + '.ply', 'w') as f:
            if len(xyzs.shape) == 3:
                # Batch of point clouds
                pn = xyzs.shape[1]
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write('element vertex %d\n' % (pn))
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('end_header\n')
                for i in range(pn):
                    f.write('%f %f %f\n' % (xyzs[fid][i][0], xyzs[fid][i][1], xyzs[fid][i][2]))
            else:
                # Single point cloud
                pn = xyzs.shape[0]
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write('element vertex %d\n' % (pn))
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('end_header\n')
                for i in range(pn):
                    f.write('%f %f %f\n' % (xyzs[i][0], xyzs[i][1], xyzs[i][2]))