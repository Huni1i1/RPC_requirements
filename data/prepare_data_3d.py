# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/21_prepare_data_3d.ipynb (unless otherwise specified).

__all__ = ['parse_args', 'world_to_camera_miqus', 'runningpose_cameras_extrinsic_params', 'convert_to_camera', 'main']

# Cell
import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

# Cell
def parse_args():
    '''Parses and returns arguments for the command line interface'''
    parser = argparse.ArgumentParser(description='3D dataset creator')
    parser.add_argument(
        '-i', '--input', type=str, default='',
        metavar='PATH', help='detections directory'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='',
        metavar='PATH', help='output suffix for 3D detections'
    )
    parser.add_argument(
        '-c', '--camera',
        help='which misqus camera (1, 2, 3) to use in runningpose dataset',
        type=int,
        choices=range(1, 4)
    )

    return parser.parse_args()

# Cell
# Need to define this here aswell since relative imports are a thing
def world_to_camera_miqus(P, R, T):
  """
  Convert points from world to camera coordinates
  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.dot( P.T - T ) # rotate and translate

  return X_cam.T

runningpose_cameras_extrinsic_params = [
    {   # Miqus video camera 1
        'rotation': np.array([
            [0.735354, 0.677463, 0.017256],
            [-0.104186, 0.087855, 0.990670],
            [0.669627, -0.730291, 0.135186]
        ]),
        'translation': [6643.345215, -2730.456543, 1153.752808],
    },
    {   # Miqus video camera 2
        'rotation': np.array([
            [0.997871, -0.064902, 0.006349],
            [-0.006020, 0.005255, 0.999968],
            [-0.064933, -0.997878 , 0.004853]
        ]),
        'translation': [-697.331482, -2968.999268, 1121.579468],
    },
    {   # Miqus video camera 3
        'rotation': np.array([
            [-0.641654, 0.766908, 0.011533],
            [-0.137808, -0.130067, 0.981882],
            [0.754513, 0.628438, 0.189144]
        ]),
        'translation': [14351.271484, 3795.722412, 1504.888672],
    },
]

# Cell
def convert_to_camera(filename, cam):
    '''
    Converts the csv file with 3D world coordinates to 3D camera coordinates.

    Returns: A numpy array with shape: (num_frames, num_keypoints, dimension)
    '''
    # TODO: check the 3D data so it doesnt have a unnamed column
    print('Processing {}'.format(filename))
    # Load the 3D world coordinates data
    data_3D_world = pd.read_csv(filename)
    # Get camera parameters.
    R = runningpose_cameras_extrinsic_params[cam-1]['rotation']
    T = np.array([runningpose_cameras_extrinsic_params[cam-1]['translation']]).T
    # Extract a keypoint column and calculate it to 3D camera coordinates
    data_3D_camera = []
    for column in data_3D_world:
        col_data = data_3D_world[column].values
        x_data = col_data[0::3]
        y_data = col_data[1::3]
        z_data = col_data[2::3]
        keypdata_world = np.array([x_data, y_data, z_data]).T
        data_3D_camera.append(world_to_camera_miqus(keypdata_world, R, T))

    # Convert to a array and transpose so that it matches our 2D data input
    data_3D_camera = np.array(data_3D_camera).transpose(1, 0, 2)

    return data_3D_camera

# Cell
def main(args):
    '''
    Creates a 3D camera coordinates dataset for data collected with
    miqus cameras.

    Returns: Dictionary with all the 3D data for each run.
    '''
    if not args.input:
        print('Please specify the input directory')
        exit(0)

    if not args.output:
        print('Please specify an output suffix (e.g. detectron_pt_coco)')
        exit(0)

    if not args.camera:
        print()

    print('Parsing 3D data from', args.input)

    output = {}
    output_prefix_3d = 'data_3d_'
    file_list = glob(args.input + '/*.csv')
    for f in file_list:
        canonical_name = os.path.splitext(os.path.basename(f))[0]
        data_3D_camera = convert_to_camera(f, args.camera)
        output[canonical_name] = data_3D_camera

    print('Saving...')
    np.savez_compressed(
        output_prefix_3d + args.output, positions_3d=output
    )

# Cell
try: from nbdev.imports import IN_NOTEBOOK
except: IN_NOTEBOOK=False

if __name__ == '__main__' and not IN_NOTEBOOK:
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    args = parse_args()
    main(args)