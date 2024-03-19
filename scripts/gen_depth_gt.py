import os
from multiprocessing import Pool

import mmcv
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):
    """
    Map a point cloud to an image using camera calibration and ego pose.

    This function takes a point cloud from a LiDAR sensor and projects it onto the 2D image captured by a camera. It accounts for the position and orientation of both the vehicle (ego pose) and the camera (calibrated sensor data). The function performs several coordinate transformations to align the 3D point cloud with the 2D camera image plane and then projects the points onto the image. Points that are too close, behind the camera, or fall outside the image frame are filtered out.

    Args:
        pc (np.ndarray): The point cloud data as a Numpy array with shape (N, 4), where N is the number of points.
        im (np.ndarray): The camera image data as a 2D Numpy array.
        cam_calibrated_sensor (dict): The calibrated sensor metadata for the camera, including translation and rotation.
        cam_ego_pose (dict): The ego pose metadata for the vehicle, including translation and rotation.
        min_dist (float, optional): Minimum distance threshold for points to be considered valid. Defaults to 0.0, allowing all points.

    Returns:
        tuple: A tuple containing two elements:
            - np.ndarray: The 2D points projected onto the image plane, with shape (2, M), where M is the number of valid points.
            - np.ndarray: The corresponding depth values for the projected points as a 1D Numpy array with shape (M,).
    
    Example:
        # Example usage of the function with a sample LiDAR point cloud and camera image
        projected_points, depths = map_pointcloud_to_image(
            pc=np.random.rand(100, 4),
            im=np.random.rand(1200, 1600, 3),
            cam_calibrated_sensor={'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0], 'camera_intrinsic': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            cam_ego_pose={'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]},
            min_dist=1.0
        )
    """
    pc = LidarPointCloud(pc)

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


data_root = 'data/nuScenes'
INFO_PATHS = ['data/nuScenes/nuscenes_infos_train.pkl',
              'data/nuScenes/nuscenes_infos_val.pkl']

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]


def worker(info):
    """
    Process lidar and camera data to generate depth ground truth images.

    Args:
        info (dict): A dictionary containing lidar and camera information.

    Returns:
        None
    """
    lidar_path = info['lidar_infos'][lidar_key]['filename']
    points = np.fromfile(os.path.join(data_root, lidar_path),
                         dtype=np.float32,
                         count=-1).reshape(-1, 5)[..., :4]
    lidar_calibrated_sensor = info['lidar_infos'][lidar_key][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][lidar_key]['ego_pose']

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(points.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    for i, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        img = mmcv.imread(
            os.path.join(data_root, info['cam_infos'][cam_key]['filename']))
        pts_img, depth = map_pointcloud_to_image(
            pc.points.copy(), img, cam_calibrated_sensor, cam_ego_pose)
        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        np.concatenate([pts_img[:2, :].T, depth[:, None]],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(data_root, 'depth_gt',
                                        f'{file_name}.bin'))
        # plot mapped points on image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img)
        # plt.scatter(pts_img[0], pts_img[1], c=depth, s=5)
        # plt.colorbar()
        # plt.title(f"{i}")
        # plt.show()
    # plt.savefig(f"{sample_idx}")


if __name__ == '__main__':
    po = Pool(24)
    mmcv.mkdir_or_exist(os.path.join(data_root, 'depth_gt'))
    for info_path in INFO_PATHS:
        infos = mmcv.load(info_path)
        for info in infos:
            po.apply_async(func=worker, args=(info, ))
    po.close()
    po.join()
