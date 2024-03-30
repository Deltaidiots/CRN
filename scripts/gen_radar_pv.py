import os

import mmcv
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


DATA_PATH = 'data/nuScenes'
RADAR_SPLIT = 'radar_bev_filter'
OUT_PATH = 'radar_pv_filter'
info_paths = ['data/nuScenes/nuscenes_infos_train.pkl', 'data/nuScenes/nuscenes_infos_val.pkl']

# DATA_PATH = 'data/nuScenes/v1.0-test'
# RADAR_SPLIT = 'radar_bev_filter_test'
# OUT_PATH = 'radar_pv_filter_test'
# info_paths = ['data/nuScenes/nuscenes_infos_test.pkl']

MIN_DISTANCE = 0.1
MAX_DISTANCE = 100.

IMG_SHAPE = (900, 1600)

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
]


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    features,
    img_shape,
    cam_calibrated_sensor,
    cam_ego_pose,
):
    """
    Projects radar points onto camera image plane, keeping depth information.

    This function transforms radar point cloud coordinates to the camera's
    coordinate system and then projects them onto the camera's image plane,
    preserving the depth information of each point relative to the camera.

    Args:
        pc (LidarPointCloud): Point cloud to be projected, wrapped in LidarPointCloud for compatibility.
        features (np.ndarray): Additional features of the points such as RCS (Radar Cross Section),
                               velocities, etc., that need to be preserved after projection.
        img_shape (tuple): Shape of the camera image to which points are being projected, defined as (height, width) i.e (900, 1600) .
        cam_calibrated_sensor (dict): Calibration data of the camera sensor, including intrinsic matrix.
        cam_ego_pose (dict): The ego pose of the camera at the time of the snapshot.

    Returns:
        tuple: Two numpy arrays representing the pixel coordinates (u,v) and depth (d) of the radar points on the image,
               and the radar features with depth concatenated.
    """
    pc = LidarPointCloud(pc)

    # Third step: transform from global into the ego vehicle frame (camera based).
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera view point.
    #This aligns the points precisely with the camera's own coordinate system, positioning them as they would appear from the camera's viewpoint.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    features = np.concatenate((depths[:, None], features), axis=1)
    #features shape: (n, 5) [depth, rcs, vx_comp, vy_comp, (dummy field for sweep info)]

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
    mask = np.logical_and(mask, depths > MIN_DISTANCE)
    mask = np.logical_and(mask, depths < MAX_DISTANCE)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img_shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img_shape[0] - 1)
    points = points[:, mask]
    #points shape: (3, n) [u, v, depth]
    features = features[mask]
    #features shape: (n, 5) [depth, rcs, vx_comp, vy_comp, (dummy field for sweep info)]

    return points, features


def worker(info):
    """
    Processes a single sample from the NuScenes dataset to project radar data onto camera images.

    This function reads radar data processed in a bird's eye view (BEV) format, transforms it to align with
    the vehicle's and camera's coordinate frames, and then projects it onto the image plane of multiple cameras.
    The result is saved as radar perspective view (RPV) data, which combines radar information with camera viewpoints.

    Args:
        info (dict): Metadata and file paths for processing a single sample. It includes paths to radar data,
                     calibration information for lidar and cameras, and ego poses.

    Side effects:
        Saves the processed RPV data to files corresponding to each camera's viewpoint in the dataset.
    """

    radar_file_name = os.path.split(info['lidar_infos']['LIDAR_TOP']['filename'])[-1]
    # load the radar points generated by gen_radar_bev.py
    points = np.fromfile(os.path.join(DATA_PATH, RADAR_SPLIT, radar_file_name),
                         dtype=np.float32,
                         count=-1).reshape(-1, 7)
    #shape of points: (n, 7) [x, y, z, rcs, vx_comp, vy_comp, (dummy field for sweep info)]
     
    # load the calibration information for the lidar sensor 
    lidar_calibrated_sensor = info['lidar_infos'][lidar_key][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][lidar_key]['ego_pose']

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.

    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    pc = LidarPointCloud(points[:, :4].T)  # use 4 dims for code compatibility
    
    features = points[:, 3:]
    #features shape: (n, 4) [rcs, vx_comp, vy_comp, (dummy field for sweep info)]

    # transform the point cloud to the ego vehicle frame (lidar based) for the timestamp of sample
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    for i, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        pts_img, features_img = map_pointcloud_to_image(
            pc.points.copy(), features.copy(), IMG_SHAPE, cam_calibrated_sensor, cam_ego_pose)

        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        np.concatenate([pts_img[:2, :].T, features_img],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(DATA_PATH, OUT_PATH,
                                        f'{file_name}.bin'))
    # plt.savefig(f"{sample_idx}")


if __name__ == '__main__':
    mmcv.mkdir_or_exist(os.path.join(DATA_PATH, OUT_PATH))
    for info_path in info_paths:
        infos = mmcv.load(info_path)
        for info in tqdm(infos):
            worker(info)
