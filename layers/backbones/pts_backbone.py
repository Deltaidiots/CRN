import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import bias_init_with_prob
from mmcv.ops import Voxelization
from mmdet3d.models import builder
import numpy as np
import matplotlib.pyplot as plt

class PtsBackbone(nn.Module):
    """
    Pillar Feature Net for processing point clouds.

    This network module is designed to process point cloud data, preparing the pillar
    features and performing a forward pass through PFNLayers to generate context features
    and occupancy grids.

    Attributes:
        pts_voxel_layer: A Voxelization layer for converting point clouds into voxels.
        pts_voxel_encoder: A voxel encoder module for encoding voxel features.
        pts_middle_encoder: A middle encoder module for processing encoded voxel features.
        pts_backbone: A backbone network for feature extraction from processed voxels.
        pts_neck (optional): A neck module for further processing of backbone features.
        return_context (bool): Whether to return context features. Default to True.
        return_occupancy (bool): Whether to return occupancy grid. Default to True.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_neck,
                 return_context=True,
                 return_occupancy=True,
                 **kwargs,
                 ):
        super(PtsBackbone, self).__init__()

        self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        self.pts_backbone = builder.build_backbone(pts_backbone)
        self.return_context = return_context
        self.return_occupancy = return_occupancy
        mid_channels = pts_backbone['out_channels'][-1]
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
            mid_channels = sum(pts_neck['out_channels'])
        else:
            self.pts_neck = None

        if self.return_context:
            if 'out_channels_pts' in kwargs:
                out_channels = kwargs['out_channels_pts']
            else:
                out_channels = 80
            self.pred_context = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

        if self.return_occupancy:
            self.pred_occupancy = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels//2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels//2,
                          1,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

            if 'occupancy_init' in kwargs:
                occupancy_init = kwargs['occupancy_init']
            else:
                occupancy_init = 0.01
            self.pred_occupancy[-1].bias.data.fill_(bias_init_with_prob(occupancy_init))
    def visualize_voxels_with_points_distinct(voxels, coors, voxel_size, point_cloud_range):
        voxels_np = voxels.cpu().numpy()
        coors_np = coors.cpu().numpy()

        # Determine the grid bounds
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range
        x_range = np.arange(x_min, x_max, voxel_size[0])
        y_range = np.arange(y_min, y_max, voxel_size[1])

        plt.figure(figsize=(12, 12))
        for x in x_range:
            plt.axvline(x, color='k', linestyle='--', linewidth=0.5)
        for y in y_range:
            plt.axhline(y, color='k', linestyle='--', linewidth=0.5)

        # Plot each voxel's points
        for idx, coor in enumerate(coors_np):
            voxel_points = voxels_np[idx]
            for point_idx, point in enumerate(voxel_points):
                if point[0] != -999:  # Filter out padding values
                    # Convert voxel coordinates to plot coordinates
                    plot_x = (coor[2] * voxel_size[0]) + x_min + voxel_size[0] / 2
                    plot_y = (coor[1] * voxel_size[1]) + y_min + voxel_size[1] / 2
                    # Add a small random offset to each point's position within the voxel
                    offset_x = np.random.uniform(-voxel_size[0] / 2, voxel_size[0] / 2)
                    offset_y = np.random.uniform(-voxel_size[1] / 2, voxel_size[1] / 2)
                    plot_x += offset_x
                    plot_y += offset_y
                    plt.plot(plot_x, plot_y, 'ro')

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.xlabel('Width')
        plt.ylabel('Depth')
        plt.title('Voxelized Radar Points with Grid')
        plt.show()


    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample/sweep.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        batch_size, _, _ = points.shape
        points_list = [points[i] for i in range(batch_size)]

        for res in points_list:
            # res shape is (P, F) e.g (1536 or x, 5)
            # plot_original_radar_points(res)
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            # visualize_voxels_with_points(res_voxels, res_coors, voxel_size=[8, 0.4, 2], point_cloud_range=[0, 2.0, 0, 704, 58.0, 2])
            # visualize_voxels_with_points_distinct(res_voxels, res_coors, voxel_size=[8, 0.4, 2], point_cloud_range=[0, 2.0, 0, 704, 58.0, 2])
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    def visualize_backbone_feature_maps(self, feature_maps):
        # Visualize backbone feature maps
        # Example code to plot intermediate feature maps
        num_maps = feature_maps.size(1)  # Get the number of feature maps
        fig, axes = plt.subplots(1, num_maps, figsize=(12, 4))
        for i in range(num_maps):
            axes[i].imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Feature Map {i+1}')
        plt.tight_layout()
        plt.show()
    def visualize_context_features(self, context_features):
        # Visualize context features
        # Example code to plot feature maps
        plt.figure(figsize=(8, 8))
        plt.imshow(context_features.squeeze().cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('Context Features')
        plt.show()

    def visualize_occupancy_grids(self, occupancy_grids):
        # Visualize occupancy grids
        # Example code to plot occupancy grids
        plt.figure(figsize=(8, 8))
        plt.imshow(occupancy_grids.squeeze().cpu().numpy(), cmap='binary', vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Occupancy Grids')
        plt.show()

    def _forward_single_sweep(self, pts):
        """
        Perform a forward pass for a single sweep.

        Args:
            pts (torch.Tensor): Input points tensor of shape (B, N, P, F), where
                B is the batch size, N is the number of points, P is the number of
                points, and F is the number of features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the context tensor
            of shape (B, 1, C) and the occupancy tensor of shape (B, 1, P), where C
            is the number of context features and P is the number of voxels.

        """
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        B, N, P, F = pts.shape
        batch_size = B * N
        pts = pts.contiguous().view(B*N, P, F)
        # points shape is (B*N, P, F) e.g (6, 1536 or x, 5)
        voxels, num_points, coors = self.voxelize(pts)
        # voxels shape is (num_point, self.pts_voxel_layer.max_num_points, F) e.g (170,8,5)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['pts_voxelize'].append(t1.elapsed_time(t2))

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        # voxel_features shape is (num_point, self.pts_voxel_encoder.num_features) e.g (170, 64)
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # x shape is (B*N, C, H, W) e.g (6, 64, self.pts_middle_encoder.output_shape[0], self.pts_middle_encoder.output_shape[1])
        x = self.pts_backbone(x)
        # x shape is tuple of length 3 where each element is (B*N, C, H, W) e.g 6, 64, 140, 88),(6, 128, 70, 44),(6, 256, 35, 22)
        if self.pts_neck is not None:
            x = self.pts_neck(x)
            # x[0] shape is (B*N, C(sum(self.pts_neck.out_channels)), H, W) e.g (6, 384, 35, 22)
        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['pts_backbone'].append(t2.elapsed_time(t3))

        x_context = None
        x_occupancy = None
        if self.return_context:
            x_context = self.pred_context(x[-1]).unsqueeze(1)
            # x_context shape is (B*N, 1, C(self.out_channels_pts), H, W) e.g (6, 1, 80, 70, 44)
        if self.return_occupancy:
            x_occupancy = self.pred_occupancy(x[-1]).unsqueeze(1).sigmoid()
            # x_occupancy shape is (B*N, 1, 1, H, W) e.g (6, 1,1, 70, 44)

        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['pts_head'].append(t3.elapsed_time(t4))

        return x_context, x_occupancy

    def forward(self, ptss, times=None):
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, _, _ = ptss.shape

        key_context, key_occupancy = self._forward_single_sweep(ptss[:, 0, ...])
        
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['pts'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            return key_context, key_occupancy, self.times

        context_list = [key_context]
        occupancy_list = [key_occupancy]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                context, occupancy = self._forward_single_sweep(ptss[:, sweep_index, ...])
                context_list.append(context)
                occupancy_list.append(occupancy)

        ret_context = None
        ret_occupancy = None
        if self.return_context:
            ret_context = torch.cat(context_list, 1)
        if self.return_occupancy:
            ret_occupancy = torch.cat(occupancy_list, 1)
        return ret_context, ret_occupancy, self.times
