import mmcv

from models.base_bev_depth import BaseBEVDepth
from layers.backbones.rvt_lss_fpn import RVTLSSFPN
from layers.backbones.pts_backbone import PtsBackbone
from layers.fuser.multimodal_feature_aggregation import MFAFuser
from layers.heads.bev_depth_head_det import BEVDepthHead

logger = mmcv.utils.get_logger('mmdet')
logger.setLevel('WARNING')

__all__ = ['CameraRadarNetDet']

import torch
import matplotlib.pyplot as plt
import numpy as np

def imshow(img, title=None):
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_input_images(sweep_imgs, sweep_idx=0, cam_idx=0):
    """
    Visualizes the input images for a given sweep and camera index.
    """
    # Assuming sweep_imgs shape is [B, num_sweeps, num_cams, C, H, W]
    img = sweep_imgs[0, sweep_idx, cam_idx].cpu()
    imshow(img, title=f"Input Image - Sweep {sweep_idx}, Camera {cam_idx}")

def visualize_pts_occupancy(pts_context, sweep_idx=0, channel_idx=None):
    """
    Visualizes the point cloud context for a given sweep and optionally a specific channel.
    """
    # Assuming pts_context shape is [num_cams, num_sweeps, C, D, W]
    context = pts_context[:, sweep_idx].cpu()
    if channel_idx is not None:
        context = context[:, channel_idx, :, :]  # Select specific channel
    avg_context = torch.mean(context, dim=0)  # Average across cameras for visualization
    imshow(avg_context, title=f"Point Cloud Context - Sweep {sweep_idx}" + (f", Channel {channel_idx}" if channel_idx is not None else ""))

def select_sweep(feats, sweep_idx=None):
    if sweep_idx is None:
        # Average over sweeps if not specified
        return torch.mean(feats, dim=1, keepdim=True)
    else:
        return feats[:, sweep_idx]

def select_channel(feature_maps, channel_idx=None):
    if channel_idx is None:
        # Average across channels if not specified
        return torch.mean(feature_maps, dim=1, keepdim=True)
    else:
        return feature_maps[:, channel_idx]

def visualize_feature_maps(feats, sweep_idx=None, channel_idx=None, title_prefix=""):
    feats = select_sweep(feats, sweep_idx)
    feats = select_channel(feats, channel_idx)
    feature_map = feats[0].cpu().numpy().squeeze()

    plt.figure(figsize=(10, 5))
    plt.imshow(feature_map, cmap='viridis')
    title = f"{title_prefix}Feature Map"
    if sweep_idx is not None:
        title += f" - Sweep {sweep_idx}"
    if channel_idx is not None:
        title += f" - Channel {channel_idx}"
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_first_sweep_images(sweep_imgs):
    """
    Visualize the first sweep images for all the cameras.

    Args:
        sweep_imgs (torch.Tensor): Input images tensor of shape (B, num_sweeps, num_cams, C, H, W).
    """
    # Get the first sweep images for all the cameras
    first_sweep_imgs = sweep_imgs[0, 0].detach().cpu().numpy()

    num_cams = first_sweep_imgs.shape[0]

    # Create a subplot for each camera
    fig, axs = plt.subplots(1, num_cams, figsize=(10, 5))

    for i in range(num_cams):
        # Get the image for the current camera
        img = first_sweep_imgs[i]

        # If the image has multiple channels, transpose it to (H, W, C) format for visualization
        if img.shape[0] > 1:
            img = img.transpose((1, 2, 0))

        # Display the image
        axs[i].imshow(img)
        axs[i].set_title(f'Camera {i+1}')
        axs[i].axis('off')

    plt.tight_layout() 
    #plt.show()
    plt.close(fig)
    return fig

def visualize_feats_mean(feats):
    """
    Visualize the feats tensor.

    Args:
        feats (torch.Tensor): Input feats tensor of shape (B, num_sweeps, C, H, W).
    """
    # Get the feats for the first sweep
    first_sweep_feats = feats[0, 0].detach().cpu().numpy()

    # If the feats has multiple channels, take the mean across the channels
    if first_sweep_feats.shape[0] > 1:
        first_sweep_feats = first_sweep_feats.mean(axis=0)

    # Display the feats
    plt.figure(figsize=(10, 5))
    plt.imshow(first_sweep_feats, cmap='viridis')
    plt.title('Feats for the first sweep')
    plt.axis('off')
    plt.colorbar()
    plt.show()

def visualize_fused(fused):
    """
    Visualize the fused tensor.

    Args:
        fused (torch.Tensor): Input fused tensor of shape (B, C, H, W).
    """
    # Get the fused tensor for the first example
    fused_example = fused[0].detach().cpu().numpy()

    # If the fused tensor has multiple channels, take the mean across the channels
    if fused_example.shape[0] > 1:
        fused_example = fused_example.mean(axis=0)

    # Display the fused tensor
    plt.figure(figsize=(10, 5))
    plt.imshow(fused_example, cmap='viridis')
    plt.title('Fused tensor for the first example')
    plt.axis('off')
    plt.colorbar()
    plt.show()

def visualize_attention_over_bev(attn_maps, bev_image, modality=0):

    bev_image = bev_image[0].detach().cpu().numpy()
    attn_maps = attn_maps.cpu().numpy()
    # If the fused tensor has multiple channels, take the mean across the channels
    if bev_image.shape[0] > 1:
        bev_image = bev_image.mean(axis=0)
    #attn_maps: [bs, num_sweeps, num_heads, num_modalities, num_samples, h, w]
    
    # Selecting the first sweep and first modality of attention maps
    attn_sweep_modality = attn_maps[0, 0, :, modality, :, :, :]
    # Averaging over num_heads and num_samples
    attn_avg = attn_sweep_modality.mean(axis=(0, 1))

    plt.figure(figsize=(10, 5))
    plt.imshow(bev_image, cmap='viridis', alpha=0.4)  # Assuming bev_image is a numpy array or similar
    plt.imshow(attn_avg, cmap='jet', alpha=0.9)
    plt.title("Attention Overlay on BEV")
    plt.axis('off')
    plt.colorbar()
    plt.show()

def visualize_attention_over_and_with_bev(attn_maps, bev_image, modality=0):
    bev_image = bev_image[0].detach().cpu().numpy()
    attn_maps = attn_maps.cpu().numpy()

    if bev_image.shape[0] > 1:
        bev_image = bev_image.mean(axis=0)
    
    attn_sweep_modality = attn_maps[0, 0, :, modality, :, :, :]
    attn_avg = attn_sweep_modality.mean(axis=(0, 1))

    # Create a subplot for the BEV image and the attention overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Display BEV image
    im1 = ax1.imshow(bev_image, cmap='viridis', alpha=0.8)
    ax1.set_title("BEV Image")
    ax1.axis('off')
    # Create colorbar for BEV image on the left
    fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

    # Display Attention Overlay
    im2 = ax2.imshow(bev_image, cmap='viridis', alpha=0.4)  # BEV as a base
    im3 = ax2.imshow(attn_avg, cmap='jet', alpha=0.9)  # Attention map overlay
    ax2.set_title("Attention Overlay on BEV")
    ax2.axis('off')
    # Create colorbar for attention overlay on the right
    fig.colorbar(im3, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

    # Show the plot
    plt.tight_layout()
    plt.show()
    

# Call the function with your data
# visualize_attention_over_and_with_bev(attn_maps, fused, 0) <- Replace with your tensor names


def plot_attention_maps(attn_maps_tensor, modality=0):
    """
    Plots attention maps for the selected modality.

    Parameters:
    attn_maps_tensor (torch.Tensor): Attention maps tensor with shape (bs, num_sweeps, num_heads, num_modalities, num_samples, h, w)
    modality (int): The modality index to plot.
    """
    # Ensure the tensor is on CPU and converted to numpy
    attn_maps = attn_maps_tensor.cpu().numpy()
    
    # Ensure modality index is within the range of available modalities
    if modality < 0 or modality >= attn_maps.shape[3]:
        raise ValueError("Modality index out of range")

    # Select the attention maps for the first sweep and the chosen modality
    # and average across heads and sampling points
    attn_maps_modality = attn_maps[0, 0, :, modality, :, :, :].mean(axis=(0, 1))

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.imshow(attn_maps_modality, cmap='viridis')
    plt.title(f"Attention Map for Modality {modality}")
    plt.colorbar()
    plt.axis('off')  # Hide the axis
    plt.show()

# Example usage:
# Assume attn_maps_tensor is a PyTorch tensor with the shape mentioned above.
# plot_attention_maps(attn_maps, modality=0) # Plot the first modality


def visualize_all_components(fused, feats, attn_maps, modality_list):
    fused = fused[0].detach().cpu().numpy().mean(axis=0)
    feats = feats[0, 0].detach().cpu().numpy().mean(axis=0)
    attn_maps = attn_maps.cpu().numpy()

    num_plots = 2 + 2 * len(modality_list)  # fused, feats, and each modality for attn_maps and overlay
    num_cols = 2  # Columns: fused, feats, and attention maps for each modality
    num_rows = (num_plots + 1) // num_cols  # Calculate rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axs = axs.ravel()  # Flatten the array of axes

    # Plot fused image
    axs[0].imshow(fused, cmap='viridis')
    axs[0].set_title('Fused Image')
    axs[0].axis('off')
    fig.colorbar(axs[0].images[0], ax=axs[0], fraction=0.046, pad=0.04)

    # Plot feats image
    axs[1].imshow(feats, cmap='viridis')
    axs[1].set_title('Feats Image')
    axs[1].axis('off')
    fig.colorbar(axs[1].images[0], ax=axs[1], fraction=0.046, pad=0.04)

    plot_idx = 2  # Starting index for subsequent plots
    for modality in modality_list:
        attn_sweep_modality = attn_maps[0, 0, :, modality, :, :, :].mean(axis=(0, 1))

        # Plot attention map
        axs[plot_idx].imshow(attn_sweep_modality, cmap='viridis', alpha=0.9)
        axs[plot_idx].set_title(f'Attention Map Modality {modality}')
        axs[plot_idx].axis('off')
        fig.colorbar(axs[plot_idx].images[0], ax=axs[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1

        # Plot attention overlay
        overlay = axs[plot_idx].imshow(feats, cmap='gray', alpha=0.8)  # base image
        overlay_attn = axs[plot_idx].imshow(attn_sweep_modality, cmap='jet', alpha=0.5)  # attention overlay
        axs[plot_idx].set_title(f'Attention Overlay Modality {modality}')
        axs[plot_idx].axis('off')

        # Add a colorbar for the overlay. The colorbar will correspond to the attention overlay.
        fig.colorbar(overlay_attn, ax=axs[plot_idx], fraction=0.046, pad=0.04)

        plot_idx += 1

    plt.tight_layout()
    #plt.show()

    #plt.close(fig)
    return fig
# Call the function with your tensors
# visualize_all_components(fused, feats, attn_maps, [0, 1])

class CameraRadarNetDet(BaseBEVDepth):
    """Source code of `CRN`, `https://arxiv.org/abs/2304.00670`.

    Args:
        backbone_img_conf (dict): Config of image backbone.
        backbone_pts_conf (dict): Config of point backbone.
        fuser_conf (dict): Config of BEV feature fuser.
        head_conf (dict): Config of head.
    """

    def __init__(self, backbone_img_conf, backbone_pts_conf, fuser_conf, head_conf):
        super(BaseBEVDepth, self).__init__()
        self.backbone_img = RVTLSSFPN(**backbone_img_conf)
        self.backbone_pts = PtsBackbone(**backbone_pts_conf)
        self.fuser = MFAFuser(**fuser_conf)
        self.head = BEVDepthHead(**head_conf)

        self.radar_view_transform = backbone_img_conf['radar_view_transform']

        # inference time measurement
        self.idx = 0
        self.times_dict = {
            'img': [],
            'img_backbone': [],
            'img_dep': [],
            'img_transform': [],
            'img_pool': [],

            'pts': [],
            'pts_voxelize': [],
            'pts_backbone': [],
            'pts_head': [],

            'fusion': [],
            'fusion_pre': [],
            'fusion_layer': [],
            'fusion_post': [],

            'head': [],
            'head_backbone': [],
            'head_head': [],
        }

    def forward(self,
                sweep_imgs,
                mats_dict,
                sweep_ptss=None,
                is_train=False
                ):
        """Forward function for BEVDepth

        Args:
            sweep_imgs (Tensor): Input images.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            sweep_ptss (Tensor): Input points.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if is_train:
            self.time = None

            ptss_context, ptss_occupancy, _ = self.backbone_pts(sweep_ptss)
            feats, depth, _ = self.backbone_img(sweep_imgs,
                                                mats_dict,
                                                ptss_context,
                                                ptss_occupancy,
                                                return_depth=True)
            fused, _ = self.fuser(feats)
            preds, _ = self.head(fused)
            return preds, depth
        else:
            if self.idx < 100:  # skip few iterations for warmup
                self.times = None
            elif self.idx == 100:
                self.times = self.times_dict
            # write a function to visuazlie the first sweep images , it should plot for all the cameras

 
            ptss_context, ptss_occupancy, self.times = self.backbone_pts(sweep_ptss,
                                                                         times=self.times)
            # shape of ptss_context: (B, num_sweeps, C, D, W)
            # shape of ptss_occupancy: (B, num_sweeps, 1, D, W)
            feats, self.times = self.backbone_img(sweep_imgs,   # sweep_imgs: (B, num_sweeps,num_cams,  C, H, W)
                                                  mats_dict,
                                                  ptss_context,
                                                  ptss_occupancy,
                                                  times=self.times)
            # shape of feats: (B, num_sweeps, C, H, W) converted to bev grid of 128x128 here c includes image plus depth
            fused, self.times, attn_maps = self.fuser(feats, times=self.times) 
            # shape of fused: (B, C, H, W) C is embedding size & attn_maps: [bs, num_sweeps, num_heads, num_modalities, num_samples, h, w]
            fused_feat_attn_fig = visualize_all_components(fused, feats, attn_maps, [0, 1])
            sweep_fig = visualize_first_sweep_images(sweep_imgs)
            # save figure in where idx is the index of the iteration
            fused_feat_attn_fig.savefig(f'/home/asad/data/fused_feat_attn_{self.idx}.png')
            sweep_fig.savefig(f'/home/asad/data/sweep_{self.idx}.png')
            
            preds, self.times = self.head(fused, times=self.times)

            if self.idx == 1000:
                time_mean = {}
                for k, v in self.times.items():
                    time_mean[k] = sum(v) / len(v)
                print('img: %.2f' % time_mean['img'])
                print('  img_backbone: %.2f' % time_mean['img_backbone'])
                print('  img_dep: %.2f' % time_mean['img_dep'])
                print('  img_transform: %.2f' % time_mean['img_transform'])
                print('  img_pool: %.2f' % time_mean['img_pool'])
                print('pts: %.2f' % time_mean['pts'])
                print('  pts_voxelize: %.2f' % time_mean['pts_voxelize'])
                print('  pts_backbone: %.2f' % time_mean['pts_backbone'])
                print('  pts_head: %.2f' % time_mean['pts_head'])
                print('fusion: %.2f' % time_mean['fusion'])
                print('  fusion_pre: %.2f' % time_mean['fusion_pre'])
                print('  fusion_layer: %.2f' % time_mean['fusion_layer'])
                print('  fusion_post: %.2f' % time_mean['fusion_post'])
                print('head: %.2f' % time_mean['head'])
                print('  head_backbone: %.2f' % time_mean['head_backbone'])
                print('  head_head: %.2f' % time_mean['head_head'])
                total = time_mean['pts'] + time_mean['img'] + time_mean['fusion'] + time_mean['head']
                print('total: %.2f' % total)
                print(' ')
                print('FPS: %.2f' % (1000/total))

            self.idx += 1
            return preds
