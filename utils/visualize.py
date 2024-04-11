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

def visualize_pts_context(pts_context, sweep_idx=0, channel_idx=None):
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
    fig, axs = plt.subplots(1, num_cams, figsize=(15, 15))

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
    plt.show()

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

import torch
import matplotlib.pyplot as plt

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
# plot_attention_maps(attn_maps_tensor, modality=0) # Plot the first modality
