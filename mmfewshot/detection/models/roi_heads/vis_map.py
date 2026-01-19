import pdb

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def compute_channelwise_cosine_similarity(feature_map, prototypes):
    """
    Compute channel-wise cosine similarity between feature map and RoI prototypes.

    Args:
        feature_map (torch.Tensor): shape (C, H, W)
        prototypes (list of torch.Tensor or torch.Tensor): each (C, 1, 1)

    Returns:
        similarity_maps (torch.Tensor): (K, H, W)
    """
    B, C, H, W = feature_map.shape

    feat_flat = feature_map.view(C, -1)  # (C, H*W)
    K = len(prototypes)

    if isinstance(prototypes, list):
        prototypes = torch.stack(prototypes, dim=0)  # (K, 1, C)
    prototypes = prototypes.view(K, C)  # (K, C)

    # Normalize
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    feat_norm = F.normalize(feat_flat, p=2, dim=0)  # normalize over channel

    similarity = torch.matmul(prototypes_norm, feat_norm)  # (K, H*W)
    similarity_maps = 1-similarity.view(-1, H, W)  # (K, H, W)

    return similarity_maps


def overlay_similarity_on_image(similarity_map, original_image, output_path, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Upsample similarity map to original image size, overlay on image, and save.

    Args:
        similarity_map (np.ndarray): 2D array (H, W), dtype float
        image_path (str): path to the original image
        output_path (str): path to save the overlaid image
        alpha (float): weight of heatmap over original image
        colormap: OpenCV colormap (e.g., cv2.COLORMAP_JET)
    """
    # Read original image
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Resize similarity map to original image size
    bs, c, h_orig, w_orig = original_image.shape[:4]
    image=np.transpose(original_image, (2, 3, 1, 0)).squeeze()
    sim_resized = cv2.resize(similarity_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    # Normalize to 0-255
    sim_normalized = ((sim_resized - sim_resized.min()) / (sim_resized.max() - sim_resized.min()) * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(sim_normalized, colormap)

    # Convert BGR to RGB (for consistency)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Blend
    overlaid = cv2.addWeighted(original_rgb.astype(np.float32), 1 - alpha, heatmap.astype(np.float32), alpha, 0)
    cv2.imwrite(output_path, np.clip(overlaid, 0, 255).astype(np.uint8))

# -----------------------------
# Example Usage
# -----------------------------
def generstae_featmap(feature_map, prototypes, img, img_name):
    # Step 1: Compute similarity maps
    similarity_maps = compute_channelwise_cosine_similarity(feature_map, prototypes)  # (K, H, W)
    similarity_maps_np = similarity_maps.detach().cpu().numpy()

    # Step 2: Define image path and output paths
    output_dir = "/home/f523/disk1/sxp/mmfewshot/vis_feat"
    img_dir=os.path.join(output_dir, img_name.split('/')[-1].replace('.jpg',''))
    os.makedirs(img_dir, exist_ok=True)
    K = len(prototypes)
    # Step 3: Overlay each prototype's similarity map
    for k in range(K):
        sim_map = similarity_maps_np[k]  # (H, W)
        output_path = os.path.join(img_dir, f"overlay_prototype_{k + 1}.png")
        overlay_similarity_on_image(
            sim_map,
            original_image=img,
            output_path=output_path,
            alpha=0.6,
            colormap=cv2.COLORMAP_HOT  # You can try: cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, etc.
        )