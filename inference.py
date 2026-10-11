import gc
import cv2
import torch
import matplotlib.pyplot as plt
from modules.xfeat import XFeat
from utils import (
    read_raw_gz, 
    visualize_image, 
    convert_to_model_input, 
    warp_corners_and_draw_matches,
    compute_evaluation_metrics
)

DATA_PATH = "data/Dron/"

reference = "2024/eidos_v2_24.raw.gz"
target = "2019/eidos_v2_19.raw.gz"

# 1. Load images safely using the custom reader
print("--- 1. LOADING IMAGES ---")
try:
    img_tensor0, width_h0, height_v0, _ = read_raw_gz(DATA_PATH + reference, verbose=1)
except Exception as e:
    print(f"Could not load reference image: {e}")
    img_tensor0 = None

try:
    img_tensor1, width_h1, height_v1, _ = read_raw_gz(DATA_PATH + target, verbose=1)
except Exception as e:
    print(f"Could not load target image: {e}")
    img_tensor1 = None

# 2. Convert tensors to model input format (Method: 'rgb') and IMMEDIATELY free heavy RAM
print("--- 2. CONVERTING TO MODEL INPUT AND CLEARING HEAVY RAM ---")
img_np0 = convert_to_model_input(img_tensor0, method="rgb", verbose=1) if img_tensor0 is not None else None
img_np1 = convert_to_model_input(img_tensor1, method="rgb", verbose=1) if img_tensor1 is not None else None

# Immediately free the giant 5-band tensors from RAM
del img_tensor0, img_tensor1
gc.collect()

# Visualize the already adapted images
if img_np0 is not None:
    visualize_image(img_np0, title_prefix="Reference Image (RGB 3-Channel)", verbose=1)
if img_np1 is not None:
    visualize_image(img_np1, title_prefix="Target Image (RGB 3-Channel)", verbose=1)


# 3. Initialize model
print("--- 3. INITIALIZING XFEAT AND RUNNING INFERENCE ---")
xfeat = XFeat()

# CASE 1: Both images loaded -> Matching & Metrics
if img_np0 is not None and img_np1 is not None:
    metrics, mkpts0, mkpts1 = compute_evaluation_metrics(xfeat, img_np0, img_np1)
    
    print("\n" + "="*40)
    print("      EVALUATION METRICS SUMMARY      ")
    print("="*40)
    print(f" - Keypoints in Reference: {metrics['kpts_reference']}")
    print(f" - Keypoints in Target:    {metrics['kpts_target']}")
    print(f" - Matched Points (Inliers): {metrics['matches']}")
    print(f" - SSIM Score:             {metrics['ssim']}")
    print(f" - Estimated Scale:        {metrics['scale']:.4f}" if metrics['scale'] is not None else " - Estimated Scale: N/A")
    print(f" - Estimated Angle (°):    {metrics['angle_deg']:.2f}" if metrics['angle_deg'] is not None else " - Estimated Angle (°): N/A")
    print(f" - Translation (dx, dy):   ({metrics['dx']:.2f}, {metrics['dy']:.2f})" if metrics['dx'] is not None else " - Translation: N/A")
    print("="*40 + "\n")

    # Display final matches with Matplotlib
    canvas = warp_corners_and_draw_matches(mkpts0, mkpts1, img_np0, img_np1)
    plt.figure(figsize=(12, 12))
    plt.imshow(canvas[..., ::-1])
    plt.title("XFeat Matches between 2019 and 2024")
    plt.show()

# CASE 2: Only one image available -> Individual keypoints
else:
    img_np = img_np0 if img_np0 is not None else img_np1
    if img_np is None:
        raise FileNotFoundError(f"Could not load any image from {DATA_PATH}")

    outputs = xfeat.detectAndCompute(img_np, top_k=4096)[0]
    keypoints = outputs['keypoints']

    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()

    print(f"Extracted keypoints: {keypoints.shape[0]}")

    cv_kpts = [cv2.KeyPoint(pt[0], pt[1], 3) for pt in keypoints]
    img_kpts = cv2.drawKeypoints(
        img_np, cv_kpts, None, 
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display single image keypoints with Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_kpts, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("XFeat Single Image Keypoints")
    plt.show()