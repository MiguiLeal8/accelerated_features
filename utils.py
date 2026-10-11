import gzip
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from sklearn.decomposition import PCA


def read_raw_gz(file_path, verbose=0):
  # 1. Open the .gz file and read decompressed bytes into memory
  with gzip.open(file_path, "rb") as f:
    raw_data = f.read()

  # 2. Read header dimensions (first 3 4-byte integers = 12 bytes)
  header = np.frombuffer(raw_data, count=3, dtype=np.uint32, offset=0).copy()
  bands, horizontal, vertical = (int(header[0]), int(header[1]), int(header[2]))

  # 3. Read the rest of the data (12-byte offset for the header)
  data = np.frombuffer(raw_data, count=bands * horizontal * vertical, offset=12, dtype=np.int32).copy()

  if verbose == 1:
    print("* Read dataset:", file_path)
    print("  Bands (B):", bands, "Horizontal (H):", horizontal, "Vertical (V):", vertical)
    print("  Read elements:", len(data))

  # 4. Reshape to (Vertical, Horizontal, Bands)
  data = data.reshape(vertical, horizontal, bands)
  data = torch.FloatTensor(data)

  return data, horizontal, vertical, bands


def visualize_image(adapted_img, title_prefix="Image", verbose=0):
  """Visualizes an already model-adapted image (numpy array). 
  Automatically detects if it's 1-channel (2D) or multi-channel (3D) from its shape.
  """
  if isinstance(adapted_img, torch.Tensor):
    img_np = adapted_img.cpu().numpy()
  else:
    img_np = adapted_img

  h, w = img_np.shape[:2]

  # Deduce channels/bands from the shape dimensions
  if img_np.ndim == 2:
    channels = 1
    cmap_to_use = "gray"
    img_to_show = img_np
  elif img_np.ndim == 3:
    channels = img_np.shape[2]
    if channels == 1:
      cmap_to_use = "gray"
      img_to_show = img_np[:, :, 0]
    else:
      cmap_to_use = None
      img_to_show = img_np  # Assumed RGB format
  else:
    raise ValueError(f"Unexpected image shape: {img_np.shape}")

  if verbose == 1:
    print(f"Visualizing adapted image with shape {img_np.shape} ({channels} channel(s))...")

  # Display the image with Matplotlib
  plt.figure(figsize=(10, 8))
  plt.imshow(img_to_show, cmap=cmap_to_use)
  plt.title(f"{title_prefix} - Width: {w}, Height: {h} ({channels} channel(s))")
  plt.xlabel("Horizontal (H)")
  plt.ylabel("Vertical (V)")
  plt.axis("on")
  plt.show()


def convert_to_model_input(data_tensor, method="rgb_gray", verbose=0):
  """Converts a 5-band tensor into a format suitable for feature matching models.

  Methods:
    - 'rgb_gray': Extracts RGB composite and converts it to grayscale (2D array).
    - 'pca': Computes PCA across the 5 bands using scikit-learn to get 1 principal component (2D array).
    - 'rgb': Keeps the 3-channel RGB composite normalized to uint8 (3D array).
  """
  data_np = data_tensor.numpy()
  h, w, b = data_np.shape

  if method == "rgb_gray":
    if verbose == 1:
      print("[i] Conversion method: Using RGB composite converted to grayscale.")
    rgb = data_np[:, :, [2, 1, 0]]
    rgb_min, rgb_max = (
        rgb.min(axis=(0, 1), keepdims=True),
        rgb.max(axis=(0, 1), keepdims=True),
    )
    rgb_norm = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-8), 0, 1)
    gray = (0.2989 * rgb_norm[:, :, 0] + 0.5870 * rgb_norm[:, :, 1] + 0.1140 * rgb_norm[:, :, 2])
    img_out = (gray * 255).astype(np.uint8)  # Shape: (H, W)

  elif method == "pca":
    if verbose == 1:
      print("[i] Conversion method: Computing PCA across 5 bands using scikit-learn.")
    
    pixel_matrix = data_np.reshape(-1, b)
    pca = PCA(n_components=1)
    pca_component = pca.fit_transform(pixel_matrix)
    pca_img = pca_component.reshape(h, w)

    p_min, p_max = pca_img.min(), pca_img.max()
    pca_norm = np.clip((pca_img - p_min) / (p_max - p_min + 1e-8), 0, 1)
    img_out = (pca_norm * 255).astype(np.uint8)  # Shape: (H, W)

  elif method == "rgb":
    if verbose == 1:
      print("[i] Conversion method: Keeping 3-channel RGB.")
    rgb = data_np[:, :, [2, 1, 0]]
    rgb_min, rgb_max = (
        rgb.min(axis=(0, 1), keepdims=True),
        rgb.max(axis=(0, 1), keepdims=True),
    )
    rgb_norm = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-8), 0, 1)
    img_out = (rgb_norm * 255).astype(np.uint8)  # Shape: (H, W, 3)

  else:
    raise ValueError(f"Unknown conversion method: {method}")

  return img_out

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None, matchColor=(0, 255, 0), flags=2)

    return img_matches


def extract_transformation_params(H):
  """Decomposes a 3x3 Homography matrix into scale, angle, dx, and dy.

  Assuming a similarity transformation approximation for aerial imagery.
  """
  # Extract rotation/scale submatrix (top-left 2x2)
  a = H[0, 0]
  b = H[0, 1]
  c = H[1, 0]
  d = H[1, 1]

  # Translation
  dx = H[0, 2]
  dy = H[1, 2]

  # Scale (average scale from the two axes)
  scale_x = np.sqrt(a**2 + c**2)
  scale_y = np.sqrt(b**2 + d**2)
  scale = float((scale_x + scale_y) / 2.0)

  # Rotation angle in degrees
  angle_rad = np.arctan2(c, a)
  angle_deg = float(np.degrees(angle_rad))

  return {"scale": scale, "angle_deg": angle_deg, "dx": float(dx), "dy": float(dy)}


def compute_evaluation_metrics(xfeat_model, img0, img1):
  """Computes evaluation metrics between reference (img0) and target (img1):

  - Number of keypoints in reference
  - Number of keypoints in target
  - Number of matched points
  - Estimated transformation parameters (scale, angle, dx, dy)
  - SSIM registration score
  """
  out0 = xfeat_model.detectAndCompute(img0, top_k=4096)[0]
  out1 = xfeat_model.detectAndCompute(img1, top_k=4096)[0]

  kpts0_count = len(out0["keypoints"])
  kpts1_count = len(out1["keypoints"])

  mkpts0, mkpts1 = xfeat_model.match_xfeat(img0, img1)
  matches_count = len(mkpts0)

  transform_params = {"scale": None, "angle_deg": None, "dx": None, "dy": None}

  if matches_count >= 4:
    H, mask = cv2.findHomography(
        mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999
    )
    if H is not None:
      transform_params = extract_transformation_params(H)

  # Compute SSIM metric using the estimated transformation
  ssim_score = compute_ssim_metric(img0, img1, transform_params)

  metrics = {
      "kpts_reference": kpts0_count,
      "kpts_target": kpts1_count,
      "matches": matches_count,
      "ssim": ssim_score,
      **transform_params,
  }

  return metrics, mkpts0, mkpts1


def _gray(rgb):
  """Converts an RGB image (uint8) to grayscale float32."""
  return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def ssim_map(a, b, L=255.0):
  """Computes the local SSIM map between two grayscale float32 images."""
  C1 = (0.01 * L) ** 2
  C2 = (0.03 * L) ** 2
  win = (11, 11)
  sig = 1.5
  mu_a = cv2.GaussianBlur(a, win, sig)
  mu_b = cv2.GaussianBlur(b, win, sig)
  mu_a2 = mu_a * mu_a
  mu_b2 = mu_b * mu_b
  mu_ab = mu_a * mu_b
  sa = cv2.GaussianBlur(a * a, win, sig) - mu_a2
  sb = cv2.GaussianBlur(b * b, win, sig) - mu_b2
  sab = cv2.GaussianBlur(a * b, win, sig) - mu_ab
  return ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))


def masked_mean(m, mask):
  """Computes the mean of the SSIM map values where the mask is active."""
  return float(m[mask].mean()) if mask is not None and mask.any() else float("nan")


def compute_ssim_metric(ref_img, target_img, transform_params):
  """Warp target image using estimated transformation parameters (scale, angle, dx, dy)

  and computes SSIM against the reference image.
  """
  h, w = ref_img.shape[:2]
  scale = transform_params.get("scale", 1.0)
  angle = transform_params.get("angle_deg", 0.0)
  dx = transform_params.get("dx", 0.0)
  dy = transform_params.get("dy", 0.0)

  # Fallback if parameters are missing
  if scale is None or angle is None or dx is None or dy is None:
    return float("nan")

  # Build rigid transformation matrix centered on the image
  center = (w / 2.0, h / 2.0)
  M = cv2.getRotationMatrix2D(center, angle, scale)
  M[0, 2] += dx
  M[1, 2] += dy

  # Warp target image to align with reference space
  warped_target = cv2.warpAffine(target_img, M, (w, h), flags=cv2.INTER_LINEAR)

  # Convert both to grayscale
  ref_gray = _gray(ref_img)
  target_gray = _gray(warped_target)

  # Create a valid region mask (excluding zero padding or edge artifacts introduced by warping)
  valid = (warped_target.sum(axis=-1) > 0) if warped_target.ndim == 3 else (warped_target > 0)

  # Calculate masked SSIM mean
  score = masked_mean(ssim_map(ref_gray, target_gray), valid)
  return round(score, 4)


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
  example_file = "data/Dron/2019/eidos_v1_19.raw.gz"

  try:
    # 1. Read dataset
    data_tensor, width_h, height_v, bands = read_raw_gz(example_file, verbose=1)
    print(f"Returned tensor shape: {data_tensor.shape}")

    # 2. Generate the three conversion versions for model input / testing
    img_rgb_gray = convert_to_model_input(data_tensor, method="rgb_gray", verbose=1)
    img_pca = convert_to_model_input(data_tensor, method="pca", verbose=1)
    img_rgb = convert_to_model_input(data_tensor, method="rgb", verbose=1)

    # 3. Visualize the already adapted model inputs directly
    visualize_image(img_rgb_gray, title_prefix="Adapted: RGB Gray", verbose=1)
    visualize_image(img_pca, title_prefix="Adapted: PCA", verbose=1)
    visualize_image(img_rgb, title_prefix="Adapted: RGB 3-Channel", verbose=1)

  except Exception as e:
    print(f"An error occurred while reading or displaying the file: {e}")