import cv2
import numpy as np
import open3d as o3d
import json

def normalize_points(points: np.array):
    # Compute the magnitude (Euclidean norm) of each point
    magnitudes = np.linalg.norm(points, axis=1, keepdims=True)

    # Avoid division by zero for points at the origin
    magnitudes[magnitudes == 0] = 1

    # Normalize each point to have a magnitude of 1
    normalized_points = points / magnitudes
    return normalized_points

def compute_disparity_map(rect_img1, rect_img2, min_disparity=0, num_disparities=5*16, block_size=6):
    """
    Compute the disparity map from a pair of rectified stereo images.

    Parameters:
    - rect_img1: Left rectified image (grayscale).
    - rect_img2: Right rectified image (grayscale).
    - min_disparity: Minimum possible disparity value.
    - num_disparities: Maximum disparity - min_disparity, should be divisible by 16.
    - block_size: Size of the block window for matching.

    Returns:
    - disp_map: Disparity map (same size as input images).
    """
    # Create the StereoSGBM matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=2,
        speckleWindowSize=100,
        speckleRange=32
    )


    # Compute the disparity map
    disp_map = stereo.compute(rect_img1, rect_img2).astype(np.float32) / 16.0
    return disp_map

def generate_point_cloud(disp_map, Q, color_image=None, max_z=None):
    # Create a mask where disparity is greater than 0
    valid_disp = disp_map > 0

    # Reproject to 3D using the valid disparity mask
    points_3D = cv2.reprojectImageTo3D(disp_map, Q)

    # Apply max_z filtering if needed
    if max_z is not None:
        depth_mask = points_3D[..., 2] <= max_z
        valid_mask = np.logical_and(valid_disp, depth_mask)
    else:
        valid_mask = valid_disp

    # Extract valid points and colors
    points = points_3D[valid_mask]

    if color_image is not None:
        colors = color_image[valid_mask]
        return points, colors
    else:
        return points

def parse_calibration_data(file_path):
    calibration = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ":" in line:
                key, value = line.strip().split(": ", 1)
                
                # Skip non-numeric entries like dates
                if key == "calib_time":
                    continue
                
                # Convert the line into a list of floats
                try:
                    values = [float(v) for v in value.split()]
                    
                    # Reshape based on the expected dimensions in the structure description
                    if key.startswith("S_") or key.startswith("S_rect_"):
                        calibration[key] = np.array(values).reshape((1, 2))        # 1x2 size
                    elif key.startswith("K_") or key.startswith("R_") or key.startswith("R_rect_"):
                        calibration[key] = np.array(values).reshape((3, 3))     # 3x3 matrix
                    elif key.startswith("T_"):
                        calibration[key] = np.array(values).reshape((3, 1))     # 3x1 vector
                    elif key.startswith("P_rect_"):
                        calibration[key] = np.array(values).reshape((3, 4))     # 3x4 matrix
                    elif key.startswith("D_"):
                        calibration[key] = np.array(values).reshape((1, 5))     # 1x5 distortion vector
                except ValueError:
                    print(f"Skipping entry due to non-numeric data: {key}")
                    
    return calibration

def calculate_q_matrix(P_left, P_right):
    # Extract focal length and principal points from P_left
    fx = P_left[0, 0]  # Focal length in x direction
    cx = P_left[0, 2]  # Principal point x-coordinate from the left camera
    cy = P_left[1, 2]  # Principal point y-coordinate from the left camera

    # Extract the principal point x-coordinate from the right camera
    cx_right = P_right[0, 2]  # Principal point x-coordinate from the right camera

    # Calculate baseline from P_right
    Tx = -P_right[0, 3] / fx  # Baseline distance (assuming P_right[0, 3] is non-zero)
    if Tx == 0:
        raise ValueError("Baseline (Tx) is zero. Check P_right[0, 3] and fx.")


    # Construct the Q matrix based on the revised stereo geometry
    Q = np.array([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0, -fx],
    [0, 0, -1 / Tx, (cx - cx_right) / Tx]
    ])

    return Q

def extract_bbox_from_txt(file_path, min_confidence=0.5):
    """
    Extracts bounding boxes from a .txt file with a confidence threshold.

    Args:
        file_path (str): Path to the .txt file containing detection data.
        min_confidence (float): Minimum confidence score for including a bounding box.

    Returns:
        dict: A dictionary with keys:
              - "bbox" : A list of 2D bounding boxes defined as [x_min, y_min, x_max, y_max].
              - "confidence": A list of confidence scores for each bounding box.
              - "class": A list of classes corresponding to each bounding box.
    """
    bounding_boxes = {"bbox": [], "confidence": [], "class": []}

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            obj_class = parts[0].strip()
            confidence = float(parts[1])
            x1, y1, x2, y2 = map(int, parts[2:])

            # Apply confidence threshold
            if confidence >= min_confidence:
                bounding_boxes["bbox"].append([x1, y1, x2, y2])
                bounding_boxes["confidence"].append(confidence)
                bounding_boxes["class"].append(obj_class)

    return bounding_boxes


def visualize_point_cloud(points, colors=None):
    """
    Visualize a 3D point cloud with optional RGB colors using Open3D.
    
    Parameters:
    - points: (N, 3) numpy array of 3D points.
    - colors: (N, 3) numpy array of RGB colors corresponding to each point, with values in [0, 255].
    """
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()

    # Assign points directly to the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Normalize colors to [0, 1] and assign if available
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Visualize the point cloud with all points, including any invalid depths
    o3d.visualization.draw_geometries([point_cloud], window_name="3D Point Cloud Visualization (All Points)")


def create_bbox_mask(image_shape, bounding_boxes):
    """
    Creates a boolean mask with the same shape as the image, where pixels inside any of the bounding boxes are True.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        bounding_boxes (list): List of bounding boxes, each defined as [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: Boolean mask where True indicates pixels inside the bounding boxes.
    """
    mask = np.zeros(image_shape[:2], dtype=bool)  # Assuming the image is grayscale or RGB

    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox

        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_shape[1] - 1, x_max)
        y_max = min(image_shape[0] - 1, y_max)

        mask[y_min:y_max+1, x_min:x_max+1] = True

    return mask

def create_seg_mask(file_path, image_shape, target_class_labels=None):
    """
    Creates a boolean mask from segmentation data in a .json file.

    Args:
        file_path (str): Path to the .json file containing segmentation data.
        image_shape (tuple): Shape of the image (height, width).
        target_class_labels (list or set): Class labels to include in the mask. If None, include all classes.

    Returns:
        np.ndarray: Boolean mask where True indicates pixels belonging to the target classes.
    """
    # Initialize mask with False (background)
    mask = np.zeros(image_shape[:2], dtype=bool)  # Assuming the image is grayscale or RGB

    # Load segmentation data from JSON file
    with open(file_path, 'r') as f:
        segmentation_data = json.load(f)
    print(len(segmentation_data))
    # Iterate over each segmentation item (object) in the data
    for segment in segmentation_data:
        class_label = segment['class_label']
        points = segment['points']  # List of [x, y] coordinates

        # Check if the class label is in the target classes
        if target_class_labels is None or class_label in target_class_labels:
            # Extract x and y coordinates
            points_array = np.array(points)
            x_coords = points_array[:, 0]
            y_coords = points_array[:, 1]

            # Ensure coordinates are within image bounds
            valid_mask = (x_coords >= 0) & (x_coords < image_shape[1]) & \
                         (y_coords >= 0) & (y_coords < image_shape[0])

            x_coords = x_coords[valid_mask].astype(int)
            y_coords = y_coords[valid_mask].astype(int)

            # Set the mask for these points
            mask[y_coords, x_coords] = True  # Note: y corresponds to row index, x to column index

    return mask

def create_seg_mask_with_mapping(file_path, image_shape, target_class_labels=None):
    """
    Creates a boolean mask from segmentation data in a .json file and maps detections to pixel indices.
    
    Returns:
        mask: Boolean mask.
        detection_map: Mapping of detection IDs to pixel coordinates.
        detection_labels: List of class labels corresponding to each detection.
    """
    mask = np.zeros(image_shape[:2], dtype=bool)  # Assuming the image is grayscale or RGB
    detection_map = {}  # Map from detection ID to pixel indices
    detection_labels = []  # List of detection labels

    with open(file_path, 'r') as f:
        segmentation_data = json.load(f)

    for idx, segment in enumerate(segmentation_data):
        class_label = segment['class_label']
        points = segment['points']  # List of [x, y] coordinates

        if target_class_labels is None or class_label in target_class_labels:
            points_array = np.array(points)
            x_coords = points_array[:, 0]
            y_coords = points_array[:, 1]

            valid_mask = (x_coords >= 0) & (x_coords < image_shape[1]) & \
                         (y_coords >= 0) & (y_coords < image_shape[0])

            x_coords = x_coords[valid_mask].astype(int)
            y_coords = y_coords[valid_mask].astype(int)

            mask[y_coords, x_coords] = True
            detection_map[idx] = (y_coords, x_coords)
            detection_labels.append(class_label)  # Store the class label for this detection

    return mask, detection_map, detection_labels

def RectImg2PC(rect_img1, rect_img2, P_left, P_right, max_z=20, color_img=None, mask=None):
    # Calculate Q-matrix
    Q = calculate_q_matrix(P_left, P_right)

    # Compute disparity map
    disp_map = compute_disparity_map(rect_img1, rect_img2, num_disparities=10 * 16, block_size=7)
    


    # Apply mask to disparity map if provided
    if mask is not None:
        # Ensure that the mask has the same dimensions as the disparity map
        if mask.shape != disp_map.shape:
            raise ValueError("Mask shape does not match disparity map shape.")
        # Set disparities outside the mask to NaN or zero
        disp_map = np.where(mask, disp_map, np.nan)

    # Generate point cloud data
    if color_img is not None:
        points, colors = generate_point_cloud(disp_map, Q, color_image=color_img, max_z=max_z)
    else:
        points = generate_point_cloud(disp_map, Q, color_image=None, max_z=max_z)
        colors = None  # No color information

    # Create Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    print(f"Points shape: {np.asarray(point_cloud.points).shape}")
    print(f"Colors shape: {colors.shape}")


    # Assign colors if available
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return points, colors

def RectImg2PC_for_clustering(rect_img1, rect_img2, P_left, P_right, max_z=20, mask=None, detection_map=None, detection_labels=None):
    """
    Generate a point cloud from stereo images and attach detection labels.
    
    Args:
        rect_img1: Left rectified grayscale image.
        rect_img2: Right rectified grayscale image.
        P_left: Left projection matrix.
        P_right: Right projection matrix.
        max_z: Maximum depth (Z-coordinate) for filtering points.
        mask: Binary mask for valid pixels (optional).
        detection_map: Map of detection IDs to pixel indices.
        detection_labels: List of detection labels corresponding to each detection ID.

    Returns:
        point_cloud: Open3D PointCloud object with points and colors.
        labels: NumPy array of detection labels for each point in the point cloud.
    """
    # Calculate Q-matrix
    Q = calculate_q_matrix(P_left, P_right)

    # Compute disparity map
    disp_map = compute_disparity_map(rect_img1, rect_img2, min_disparity=0, num_disparities=6 * 16, block_size=5)
    
    disp_visual = (disp_map - disp_map.min()) / (disp_map.max() - disp_map.min()) * 255
    cv2.imwrite("debug_disparity.png", disp_visual.astype(np.uint8))
    # Apply mask to disparity map if provided
    if mask is not None:
        if mask.shape != disp_map.shape:
            raise ValueError("Mask shape does not match disparity map shape.")
        disp_map = np.where(mask, disp_map, 0)
    # Create a valid mask for disparity and depth filtering
    valid_disp = disp_map > 0  # Disparity must be positive
    # Reproject to 3D
    points_3D = cv2.reprojectImageTo3D(disp_map, Q, handleMissingValues=True)

    # Apply max_z filtering if provided
    if max_z is not None:
        depth_mask = points_3D[..., 2] > 0  # Depth must be positive
        depth_mask &= points_3D[..., 2] <= max_z
        valid_disp &= depth_mask
        points = points_3D[valid_disp]

    # Normalize coordinates
    print(points)
    # points = normalize_points(points)
    # print(points.shape)

    # Initialize colors and labels array
    colors = np.zeros((points.shape[0], 3))  # Default: all black
    labels = np.full(points.shape[0], -1, dtype=int)  # Initialize with -1 (no label)

    if detection_map is not None and detection_labels is not None:
        unique_colors = np.random.rand(len(detection_map), 3)  # Generate random colors for detections

        # Assign colors and labels based on detection_map
        for detection_id, (y_coords, x_coords) in detection_map.items():
            detection_mask = np.zeros(disp_map.shape, dtype=bool)
            detection_mask[y_coords, x_coords] = True

            # Filter detection_mask to match valid 3D points
            valid_detection_mask = detection_mask[valid_disp]

            # Assign colors and labels for the current detection
            colors[valid_detection_mask] = unique_colors[detection_id]
            labels[valid_detection_mask] = detection_labels[detection_id]

    # Create Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Assign colors
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud, labels

