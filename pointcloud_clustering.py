from ultralytics import YOLO
import os
import numpy as np
import cv2
import json
import open3d as o3d
from image_processing import parse_calibration_data, create_seg_mask_with_mapping, RectImg2PC_for_clustering, RectImg2PC, generate_point_cloud
from sklearn.cluster import DBSCAN
from collections import Counter

def get_polygon_points(image, mask_file, output_json):
    """
    Given an image and a mask file with polygon coordinates,
    extracts all pixel coordinates inside the polygon(s) and saves them into a JSON file,
    including class labels and polygon coordinates for each polygon.

    Parameters:
    - image: NumPy array representing the image (used for resolution).
    - mask_file: Path to the mask text file.
    - output_json: Path to the output JSON file.

    Returns:
    - objects_with_points: A list of dictionaries containing class labels, polygon coordinates, and points inside the polygon.
    """
    img_height, img_width = image.shape[:2]
    
    # Step 1: Parse the mask file
    objects = []
    with open(mask_file, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) < 3:
                continue
            class_label = int(tokens[0])
            coords = list(map(float, tokens[1:]))
            if len(coords) % 2 != 0:
                continue
            polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            objects.append({
                'class_label': class_label,
                'polygon': polygon
            })
    
    # Step 2: Convert normalized coordinates to pixel coordinates
    for obj in objects:
        polygon = obj['polygon']
        pixel_polygon = [(int(round(x * img_width)), int(round(y * img_height))) for x, y in polygon]
        obj['pixel_polygon'] = pixel_polygon
    
    # Initialize the list to store objects with their points
    objects_with_points = []

    # For each object, create a mask and extract points
    for obj in objects:
        class_label = obj['class_label']
        pixel_polygon = np.array([obj['pixel_polygon']], dtype=np.int32)
        # Create a mask for the current polygon
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, pixel_polygon, color=1)
        # Extract points inside the polygon
        y_coords, x_coords = np.where(mask == 1)
        points_inside = list(zip(map(int, x_coords), map(int, y_coords)))
        # Store the data
        objects_with_points.append({
            'class_label': class_label,
            # 'polygon': obj['pixel_polygon'],
            'points': points_inside
        })  
    
    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(objects_with_points, f)
    
    return objects_with_points

def get_right_image_path(left_image_path):
    """
    Generate the right image path corresponding to a given left image path.

    Args:
        left_image_path (str): Path to the left image.

    Returns:
        str: Path to the corresponding right image.

    Raises:
        ValueError: If 'image_02' is not found in the left image path.
    """
    # Normalize and split the path into components
    left_image_path = os.path.normpath(left_image_path)
    leading_slash = left_image_path.startswith("/")  # Check if path starts with a leading slash
    path_parts = left_image_path.split(os.sep)

    # Check and replace 'image_02' with 'image_03'
    if 'image_02' in path_parts:
        path_parts[path_parts.index('image_02')] = 'image_03'
    else:
        raise ValueError(f"Left image path does not contain 'image_02': {left_image_path}")

    # Reconstruct the path
    right_image_path = os.path.join(*path_parts)
    if leading_slash:  # Re-add leading slash if it was present
        right_image_path = "/" + right_image_path

    # Normalize again for consistent formatting
    right_image_path = os.path.normpath(right_image_path)

    # Debug log
    print(f"Left image path: {left_image_path}")
    print(f"Right image path: {right_image_path}")

    return right_image_path



def generate_segmented_point_cloud(
    left_image_path: str,
    right_image_path: str,
    calibration_file_path: str,
    seg_json_path: str,
    max_z: float = 20.0,
    target_class_labels: list = None,
    save_point_cloud: bool = False,
    save_path: str = 'segmented_point_cloud.npy',
    visualize: bool = False
) -> tuple:
    """
    Generates a segmented point cloud from stereo images and segmentation data.
    
    Returns:
    - point_cloud (np.ndarray): Array of shape (N, 3) or (N, 6) if colors are included.
    - mask (np.ndarray): Binary mask used for segmentation.
    """
    # Step 1: Load and convert images to grayscale
    color_img = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    rect_img1 = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    rect_img2 = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    if rect_img1 is None:
        raise FileNotFoundError(f"Left grayscale image not found at path: {left_image_path}")
    if rect_img2 is None:
        raise FileNotFoundError(f"Right grayscale image not found at path: {right_image_path}")
    
    # Step 3: Load and parse calibration data
    calibration = parse_calibration_data(calibration_file_path)
    if "P_rect_02" not in calibration or "P_rect_03" not in calibration:
        raise KeyError("Calibration data must contain 'P_rect_02' and 'P_rect_03'.")
    P_left = calibration["P_rect_02"]
    P_right = calibration["P_rect_03"]
    
    seg_mask, detection_map, detection_labels = create_seg_mask_with_mapping(
        file_path=seg_json_path,
        image_shape=rect_img1.shape,
        target_class_labels=target_class_labels
    )
    # cv2.imwrite("debug_mask.png", seg_mask.astype(np.uint8) * 255)
    for detection_id, (y_coords, x_coords) in detection_map.items():
        print(f"Detection {detection_id}: X max={x_coords.max()}, Y max={y_coords.max()}")


    point_cloud, labels = RectImg2PC_for_clustering(
        rect_img1=rect_img1,
        rect_img2=rect_img2,
        P_left=P_left,
        P_right=P_right,
        max_z=max_z,
        mask=seg_mask,
        detection_map=detection_map,
        detection_labels=detection_labels
    )
    points, colors = RectImg2PC(
        rect_img1=rect_img1,
        rect_img2=rect_img2,
        P_left=P_left,
        P_right=P_right,
        max_z=max_z,
        
        color_img=color_img,
    )

    # Step 6: Save the point cloud if required
    if save_point_cloud:
        o3d.io.write_point_cloud(save_path, point_cloud)
        print(f"Point cloud saved to {save_path}")
    
    # Step 7: Visualize the point cloud if required
    if visualize:
        # if point_cloud.points.shape[1] < 6:
        #     raise ValueError("Point cloud does not contain color information (requires at least 6 columns).")
        
        # Separate XYZ and RGB
        # xyz = point_cloud.points[:, :3]
        # rgb = point_cloud[:, 3:] / 255.0  # Normalize RGB to [0, 1]
        
        # # Create Open3D point cloud object
        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(xyz)
        # o3d_pcd.colors = o3d.utility.Vector3dVector(rgb)
        
        # Visualize
        o3d.visualization.draw_geometries([point_cloud], window_name="Segmented Point Cloud")
    
    return point_cloud, seg_mask,labels, points, colors

def dbscan_with_labels_and_outlier_removal(point_cloud, labels, eps=0.5, min_samples=10):
    """
    Perform DBSCAN clustering on a point cloud, assign detection labels to centroids,
    and remove outliers from the resulting point cloud.
    """
    voxel_size = 0.1
    point_cloud_down, _, mapping_indices = point_cloud.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=point_cloud.get_min_bound(),
        max_bound=point_cloud.get_max_bound(),
    )
    # o3d.visualization.draw_geometries([point_cloud_down])
    # print("Downsampled point cloud size:", np.asarray(point_cloud_down.points).shape)
    
    # Extract points and colors
    points = np.asarray(point_cloud_down.points)
    colors = np.asarray(point_cloud_down.colors)
    features = np.hstack((points, colors))
    # print(points.shape)
    # Aggregate labels for downsampled points
    downsampled_labels = []
    for indices in mapping_indices:
        labels_in_voxel = labels[indices]
        most_common_label = Counter(labels_in_voxel).most_common(1)[0][0]
        downsampled_labels.append(most_common_label)
    labels = np.array(downsampled_labels)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = dbscan.fit_predict(features)

    inlier_mask = cluster_ids != -1
    inlier_points = points[inlier_mask]
    inlier_colors = colors[inlier_mask]
    inlier_labels = labels[inlier_mask]
    inlier_cluster_ids = cluster_ids[inlier_mask]

    clustered_point_cloud = o3d.geometry.PointCloud()
    clustered_point_cloud.points = o3d.utility.Vector3dVector(inlier_points)
    clustered_point_cloud.colors = o3d.utility.Vector3dVector(inlier_colors)

    centroids = []
    cluster_labels = []

    for cluster_id in np.unique(inlier_cluster_ids):
        cluster_indices = np.where(inlier_cluster_ids == cluster_id)[0]
        cluster_points = inlier_points[cluster_indices]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

        cluster_detection_labels = inlier_labels[cluster_indices]
        most_common_label = Counter(cluster_detection_labels).most_common(1)[0][0]
        cluster_labels.append(most_common_label)
    # print(cluster_labels)
    return np.array(centroids), np.array(cluster_labels), clustered_point_cloud, len(np.unique(inlier_cluster_ids))

def visualize_point_cloud_with_centroids(point_cloud, detections):
    """
    Visualize a point cloud with detected object centroids using spheres.

    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud
        detections (List[Detection]): List of Detection objects containing centroids and class labels
    """
    # Create a list of geometry objects for visualization
    geometries = [point_cloud]

    # Add centroids as spheres
    for detection in detections:
        # Create a sphere for the centroid
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.translate(detection.centroid)  # Move sphere to centroid position
        sphere.paint_uniform_color([200, 0, 200])  # Black color for centroids
        geometries.append(sphere)

    # Create visualizer to get view parameters
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud with Centroids")
    
    # Add geometries
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Run visualization
    vis.run()
    
    # Get and print view parameters before closing
    view_control = vis.get_view_control()
    cam = view_control.convert_to_pinhole_camera_parameters()
    
    print("\nView Parameters:")
    print(f"Extrinsic matrix:\n{cam.extrinsic}")
    print(f"Front vector: {-cam.extrinsic[2, :3]}")
    print(f"Lookat point: {-cam.extrinsic[:3, 3]}")
    print(f"Up vector: {-cam.extrinsic[1, :3]}")
    print(f"Zoom: {view_control.get_field_of_view()}")
    
    vis.destroy_window()

def visualize_original_pc_with_centroids(
    points, colors, cluster_labels, centroids, centroid_labels, unique_label_colors=None
):
    """
    Visualize a point cloud with the original colors and add centroids using spheres.

    Args:
        points (np.ndarray): (N, 3) Array of 3D points.
        colors (np.ndarray): (N, 3) Array of RGB colors corresponding to the points.
        cluster_labels (np.ndarray): Cluster labels for each point.
        centroids (np.ndarray): (M, 3) Array of centroid positions for each cluster.
        centroid_labels (np.ndarray): Labels assigned to each centroid.
        unique_label_colors (dict or None): Predefined colors for clusters and centroids.
    """
    # Ensure colors are normalized to [0, 1] for Open3D
    if colors is not None:
        colors = colors / 255.0

    # Create an Open3D point cloud object and assign points and colors
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Generate unique colors for clusters if not provided
    if unique_label_colors is None:
        unique_labels = np.unique(cluster_labels)
        unique_label_colors = {
            label: np.random.rand(3) for label in unique_labels if label >= 0
        }

    # Create a list of geometries for visualization
    geometries = [point_cloud]

    # Add centroids as spheres with cluster colors
    for i, (centroid, label) in enumerate(zip(centroids, centroid_labels)):
        # Create a sphere for each centroid
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.translate(centroid)  # Move the sphere to the centroid position

        # Use the cluster color or a default color if unavailable
        sphere_color = [200,0,200]
        sphere.paint_uniform_color(sphere_color)
        geometries.append(sphere)

    # Visualize the point cloud and centroids
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud with Centroids",
    )

def visualize_label_bounding_box(point_cloud, cluster_labels, label_of_interest):
    # Extract points with the specified label
    points = np.asarray(point_cloud.points)
    points_label = points[cluster_labels == label_of_interest]
    
    if points_label.size == 0:
        print(f"No points found with label {label_of_interest}.")
        return

    # Create a PointCloud object for the label
    pcd_label = o3d.geometry.PointCloud()
    pcd_label.points = o3d.utility.Vector3dVector(points_label)
    
    # Compute the axis-aligned bounding box
    aabb = pcd_label.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)  # Red color for the bounding box

    # Visualize the point cloud and bounding box
    o3d.visualization.draw_geometries([point_cloud, aabb])


def compute_min_max_coordinates_for_labels(point_cloud, cluster_labels, labels_of_interest):
    def compute_min_max_coordinates(point_cloud, cluster_labels, label_of_interest):
        """
        Compute the minimum and maximum x, y, z coordinates for points with a specific cluster label,
        and calculate the differences (delta_x, delta_y, delta_z) between them.

        Args:
            point_cloud (o3d.geometry.PointCloud): The point cloud.
            cluster_labels (np.ndarray): Cluster labels for each point.
            label_of_interest (int): The label for which to compute min and max coordinates.

        Returns:
            min_coords (np.ndarray): Minimum x, y, z coordinates.
            max_coords (np.ndarray): Maximum x, y, z coordinates.
            deltas (np.ndarray): Differences between max and min coordinates (delta_x, delta_y, delta_z).
        """
        # Ensure cluster_labels is a NumPy array
        cluster_labels = np.array(cluster_labels)
        
        # Extract points from the point cloud
        points = np.asarray(point_cloud.points)
        
        # Filter points with the specified label
        mask = cluster_labels == label_of_interest
        points_label = points[mask]
        
        # Check if any points have the label
        if points_label.size == 0:
            print(f"No points found with label {label_of_interest}.")
            return None, None, None

        # Compute minimum and maximum coordinates along each axis
        min_coords = np.min(points_label, axis=0)
        max_coords = np.max(points_label, axis=0)
        
        # Compute deltas (extents) along each axis
        deltas = max_coords - min_coords

        # Display the results
        print(f"Label {label_of_interest}:")
        print(f"  Minimum coordinates: x={min_coords[0]:.2f}, y={min_coords[1]:.2f}, z={min_coords[2]:.2f}")
        print(f"  Maximum coordinates: x={max_coords[0]:.2f}, y={max_coords[1]:.2f}, z={max_coords[2]:.2f}")
        print(f"  Deltas: delta_x={deltas[0]:.2f}, delta_y={deltas[1]:.2f}, delta_z={deltas[2]:.2f}")
        
        return min_coords, max_coords, deltas

    results = {}
    for label in labels_of_interest:
        min_coords, max_coords, deltas = compute_min_max_coordinates(point_cloud, cluster_labels, label)
        if min_coords is not None:
            results[label] = {
                'min_coords': min_coords,
                'max_coords': max_coords,
                'delta_x': deltas[0],
                'delta_y': deltas[1],
                'delta_z': deltas[2]
            }
    return results

def remove_background_with_roi(point_cloud, centroid_estimate, roi_size=(2.0, 2.0, 2.0)):
    """
    Remove background points using a 3D ROI around the estimated centroid.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud
        centroid_estimate (np.ndarray): Estimated centroid from segmentation
        roi_size (tuple): Size of ROI box (dx, dy, dz)
    """
    points = np.asarray(point_cloud.points)
    
    # Create bounds around centroid
    min_bound = centroid_estimate - np.array(roi_size) / 2
    max_bound = centroid_estimate + np.array(roi_size) / 2
    
    # Filter points within bounds
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if point_cloud.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[mask])
    
    return filtered_pcd

def remove_background_with_distance(point_cloud, centroid_estimate, max_distance=2.0):
    """
    Remove background points based on distance from estimated centroid.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud
        centroid_estimate (np.ndarray): Estimated centroid from segmentation
        max_distance (float): Maximum distance from centroid to keep points
    """
    points = np.asarray(point_cloud.points)
    
    # Calculate distances from each point to centroid
    distances = np.linalg.norm(points - centroid_estimate, axis=1)
    
    # Keep points within threshold
    mask = distances <= max_distance
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if point_cloud.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[mask])
    
    return filtered_pcd

def remove_background_density_based(point_cloud, centroid_estimate, radius=1.0, min_points=10):
    """
    Remove background using density-based filtering around centroid.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud
        centroid_estimate (np.ndarray): Estimated centroid from segmentation
        radius (float): Radius for density calculation
        min_points (int): Minimum number of points in radius to keep
    """
    # First, do rough distance-based filtering
    rough_filtered = remove_background_with_distance(point_cloud, centroid_estimate, max_distance=radius*2)
    
    # Build KD-tree for efficient neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(rough_filtered)
    
    points = np.asarray(rough_filtered.points)
    keep_indices = []
    
    # For each point, check density in its neighborhood
    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(points[i], radius)
        if k >= min_points:
            keep_indices.append(i)
    
    # Create filtered point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[keep_indices])
    if rough_filtered.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(rough_filtered.colors)[keep_indices])
    
    return filtered_pcd

def get_refined_centroid(point_cloud, initial_centroid, class_id):
    """
    Get refined centroid using multiple background removal methods.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud
        initial_centroid (np.ndarray): Initial centroid estimate from segmentation
        class_id (int): Class ID for adjusting parameters
    """
    # Adjust parameters based on class
    if class_id == 0:  # car
        roi_size = (4.0, 2.0, 2.0)
        max_distance = 3.0
    elif class_id == 2:  # person
        roi_size = (1.0, 1.0, 2.0)
        max_distance = 1.0
    else:  # bike or other
        roi_size = (2.0, 1.0, 2.0)
        max_distance = 1.5

    # Apply ROI filtering first
    roi_filtered = remove_background_with_roi(point_cloud, initial_centroid, roi_size)
    
    # Then apply density-based filtering
    density_filtered = remove_background_density_based(roi_filtered, initial_centroid)
    
    # Calculate final centroid
    if len(np.asarray(density_filtered.points)) > 0:
        final_centroid = np.mean(np.asarray(density_filtered.points), axis=0)
    else:
        final_centroid = initial_centroid
    
    return final_centroid, density_filtered

def get_centroid_from_point_cloud(point_cloud, outlier_std_ratio=2.0):
    """
    Get centroid from point cloud after statistical outlier removal.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud
        outlier_std_ratio (float): Standard deviation ratio for outlier removal
        
    Returns:
        np.ndarray: Centroid coordinates [x, y, z]
    """
    # First, perform statistical outlier removal
    cleaned_pcd, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20,  # Number of neighbors to analyze
        std_ratio=outlier_std_ratio  # Standard deviation threshold
    )
    
    # Convert to numpy array for k-means
    points = np.asarray(cleaned_pcd.points)
    
    if len(points) == 0:
        return None
        
    # Perform k-means clustering with k=1
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(points)
    
    # The centroid is the cluster center
    centroid = kmeans.cluster_centers_[0]
    
    return centroid

def get_centroid_from_point_cloud(point_cloud, outlier_std_ratio=2.0):
    """
    Get centroid from point cloud using DBSCAN with grid search.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud
        outlier_std_ratio (float): Standard deviation ratio for outlier removal
        
    Returns:
        np.ndarray: Centroid coordinates [x, y, z]
    """
    # First, perform statistical outlier removal
    cleaned_pcd, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20,  # Number of neighbors to analyze
        std_ratio=outlier_std_ratio  # Standard deviation threshold
    )
    
    # Convert to numpy array for DBSCAN
    points = np.asarray(cleaned_pcd.points)
    
    if len(points) == 0:
        return None
        
    # Grid search parameters
    eps_values = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    min_samples_values = [10, 20, 30, 50, 75, 100]
    
    best_centroid = None
    best_score = 0  # Number of inlier points
    
    # Try different parameter combinations
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                # Perform DBSCAN clustering
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(points)
                
                # Count number of clusters (excluding noise points labeled as -1)
                n_clusters = len(set(cluster_labels[cluster_labels >= 0]))
                
                # If we get exactly one cluster
                if n_clusters == 1:
                    # Get points from the cluster (excluding noise points)
                    cluster_points = points[cluster_labels == 0]
                    n_points = len(cluster_points)
                    
                    # If this cluster has more points than previous best
                    if n_points > best_score:
                        centroid = np.mean(cluster_points, axis=0)
                        best_centroid = centroid
                        best_score = n_points
                        
            except Exception as e:
                continue
    
    return best_centroid

def process_segmented_point_cloud(point_cloud, class_id, bbox):
    """
    Process a segmented point cloud to get centroid coordinates.
    Includes visualization options for debugging.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud
        class_id (int): Class ID of the detection
        
    Returns:
        Detection: Detection object with centroid and class label, or None if processing fails
    """
    # Check if point cloud is empty
    if len(np.asarray(point_cloud.points)) == 0:
        print(f"Warning: Empty point cloud received for class {class_id}")
        return None
    
    # Optional: Voxel downsampling to reduce computation time
    voxel_size = 0.05  # Adjust based on your point cloud scale
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size)
    
    # Get initial centroid
    initial_centroid = get_centroid_from_point_cloud(downsampled_pcd)
    
    # If we couldn't get an initial centroid, try using mean of all points
    if initial_centroid is None:
        points = np.asarray(point_cloud.points)
        if len(points) > 0:
            initial_centroid = np.mean(points, axis=0)
        else:
            print(f"Warning: Could not compute centroid for class {class_id}")
            return None

    try:
        # Try to refine the centroid
        refined_centroid, filtered_pcd = get_refined_centroid(point_cloud, initial_centroid, class_id)
        
        # Verify the refined centroid
        if refined_centroid is not None and not np.any(np.isnan(refined_centroid)):
            return Detection(refined_centroid, class_id,bbox)
        else:
            # If refinement failed, fall back to initial centroid
            print(f"Warning: Centroid refinement failed for class {class_id}, using initial centroid")
            return Detection(initial_centroid, class_id,bbox)
            
    except Exception as e:
        print(f"Error processing point cloud for class {class_id}: {str(e)}")
        # Fall back to initial centroid if refinement fails
        if initial_centroid is not None:
            print("Falling back to initial centroid")
            return Detection(initial_centroid, class_id,bbox)
        return None

def match_stereo_detections(left_det, right_detections, img_width, img_height, max_disparity=128):
    """
    Match a detection from left image to best corresponding detection in right image
    with relaxed matching criteria.
    """
    # Parse left detection
    left_class = int(left_det.split()[0])
    left_points = np.array([float(x) for x in left_det.split()[1:]]).reshape(-1, 2)
    left_points[:, 0] *= img_width
    left_points[:, 1] *= img_height
    left_points = left_points.astype(np.int32)
    
    # Create mask for left detection
    left_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(left_mask, [left_points], 1)
    
    # Calculate vertical bounds of left detection
    left_y_min = np.min(left_points[:, 1])
    left_y_max = np.max(left_points[:, 1])
    left_height = left_y_max - left_y_min
    
    best_match = None
    best_disparity = None
    best_iou = 0
    
    # Try each right detection of same class
    for right_det in right_detections:
        right_class = int(right_det.split()[0])
        
        # Only consider detections of same class
        if right_class != left_class:
            continue
            
        # Parse right detection
        right_points = np.array([float(x) for x in right_det.split()[1:]]).reshape(-1, 2)
        right_points[:, 0] *= img_width
        right_points[:, 1] *= img_height
        right_points = right_points.astype(np.int32)
        
        # Check vertical position similarity with relaxed constraints
        right_y_min = np.min(right_points[:, 1])
        right_y_max = np.max(right_points[:, 1])
        right_height = right_y_max - right_y_min
        
        # Calculate height difference ratio
        height_ratio = min(left_height, right_height) / max(left_height, right_height)
        
        # Relaxed vertical position check (40% of height instead of 20%)
        # Relaxed height ratio check (0.5 instead of 0.7)
        if (abs(left_y_min - right_y_min) > 0.7 * left_height or 
            abs(left_y_max - right_y_max) > 0.7 * left_height or
            height_ratio < 0.7):
            continue
        
        # Create mask for right detection
        right_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(right_mask, [right_points], 1)
        
        # Try different disparity values
        for d in range(max_disparity):
            # Shift right mask left
            shift_matrix = np.float32([[1, 0, -d], [0, 1, 0]])
            shifted_right_mask = cv2.warpAffine(right_mask, shift_matrix, (img_width, img_height))
            
            # Calculate IoU
            intersection = cv2.bitwise_and(left_mask, shifted_right_mask)
            union = cv2.bitwise_or(left_mask, shifted_right_mask)
            iou = np.sum(intersection) / (np.sum(union) + 1e-6)
            
            if iou > best_iou:
                best_iou = iou
                best_match = right_det
                best_disparity = d
    
    return best_match, best_disparity, best_iou

def process_individual_detections_with_intersection(model, classes, img_left_path, img_right_path, calibration_file_path, conf=0.7, max_z=100.0, visualize=True):
    """
    Process each detection individually using intersection of left and right camera masks,
    with improved stereo matching.
    
    Returns:
        List[Detection]: List of Detection objects with centroids and class labels
    """
    # Get file paths and create temp directory
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    temp_directory_path = os.path.join(parent_directory, "temp")
    os.makedirs(temp_directory_path, exist_ok=True)
    
    # Clear existing temp files
    mask_left_path = os.path.join(temp_directory_path, "temp_left.txt")
    mask_right_path = os.path.join(temp_directory_path, "temp_right.txt")
    if os.path.exists(mask_left_path):
        open(mask_left_path, 'w').close()
    if os.path.exists(mask_right_path):
        open(mask_right_path, 'w').close()
    
    # Get detections from both images
    results_left = model.predict(source=img_left_path, classes=classes, conf=conf)
    results_right = model.predict(source=img_right_path, classes=classes, conf=conf)
    
    # Save detection results
    results_left[0].save_txt(mask_left_path)
    results_right[0].save_txt(mask_right_path)
    
    # Read detections
    with open(mask_left_path, 'r') as f:
        left_detections = f.readlines()
    with open(mask_right_path, 'r') as f:
        right_detections = f.readlines()
    
    detections = []
    image = cv2.imread(img_left_path)
    height, width = image.shape[:2]
    
    # Process each detection from left image
    for i, left_det in enumerate(left_detections):
        # Create temporary files for single detection
        single_mask_path = os.path.join(temp_directory_path, f"temp_{i}.txt")
        output_json = os.path.join(temp_directory_path, f"points_{i}.json")
        
        # Flush existing files
        if os.path.exists(single_mask_path):
            open(single_mask_path, 'w').close()
        if os.path.exists(output_json):
            open(output_json, 'w').close()
            
        # Find matching detection in right image
        right_match, disparity, iou = match_stereo_detections(
            left_det, right_detections, width, height, max_disparity=128
        )
        
        # Only process if we found a good match
        if right_match is not None and iou > 0.1:  # Adjust threshold as needed
            # Get class label from detection
            class_id = int(left_det.split()[0])
            
            # Parse left detection points
            left_points = np.array([float(x) for x in left_det.split()[1:]]).reshape(-1, 2)
            left_points[:, 0] *= width
            left_points[:, 1] *= height
            left_points = left_points.astype(np.int32)
            
            # Create mask for left detection
            left_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(left_mask, [left_points], 1)
            
            # Parse right detection points and create mask
            right_points = np.array([float(x) for x in right_match.split()[1:]]).reshape(-1, 2)
            right_points[:, 0] *= width
            right_points[:, 1] *= height
            right_points = right_points.astype(np.int32)
            
            # Create mask for right detection
            right_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(right_mask, [right_points], 1)
            
            # Shift right mask by found disparity
            shift_matrix = np.float32([[1, 0, -disparity], [0, 1, 0]])
            shifted_right_mask = cv2.warpAffine(right_mask, shift_matrix, (width, height))
            
            # Calculate intersection
            intersection = cv2.bitwise_and(left_mask, shifted_right_mask)
            
            # Convert intersection mask to polygon points
            contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Save intersection polygon to file
                with open(single_mask_path, 'w') as f:
                    points_str = " ".join([f"{pt[0][0]/width} {pt[0][1]/height}" for pt in approx])
                    f.write(f"{class_id} {points_str}\n")
                
                points_with_labels = get_polygon_points(image, single_mask_path, output_json)
                
                # Generate point cloud for intersection
                point_cloud, seg_mask, labels, points_original, colors_original = generate_segmented_point_cloud(
                    left_image_path=img_left_path,
                    right_image_path=img_right_path,
                    calibration_file_path=calibration_file_path,
                    seg_json_path=output_json,
                    max_z=max_z,
                    target_class_labels=[class_id],
                    save_point_cloud=False,
                    visualize=False
                )
                
                detection = process_segmented_point_cloud(point_cloud, class_id)
                
                if detection is not None:
                    detections.append(detection)
                    if visualize:
                        print(f"Detection {i}: Class {class_id}, Disparity: {disparity} pixels, IoU: {iou:.3f}")
                        visualize_point_cloud_with_centroids(point_cloud, detections)
    
    return detections

from tracking import Detection
def process_individual_detections(model, classes, img_left_path, img_right_path, calibration_file_path, conf=0.7, max_z=100.0, visualize = True):
    """
    Process each detection individually to get more accurate centroids and track objects across frames.
    
    Returns:
        List[Detection]: List of Detection objects with centroids and class labels
    """
    # Get file paths and create temp directory
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    temp_directory_path = os.path.join(parent_directory, "temp")
    os.makedirs(temp_directory_path, exist_ok=True)
    
    # Clear existing temp files
    mask_path = os.path.join(temp_directory_path, "temp.txt")
    if os.path.exists(mask_path):
        # Flush the main mask file
        open(mask_path, 'w').close()
    
    # Get detections from both images
    results1 = model.predict(source=img_left_path, classes=classes, conf=conf)
    # results1[0].show()
    # print(results1[0].boxes.xyxy.numpy())
    
    # results2 = model.predict(source=img_right_path, classes=classes, conf=conf)
    
    # Use results with more detections
    # results = results1 if len(results1[0].boxes) >= len(results2[0].boxes) else results2
    results = results1
    bbox = results[0].boxes.xyxy.numpy()
    # Save all detections to the mask file
    results[0].save_txt(mask_path)

    output_json = os.path.join(temp_directory_path,"points.json")
    
    
    detections = []
    image = cv2.imread(img_left_path)
    points_with_labels = get_polygon_points(image, mask_path, output_json)
    
    # Read all detections from the mask file
    with open(mask_path, 'r') as f:
        mask_lines = f.readlines()
    
    # Process each detection individually
    for i, line in enumerate(mask_lines):
        # Create temporary mask file with single detection
        single_mask_path = os.path.join(temp_directory_path, f"temp_{i}.txt")
        # Flush existing single mask file if it exists
        if os.path.exists(single_mask_path):
            open(single_mask_path, 'w').close()
            
        # Write new detection data
        with open(single_mask_path, 'w') as f:
            f.write(line)
        
        # Generate JSON for single detection
        output_json = os.path.join(temp_directory_path, f"points_{i}.json")
        # Flush existing JSON file if it exists
        if os.path.exists(output_json):
            open(output_json, 'w').close()
            
        points_with_labels = get_polygon_points(image, single_mask_path, output_json)
        
        # Get class label from the line (first number in the line)
        class_id = int(line.split()[0])
        
        # Generate point cloud for single detection
        point_cloud, seg_mask, labels, points_original, colors_original = generate_segmented_point_cloud(
            left_image_path=img_left_path,
            right_image_path=img_right_path,
            calibration_file_path=calibration_file_path,
            seg_json_path=output_json,
            max_z=max_z,
            target_class_labels=[class_id],
            save_point_cloud=False,
            visualize=False
        )
        detection = process_segmented_point_cloud(point_cloud, class_id, bbox[i,...])
        
        
        # If we found a valid centroid, create a Detection object
        if detection is not None:
            detections.append(detection)

        if visualize:
            # Visualize the clustered point cloud with centroids and labels
            visualize_point_cloud_with_centroids(point_cloud, [detection])
            # points = np.asarray(point_cloud.points)
            # visualize_original_pc_with_centroids(
            #     points=points_original,
            #     cluster_labels=labels,
            #     centroids=best_centroids,
            #     centroid_labels=best_cluster_labels,
            #     colors=colors_original
            # )
    if visualize:
        visualize_original_pc_with_centroids(
            points=points_original,
            cluster_labels=labels,
            centroids=[detection.centroid for detection in detections],
            centroid_labels=[detection.class_label for detection in detections],
            colors=colors_original
        )
        
    return detections

def cluster_from_stereo(model, classes, img_left_path, img_right_path, calibration_file_path, conf= 0.7, max_z: float=100.0, save_results= False, visualize = False):
    """
    Calculates pointcloud from images, must keep project image structure.

    Returns:
        centroid_coords: List[float, float, float]
    """
    # Get the current file's directory
    current_file_path = os.path.abspath(__file__)

    filename = os.path.basename(img_left_path)  # Get the filename: '0000000090.png'
    file_stem = os.path.splitext(filename)[0]  # Remove the extension: '0000000090'
    # Get the parent directory
    parent_directory = os.path.dirname(current_file_path)

    temp_directory_path = os.path.join(parent_directory,"temp")
    
    # Create the directory path
    os.makedirs(temp_directory_path, exist_ok=True)

    mask_path = os.path.join(temp_directory_path,"temp.txt")

    # Perform prediction

    results1 = model.predict(source=img_left_path, classes = classes,conf=conf)
    results2 = model.predict(source=img_right_path, classes = classes,conf=conf)

    if len(results1[0].names) >= len(results2[0].names):
        results = results1
    else:
        results = results2

    # Clear the file content (overwrite it)
    with open(mask_path, 'w') as file:
        pass  # This ensures the file is emptied before writing new content

    # Save the results to the file
    results[0].save_txt(mask_path)  # Save the results of the first prediction

    if visualize:
        results[0].show()
    results_directory_path = os.path.join(parent_directory,"results")
    
    if save_results:    
    
        # Create the directory path
        os.makedirs(results_directory_path, exist_ok=True)

        results[0].save(filename=os.path.join(results_directory_path,f"{file_stem}_detection.jpg"))  # display to screen
    
    image = cv2.imread(img_left_path)

    output_json = os.path.join(temp_directory_path,"points.json")

    # Save to JSON
    points_with_labels = get_polygon_points(image, mask_path, output_json)

    point_cloud, seg_mask, labels, points_original, colors_original = generate_segmented_point_cloud(
        left_image_path=img_left_path,
        right_image_path=img_right_path,
        calibration_file_path=calibration_file_path,
        seg_json_path=output_json,
        max_z=max_z,
        target_class_labels=None,
        save_point_cloud=save_results,
        save_path=os.path.join(results_directory_path,f'{file_stem}_segmented_point_cloud.ply'),
        visualize=False  # Set to True if you want to visualize
    )

    # Extract the number of clusters based on unique colors
    colors = np.asarray(point_cloud.colors)
    unique_colors = np.unique(colors, axis=0)
    n_clusters = len(unique_colors)

    # Grid search for DBSCAN parameters
    eps_values = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,0.6]) 
    min_samples_values = [50,75,100, 150, 200, 350, 300, 400, 500,750, 1000]
    min_samples_values = np.array(min_samples_values[::-1])
    min_samples_values = min_samples_values.astype(int)
    best_params = None
    best_centroids = None
    best_cluster_labels = None
    best_clustered_pcd = None

    detections = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            # print(f"Trying eps={eps}, min_samples={min_samples}...")
            try:
                centroids, cluster_labels, clustered_pcd, num_clusters = dbscan_with_labels_and_outlier_removal(
                    point_cloud=point_cloud,
                    labels=labels,
                    eps=eps,
                    min_samples=min_samples,
                )

                # Match number of clusters with expected count
                if num_clusters == n_clusters:
                    # print(f"Match found: eps={eps}, min_samples={min_samples}, clusters={num_clusters}")
                    best_params = (eps, min_samples)
                    best_centroids = centroids
                    best_cluster_labels = cluster_labels
                    best_clustered_pcd = clustered_pcd
                    break
            except Exception as e:
                print(f"Error with eps={eps}, min_samples={min_samples}: {e}")

        if best_params:
            break  # Exit outer loop if a match is found

    # Visualize the best result
    if best_clustered_pcd:
        print(f"Best parameters: eps={best_params[0]}, min_samples={best_params[1]}")
        
        # o3d.visualization.draw_geometries([best_clustered_pcd], window_name="Best Clustered Point Cloud")
        for i, (centroid, label) in enumerate(zip(best_centroids, best_cluster_labels)):
            # Print the label in the terminal for reference
            detections.append(Detection(centroid,label))
            print(f"Centroid {i}: Position={centroid}, Class={label}")
    else:
        print("No matching parameters found in the grid search.")

    if visualize and best_clustered_pcd:
        # Visualize the clustered point cloud with centroids and labels
        visualize_point_cloud_with_centroids(
            point_cloud=best_clustered_pcd,
            cluster_labels=labels,
            centroids=best_centroids,
            centroid_labels=best_cluster_labels
        )
        # points = np.asarray(point_cloud.points)
        visualize_original_pc_with_centroids(
            points=points_original,
            cluster_labels=labels,
            centroids=best_centroids,
            centroid_labels=best_cluster_labels,
            colors=colors_original
        )

    return detections

from main import *
if __name__=="__main__":
    # Load the model
    model = YOLO(r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\best.pt")

    # Test image
    img_left = r'..\34759_final_project_rect\seq_01\image_02\data\000006.png'
    img_right = r'..\34759_final_project_rect\seq_01\image_03\data\000006.png'
    calibration_file_path = r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\34759_final_project_rect\calib_cam_to_cam.txt"

    max_z = 30.0

    frame_detections = process_individual_detections(
            model=model,
            classes=[0,1,2],
            img_left_path=img_left,
            img_right_path=img_right,
            calibration_file_path=calibration_file_path,
            conf=0.7,
            max_z=max_z,
            visualize=True
        )
    
    # evaluate_sequence(predictions, "ground_truth.txt")
    
    
    
    print(len(frame_detections))
    # detections = cluster_from_stereo(model=model, classes = [0,1,2], img_left_path=img_left, img_right_path=img_right, calibration_file_path=calibration_file_path, conf=0.7,max_z=max_z, save_results=True, visualize = True)

    # View Parameters:
    # Extrinsic matrix:
    # [[    0.96548    -0.17072     0.19672    -0.83989]
    # [     0.2254      0.9261    -0.30253      2.7364]
    # [   -0.13053     0.33643     0.93262      2.3552]
    # [          0           0           0           1]]
    # Front vector: [    0.13053    -0.33643    -0.93262]
    # Lookat point: [    0.83989     -2.7364     -2.3552]
    # Up vector: [    -0.2254     -0.9261     0.30253]
    # Zoom: 60.0