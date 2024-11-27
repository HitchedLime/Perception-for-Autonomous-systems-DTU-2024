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
    voxel_size = 0.05
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

def visualize_point_cloud_with_centroids(
    point_cloud, cluster_labels, centroids, centroid_labels, unique_label_colors=None
):
    """
    Visualize a point cloud with clusters and centroids using spheres for centroids.

    Args:
        point_cloud (o3d.geometry.PointCloud): The clustered point cloud.
        cluster_labels (np.ndarray): Cluster labels for each point.
        centroids (np.ndarray): Centroid positions for each cluster.
        centroid_labels (np.ndarray): Labels assigned to each centroid.
        unique_label_colors (dict or None): Predefined colors for clusters and centroids.
    """
    # Generate unique colors for each cluster if not provided
    if unique_label_colors is None:
        unique_labels = np.unique(cluster_labels)
        unique_label_colors = {
            label: np.random.rand(3) for label in unique_labels if label >= 0
        }
    
    # Assign colors to each point based on its cluster label
    points = np.asarray(point_cloud.points)
    colors = np.array([
        unique_label_colors[label] if label >= 0 else [0, 0, 0]  # Black for outliers
        for label in cluster_labels
    ])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Create a list of geometry objects for visualization
    geometries = [point_cloud]

    # Add centroids as spheres
    for i, (centroid, label) in enumerate(zip(centroids, centroid_labels)):
        # Create a sphere for the centroid
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.translate(centroid)  # Move sphere to centroid position
        sphere.paint_uniform_color([0, 0, 0])  # Use cluster color
        geometries.append(sphere)

    # Visualize the point cloud and centroids
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud with Centroids",
    )

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
        sphere_color = unique_label_colors.get(label, [0, 0, 0])
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

from tracking import Detection

def cluster_from_stereo(model, classes, img_left_path, img_right_path, calibration_file_path, conf= 0.7, save_results= False, visualize = False):
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

    results = model.predict(source=img_left_path, classes = classes,conf=conf)

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
        max_z=30.0,
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


if __name__=="__main__":
    # Load the model
    model = YOLO(r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\fine_tuned_yolo.pt")

    # Test image
    img_left = r'..\34759_final_project_rect\seq_02\image_02\data\0000000143.png'

    best_centroids, best_cluster_labels = cluster_from_stereo(model=model, classes = [0,1,2,3,4,5,6,7,8], img_left=img_left, conf=0.7,save_results=True, visualize = True)