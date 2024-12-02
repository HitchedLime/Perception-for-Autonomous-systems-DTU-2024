import os
import glob
from pointcloud_clustering import *
from tracking import *
from image_processing import *
from ultralytics import YOLO
from kalmanfilter import *
from datetime import datetime
import shutil
import time


def clean_temp_folder():
    """Clean all files in the temp folder"""
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

def save_frame_data_with_metadata(points, colors, frame_idx, tracked_objects, output_dir):
    """
    Save point cloud data and visualization metadata separately.
    
    Args:
        points (np.ndarray): Point cloud points
        colors (np.ndarray): Point cloud colors (0-255)
        frame_idx (int): Frame number
        tracked_objects (list): List of TrackedObject instances
        output_dir (str): Output directory path
    """
    # Create subdirectories
    pc_dir = os.path.join(output_dir, "point_clouds")
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(pc_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # Save original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    pc_filename = f"frame_{frame_idx:04d}.ply"
    o3d.io.write_point_cloud(os.path.join(pc_dir, pc_filename), pcd)
    
    # Prepare visualization metadata
    metadata = {
        'frame_idx': frame_idx,
        'objects': []
    }
    
    # Define colors for different classes
    class_colors = {
        0: [1, 0, 0],  # Red for class 0
        1: [0, 1, 0],  # Green for class 1
        2: [0, 0, 1]   # Blue for class 2
    }
    
    # Add metadata for each tracked object
    for obj in tracked_objects:
        object_data = {
            'id': obj.id,
            'class_label': obj.class_label,
            'color': class_colors.get(obj.class_label, [0.5, 0.5, 0.5]),
            'centroid': obj.centroid.tolist(),
            'trail': [point.tolist() for point in obj.history],
            'bbox': obj.bbox.tolist()
        }
        metadata['objects'].append(object_data)
    
    # Save metadata
    meta_filename = f"frame_{frame_idx:04d}.json"
    with open(os.path.join(meta_dir, meta_filename), 'w') as f:
        json.dump(metadata, f)

def create_sphere_mesh(center, radius, color):
    """Create a sphere mesh at given center position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def play_point_cloud_sequence_with_trails(folder_path, view_params=None, frame_delay=0.1):
    """
    Play a sequence of point clouds with visualization of trails and centroids.
    
    Args:
        folder_path (str): Base path containing point_clouds and metadata folders
        view_params (dict): View parameters for visualization
        frame_delay (float): Delay between frames in seconds
    """
    pc_dir = os.path.join(folder_path, "point_clouds")
    meta_dir = os.path.join(folder_path, "metadata")
    
    # Get all PLY files
    ply_files = sorted(glob.glob(os.path.join(pc_dir, "frame_*.ply")))
    
    if not ply_files:
        print("No point cloud files found!")
        return
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Set render options for thicker lines
    render_option = vis.get_render_option()
    render_option.line_width = 20.0  # Increase line width
    render_option.point_size = 1.5   # Adjust point size if needed
    
    # Load first frame
    pcd = o3d.io.read_point_cloud(ply_files[0])
    vis.add_geometry(pcd)
    
    # Dictionaries to store geometries
    line_sets = {}
    centroid_spheres = {}
    
    # Set view parameters
    ctr = vis.get_view_control()
    if view_params:
        if 'lookat' in view_params:
            ctr.set_lookat(view_params['lookat'])
        if 'front' in view_params:
            ctr.set_front(view_params['front'])
        if 'up' in view_params:
            ctr.set_up(view_params['up'])
        if 'zoom' in view_params:
            ctr.set_zoom(view_params['zoom'])
    
    try:
        for ply_file in ply_files:
            frame_idx = int(os.path.splitext(os.path.basename(ply_file))[0].split('_')[1])
            meta_file = os.path.join(meta_dir, f"frame_{frame_idx:04d}.json")
            
            # Load point cloud and metadata
            new_pcd = o3d.io.read_point_cloud(ply_file)
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Update point cloud
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            vis.update_geometry(pcd)
            
            # Remove old geometries
            for line_set in line_sets.values():
                vis.remove_geometry(line_set, False)
            for sphere in centroid_spheres.values():
                vis.remove_geometry(sphere, False)
            line_sets.clear()
            centroid_spheres.clear()
            
            # Create new geometries for each object
            for obj_data in metadata['objects']:
                obj_id = obj_data['id']
                color = obj_data['color']
                trail_points = np.array(obj_data['trail'])
                centroid = np.array(obj_data['centroid'])
                
                # Create trail lines
                if len(trail_points) > 1:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(trail_points)
                    line_set.lines = o3d.utility.Vector2iVector(
                        [[i, i+1] for i in range(len(trail_points)-1)]
                    )
                    line_set.colors = o3d.utility.Vector3dVector([color] * (len(trail_points)-1))
                    line_sets[obj_id] = line_set
                    vis.add_geometry(line_set, False)
                
                # Create centroid sphere (smaller radius for better performance)
                sphere = create_sphere_mesh(centroid, radius=0.2, color=color)
                centroid_spheres[obj_id] = sphere
                vis.add_geometry(sphere, False)
            
            # Update visualization
            vis.poll_events()
            vis.update_renderer()
            time.sleep(frame_delay/10)
            
            print(f"Playing frame: {os.path.basename(ply_file)}", end='\r')
    
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    finally:
        vis.destroy_window()

def capture_point_cloud_sequence(folder_path, output_image_dir, view_params=None, width=1920, height=1080):
    """
    Capture each frame of the point cloud sequence as an image.
    
    Args:
        folder_path (str): Base path containing point_clouds and metadata folders
        output_image_dir (str): Directory to save captured images
        view_params (dict): View parameters for visualization
        width (int): Width of output image
        height (int): Height of output image
    """
    pc_dir = os.path.join(folder_path, "point_clouds")
    meta_dir = os.path.join(folder_path, "metadata")
    os.makedirs(output_image_dir, exist_ok=True)
    
    # Get all PLY files
    ply_files = sorted(glob.glob(os.path.join(pc_dir, "frame_*.ply")))
    
    if not ply_files:
        print("No point cloud files found!")
        return
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)  # Make window invisible
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.line_width = 5.0
    render_option.point_size = 2.0
    render_option.background_color = np.array([255, 255, 255])  # Black background
    
    # Load first frame
    pcd = o3d.io.read_point_cloud(ply_files[0])
    vis.add_geometry(pcd)
    
    # Dictionaries to store geometries
    line_sets = {}
    centroid_spheres = {}
    
    # Set view parameters
    ctr = vis.get_view_control()
    if view_params:
        if 'lookat' in view_params:
            ctr.set_lookat(view_params['lookat'])
        if 'front' in view_params:
            ctr.set_front(view_params['front'])
        if 'up' in view_params:
            ctr.set_up(view_params['up'])
        if 'zoom' in view_params:
            ctr.set_zoom(view_params['zoom'])
    
    try:
        for ply_file in ply_files:
            frame_idx = int(os.path.splitext(os.path.basename(ply_file))[0].split('_')[1])
            meta_file = os.path.join(meta_dir, f"frame_{frame_idx:04d}.json")
            
            # Load point cloud and metadata
            new_pcd = o3d.io.read_point_cloud(ply_file)
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Update point cloud
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            vis.update_geometry(pcd)
            
            # Remove old geometries
            for line_set in line_sets.values():
                vis.remove_geometry(line_set, False)
            for sphere in centroid_spheres.values():
                vis.remove_geometry(sphere, False)
            line_sets.clear()
            centroid_spheres.clear()
            
            # Create new geometries for each object
            for obj_data in metadata['objects']:
                obj_id = obj_data['id']
                color = obj_data['color']
                trail_points = np.array(obj_data['trail'])
                centroid = np.array(obj_data['centroid'])
                
                # Create trail lines
                if len(trail_points) > 1:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(trail_points)
                    line_set.lines = o3d.utility.Vector2iVector(
                        [[i, i+1] for i in range(len(trail_points)-1)]
                    )
                    line_set.colors = o3d.utility.Vector3dVector([color] * (len(trail_points)-1))
                    line_sets[obj_id] = line_set
                    vis.add_geometry(line_set, False)
                
                # Create centroid sphere
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
                sphere.translate(centroid)
                sphere.paint_uniform_color(color)
                centroid_spheres[obj_id] = sphere
                vis.add_geometry(sphere, False)
            
            # Update view and capture image
            vis.poll_events()
            vis.update_renderer()
            
            # Save image
            image_path = os.path.join(output_image_dir, f"frame_{frame_idx:04d}.png")
            vis.capture_screen_image(image_path, do_render=True)
            
            print(f"Captured frame: {frame_idx:04d}", end='\r')
    
    finally:
        vis.destroy_window()
        print("\nCapture complete!")

def load_frame_data(frame_idx, output_dir):
    """Load point cloud data for a frame"""
    frame_filename = f"frame_{frame_idx:04d}.ply"
    file_path = os.path.join(output_dir, frame_filename)
    
    if os.path.exists(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255.0  # Convert back to 0-255 range
        return points, colors
    return None, None
def parse_timestamps(timestamp_file):
    """
    Parse timestamps from file and return list of datetime objects
    """
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            # Split timestamp into main part and microseconds
            timestamp = line.strip()
            # Get everything before the decimal point and first 6 digits after (microseconds)
            main_part = timestamp.split('.')[0]
            microseconds = timestamp.split('.')[1][:6]  # Take only first 6 digits
            
            # Combine and parse
            dt_string = f"{main_part}.{microseconds}"
            dt = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(dt)
    
    return timestamps

def getImageSeq(path: str = "", seq: str = "", frame_start: int = 1, frame_count= None):
    seq_path_list = {}
    # if frame_count < frame_start:
    #     print("fail")
    # Get left image paths
    left_img_path = os.path.join(path, seq, "image_02", "data")
    left_img_files = sorted(glob.glob(os.path.join(left_img_path, "*.png")))
    if frame_count:
        left_img_files = left_img_files[frame_start-1:frame_count]
    else:
        frame_count= len(left_img_files)
    seq_path_list["left"] = left_img_files

    # Get right image paths
    right_img_path = os.path.join(path, seq, "image_03", "data")
    right_img_files = sorted(glob.glob(os.path.join(right_img_path, "*.png")))[frame_start-1:frame_count]
    seq_path_list["right"] = right_img_files

    seq_path_list["calibration"] = os.path.join(path, "calib_cam_to_cam.txt")

    seq_path_list["timestamps"] = parse_timestamps(os.path.join(path, seq, "image_02","timestamps.txt"))[frame_start-1:frame_count]

    return seq_path_list

def set_axes_equal(ax):
    """Set equal scaling for all axes on a 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate the ranges for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # Find the max range
    max_range = max(x_range, y_range, z_range)

    # Set the limits for all axes to be centered around the same midpoint
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])

def visualize_tracked_objects_histories(tracked_objects):
    """
    Visualize the trajectories of all tracked objects in 3D space after the run.

    Parameters:
    - tracked_objects: A list of TrackedObject instances.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Define colors for different classes
    class_colors = {
        0: 'red',       # Class label 0: red
        1: 'green',     # Class label 1: green
        2: 'blue',      # Class label 2: blue
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the history of each tracked object
    for obj in tracked_objects:
        history = np.array(obj.history)
        if history.size == 0:
            continue  # Skip if there's no history
        color = class_colors.get(obj.class_label, 'black')  # Default to black if class not specified
        ax.plot(history[:, 0], history[:, 1], history[:, 2], color=color, label=f'Object {obj.id}')
        # Optionally, mark the starting and ending positions
        ax.scatter(history[0, 0], history[0, 1], history[0, 2], color=color, marker='o', s=50)
        ax.scatter(history[-1, 0], history[-1, 1], history[-1, 2], color=color, marker='X', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectories of Tracked Objects')
    set_axes_equal(ax=ax)
    # Avoid duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    plt.show()

def visualize_tracking_with_pointcloud(points, colors, tracked_objects, window_name="Tracking Visualization", view_params=None):
    """
    Visualize current point cloud with historical tracking trails.
    
    Args:
        points (np.ndarray): Current frame's point cloud points
        colors (np.ndarray): Current frame's point cloud colors
        tracked_objects (list): List of TrackedObject instances
        window_name (str): Name of the visualization window
        view_params (dict): Optional view parameters
    """
    # Create visualization geometries list
    geometries = []
    
    # Add current point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors
    geometries.append(pcd)
    
    # Define colors for different classes
    class_colors = {
        0: [1, 0, 0],  # Red for class 0
        1: [0, 1, 0],  # Green for class 1
        2: [0, 0, 1]   # Blue for class 2
    }
    
    # Add trajectory lines and current position markers for each tracked object
    for obj in tracked_objects:
        if len(obj.history) > 1:
            # Create line set for trajectory
            line_points = np.array(obj.history)
            lines = [[i, i+1] for i in range(len(line_points)-1)]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # Set color based on object class
            color = class_colors.get(obj.class_label, [0.5, 0.5, 0.5])  # Default gray if class not found
            line_colors = [color for _ in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            
            geometries.append(line_set)
            
            # Add sphere at current position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
            sphere.translate(obj.centroid)
            sphere.paint_uniform_color(color)
            geometries.append(sphere)
    
    # Create visualizer and set view
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # Add all geometries
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Set view parameters if provided
    if view_params:
        ctr = vis.get_view_control()
        if 'lookat' in view_params:
            ctr.set_lookat(view_params['lookat'])
        if 'front' in view_params:
            ctr.set_front(view_params['front'])
        if 'up' in view_params:
            ctr.set_up(view_params['up'])
        if 'zoom' in view_params:
            ctr.set_zoom(view_params['zoom'])
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def match_detections(pred_centroids, gt_centroids, pred_boxes, gt_boxes, threshold=0.5):
    """
    Match predictions to ground truth using Hungarian algorithm
    
    Args:
        pred_centroids: Nx3 array of predicted 3D centroids
        gt_centroids: Mx3 array of ground truth 3D centroids
        pred_boxes: Nx4 array of predicted 2D boxes [x1,y1,x2,y2]
        gt_boxes: Mx4 array of ground truth 2D boxes
        threshold: IoU threshold for valid matches
    
    Returns:
        matches: List of (pred_idx, gt_idx) tuples
        unmatched_pred: List of unmatched prediction indices
        unmatched_gt: List of unmatched ground truth indices
    """
    if len(pred_centroids) == 0 or len(gt_centroids) == 0:
        return [], list(range(len(pred_centroids))), list(range(len(gt_centroids)))
    
    # Calculate cost matrix using centroid distances and box IoUs
    cost_matrix = np.zeros((len(pred_centroids), len(gt_centroids)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > threshold:
                # Use negative distance as cost (higher IoU = lower cost)
                dist = np.linalg.norm(pred_centroids[i] - gt_centroids[j])
                cost_matrix[i,j] = dist
            else:
                cost_matrix[i,j] = float('inf')
    
    # Use Hungarian algorithm for optimal matching
    from scipy.optimize import linear_sum_assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Filter out invalid matches (infinite cost)
    valid_matches = [(pred_idx, gt_idx) for pred_idx, gt_idx in zip(pred_indices, gt_indices) 
                    if cost_matrix[pred_idx, gt_idx] != float('inf')]
    
    matched_pred = {idx for idx, _ in valid_matches}
    matched_gt = {idx for _, idx in valid_matches}
    
    unmatched_pred = [idx for idx in range(len(pred_centroids)) if idx not in matched_pred]
    unmatched_gt = [idx for idx in range(len(gt_centroids)) if idx not in matched_gt]
    
    return valid_matches, unmatched_pred, unmatched_gt

def evaluate_boxes_and_centroids(predictions, ground_truth_file):
    """Evaluates boxes and centroids separately against ground truth"""
    from labelextract import parse_label_file, filter_and_extract_locations
    import numpy as np
    
    gt_data = parse_label_file(ground_truth_file)
    frame_ious = []
    frame_rmse = []
    max_frame = max([det["frame"] for det in gt_data])
    
    for frame in range(max_frame + 1):
        # Get ground truth
        gt_locations, gt_boxes = filter_and_extract_locations(gt_data, frame=frame)
        
        if frame >= len(predictions):
            frame_ious.append(0)
            frame_rmse.append(float('inf'))
            continue
            
        frame_preds = predictions[frame]
        if not frame_preds:
            frame_ious.append(0)
            frame_rmse.append(float('inf'))
            continue

        pred_centroids = np.array([det.centroid for det in frame_preds])
        pred_boxes = np.array([det.bbox for det in frame_preds])

        # Evaluate boxes
        best_ious = []
        for pred_box in pred_boxes:
            ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
            best_ious.append(max(ious) if ious else 0)
        frame_ious.append(np.mean(best_ious) if best_ious else 0)

        # Evaluate centroids
        min_distances = []
        for pred_centroid in pred_centroids:
            distances = [np.linalg.norm(pred_centroid - gt_centroid) for gt_centroid in gt_locations]
            min_distances.append(min(distances) if distances else float('inf'))
        frame_rmse.append(np.sqrt(np.mean(np.square(min_distances))) if min_distances else float('inf'))

    return frame_ious, frame_rmse

def plot_separate_metrics(frame_ious, frame_rmse):
    """Creates separate plots for IoU and RMSE metrics"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(frame_ious, 'b-')
    plt.xlabel('Frame')
    plt.ylabel('Best IoU')
    plt.title('Bounding Box IoU per Frame')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(frame_rmse, 'r-')
    plt.xlabel('Frame')
    plt.ylabel('RMSE (meters)')
    plt.title('Centroid RMSE per Frame')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def evaluate_predictions(predictions, ground_truth_file):
    from labelextract import parse_label_file, filter_and_extract_locations
    import numpy as np
    
    gt_data = parse_label_file(ground_truth_file)
    pred_ious = []
    pred_rmses = []
    max_frame = max([det["frame"] for det in gt_data])
    
    for frame in range(max_frame):
        gt_locations, gt_boxes = filter_and_extract_locations(gt_data, frame=frame)
        
        if frame >= len(predictions) or not predictions[frame]:
            continue
            
        frame_preds = predictions[frame]
        pred_boxes = np.array([det.bbox for det in frame_preds])
        pred_centroids = np.array([det.centroid for det in frame_preds])
        
        frame_ious = []
        frame_rmses = []
        
        for pred_idx, (pred_box, pred_centroid) in enumerate(zip(pred_boxes, pred_centroids)):
            # Get IoUs and find best match
            ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
            best_iou = max(ious) if ious else 0
            
            # Only calculate RMSE if IoU > 0
            if best_iou > 0:
                frame_ious.append(best_iou)
                # Get best matching ground truth index
                best_gt_idx = np.argmax(ious)
                # Calculate RMSE with corresponding ground truth centroid
                rmse = np.linalg.norm(pred_centroid - gt_locations[best_gt_idx])
                frame_rmses.append(rmse)
            
        pred_ious.append(frame_ious)
        pred_rmses.append(frame_rmses)

    return pred_ious, pred_rmses

def plot_metrics(pred_ious, pred_rmses, output_dir):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # Plot IoUs
    plt.subplot(1, 2, 1)
    for i in range(max(len(ious) for ious in pred_ious)):
        values = [ious[i] if i < len(ious) else None for ious in pred_ious]
        frames = range(len(pred_ious))
        valid_idx = [idx for idx, val in enumerate(values) if val is not None]
        valid_values = [values[idx] for idx in valid_idx]
        if valid_values:  # Only plot if there are valid values
            plt.plot(valid_idx, valid_values, label=f'Prediction {i+1}')
    
    plt.xlabel('Frame')
    plt.ylabel('IoU')
    plt.title('IoU per Prediction over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot RMSEs
    plt.subplot(1, 2, 2)
    for i in range(max(len(rmses) for rmses in pred_rmses)):
        values = [rmses[i] if i < len(rmses) else None for rmses in pred_rmses]
        frames = range(len(pred_rmses))
        valid_idx = [idx for idx, val in enumerate(values) if val is not None]
        valid_values = [values[idx] for idx in valid_idx]
        if valid_values:  # Only plot if there are valid values
            plt.plot(valid_idx, valid_values, label=f'Prediction {i+1}')
    
    plt.xlabel('Frame')
    plt.ylabel('RMSE (meters)')
    plt.title('RMSE per Prediction over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot to the output directory
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "metrics_plot.png")
    plt.savefig(plot_path)
    print(f"Metrics plot saved to: {plot_path}")

    # Save the IoU and RMSE values to a JSON file
    metrics_data = {
        "ious": pred_ious,
        "rmses": pred_rmses
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f)
    print(f"Metrics data saved to: {metrics_path}")

if __name__ == "__main__":
    # Clean temp folder at start
    clean_temp_folder()
    
    # Create output directory for frame data
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seq_01")
    os.makedirs(output_dir, exist_ok=True)
    
    # Your existing setup code
    model = YOLO(r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\best.pt")
    rect_folder = r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\34759_final_project_rect"
    seq = "seq_01"
    frame_start = 96
    frame_count = 98
    max_z = 50.0
    classes = [0, 1, 2]
    
    seq_list = getImageSeq(path=rect_folder, seq=seq, frame_start= frame_start, frame_count=frame_count)
    
    # Tracking parameters
    cost_threshold = 15
    class_mismatch_penalty = 1000
    max_age = 30
    
    # Main tracking loop
    tracked_objects = []
    all_tracked_objects = []
    next_object_id = 0
    frame_point_clouds = []
    
    tracked_objects_by_frame = []
    
    for frame_idx, (frame_left, frame_right, current_timestamp) in enumerate(zip(seq_list["left"], seq_list["right"], seq_list["timestamps"])):
        print(f"\nProcessing frame {frame_idx + 1}/{len(seq_list['left'])}")
        
        calibration_file_path = seq_list["calibration"]
        
        # Process detections
        current_detections = process_individual_detections(
            model=model,
            classes=[0, 1, 2],
            img_left_path=frame_left,
            img_right_path=frame_right,
            calibration_file_path=calibration_file_path,
            conf=0.8,
            max_z=max_z,
            visualize=False
        )


        
        # Generate and save point cloud
        point_cloud, seg_mask, labels, points_original, colors_original = generate_segmented_point_cloud(
            left_image_path=frame_left,
            right_image_path=frame_right,
            calibration_file_path=calibration_file_path,
            seg_json_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "points.json"),
            max_z=max_z,
            target_class_labels=None,
            save_point_cloud=False,
            visualize=False
        )
        
        # Save frame data
        save_frame_data_with_metadata(
            points_original, 
            colors_original, 
            frame_idx,
            tracked_objects, 
            output_dir,
        )
        
        # Predict tracked object positions
        for obj in tracked_objects:
            obj.predict(current_timestamp)
        
        # Assign detections to tracked objects
        matches, unmatched_prev, unmatched_curr = assign_centroids(
            tracked_objects,  # Pass tracked objects directly
            current_detections,
            cost_threshold=cost_threshold,
            class_mismatch_penalty=class_mismatch_penalty
        )
        
        # Update matched tracked objects
        for prev_idx, curr_idx in matches:
            tracked_objects[prev_idx].update(current_detections[curr_idx], current_timestamp)
        
        # Increase time_since_update for unmatched previous objects
        for idx in unmatched_prev:
            tracked_objects[idx].time_since_update += 1
        
        # Create new tracked objects for unmatched current detections
        for idx in unmatched_curr:
            obj = TrackedObject(current_detections[idx], next_object_id)
            tracked_objects.append(obj)
            next_object_id += 1
        
        # Remove lost tracked objects
        lost_objects = [obj for obj in tracked_objects if obj.time_since_update > max_age]
        tracked_objects = [obj for obj in tracked_objects if obj.time_since_update <= max_age]
        all_tracked_objects.extend(lost_objects)
        
        # Visualize current state with accumulated point clouds
        all_points = []
        all_colors = []
        for i in range(frame_idx + 1):
            points, colors = load_frame_data(i, output_dir)
            if points is not None and colors is not None:
                all_points.append(points)
                all_colors.append(colors)
        
    # After the loop, add remaining tracked objects and save final tracking data
    all_tracked_objects.extend(tracked_objects)
    
    # Final visualization
    visualize_tracked_objects_histories(all_tracked_objects)