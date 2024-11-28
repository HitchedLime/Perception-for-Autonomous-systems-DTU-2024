import os
import glob
from pointcloud_clustering import *
from tracking import *
from image_processing import *
from ultralytics import YOLO
from kalmanfilter import *
from datetime import datetime

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

def getImageSeq(path: str = "", seq: str = "", frame_count: int = 1):
    seq_path_list = {}

    # Get left image paths
    left_img_path = os.path.join(path, seq, "image_02", "data")
    left_img_files = sorted(glob.glob(os.path.join(left_img_path, "*.png")))[:frame_count]
    seq_path_list["left"] = left_img_files

    # Get right image paths
    right_img_path = os.path.join(path, seq, "image_03", "data")
    right_img_files = sorted(glob.glob(os.path.join(right_img_path, "*.png")))[:frame_count]
    seq_path_list["right"] = right_img_files

    seq_path_list["calibration"] = os.path.join(path, "calib_cam_to_cam.txt")

    seq_path_list["timestamps"] = parse_timestamps(os.path.join(path, seq, "image_02","timestamps.txt"))[:frame_count]

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

def visualize_tracking_with_pointcloud(points, colors, tracked_objects, window_name="Tracking Visualization",
                                     view_params=None):
    """
    Visualize point cloud with tracked object paths in real-time.
    
    Args:
        points (np.ndarray): Current frame's point cloud points
        colors (np.ndarray): Current frame's point cloud colors
        tracked_objects (list): List of TrackedObject instances
        window_name (str): Name of the visualization window
        view_params (dict): Optional view parameters with keys:
            - 'front': Camera front direction [x, y, z]
            - 'lookat': Point to look at [x, y, z]
            - 'up': Up vector [x, y, z]
            - 'zoom': Zoom factor (float)
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

if __name__ == "__main__":
    # model = YOLO(r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\fine_tuned_yolo.pt")
    model = YOLO(r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\yolo11x-seg.pt")

    rect_folder = r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\34759_final_project_rect"
    seq = "seq_02"
    frame_count = 9

    max_z=50.0

    # classes = [0,4,8]
    classes = [0,1,2]

    seq_list = getImageSeq(path=rect_folder, seq=seq, frame_count=frame_count)

    # Parameters
    cost_threshold = 10
    class_mismatch_penalty = 1000
    max_age = 20

    # Main tracking loop
    tracked_objects = []
    all_tracked_objects = []
    next_object_id = 0
    point_clouds = []
    for frame_idx, (frame_left, frame_right, current_timestamp) in enumerate(zip(seq_list["left"],seq_list["right"],seq_list["timestamps"])):
        print(f"\nProcessing frame {frame_idx + 1}/{len(seq_list['left'])}")
        
        calibration_file_path = seq_list["calibration"]
        current_detections = cluster_from_stereo(model, classes, frame_left, frame_right, calibration_file_path, conf= 0.7, max_z = max_z, save_results= False, visualize = False)
        print(f"Number of current detections: {len(current_detections)}")

        # print("Measured centroids: ")
        # [print(detection.centroid) for detection in current_detections]

        # Get original pointcloud for visualization
        point_cloud, seg_mask, labels, points_original, colors_original = generate_segmented_point_cloud(
            left_image_path=frame_left,
            right_image_path=frame_right,
            calibration_file_path=calibration_file_path,
            seg_json_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"temp","points.json"),
            max_z=max_z,
            target_class_labels=None,
            save_point_cloud=False,
            save_path="",
            visualize=False  # Set to True if you want to visualize
        )
        point_clouds.append(point_cloud)

        # Predict tracked object positions
        for obj in tracked_objects:
            obj.predict(current_timestamp)
        
        # Assign detections to tracked objects
        previous_detections = [Detection(obj.centroid, obj.class_label) for obj in tracked_objects]
        
        matches, unmatched_prev, unmatched_curr = assign_centroids(
            previous_detections,
            current_detections,
            cost_threshold=cost_threshold,
            class_mismatch_penalty=class_mismatch_penalty
        )
        
        print(f"Matches: {matches}")
        print(f"Unmatched previous detections: {unmatched_prev}")
        print(f"Unmatched current detections: {unmatched_curr}")
        
        # Update matched tracked objects
        for prev_idx, curr_idx in matches:
            tracked_objects[prev_idx].update(current_detections[curr_idx],current_timestamp)
        
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

        
        # visualize_tracking_with_pointcloud(
        #     pointcloud=point_clouds,
        #     tracked_objects=tracked_objects
        # )

        # o3d.visualization.draw_geometries(point_clouds)
        # Example: Bird's eye view
        scaler = 4
        view_params = {
            'front': [0.25, -0.2, -1],  # More top-down view
            'lookat': scaler * np.array([   -0.75  ,   0.59465    ,  6.0202]),
            'up': [0, -1, 0],
            'zoom': 0.225           # Even closer
        }

        # Example: Side view
        # view_params = {
        #     'front': [-1, 0, 0],    # Looking from positive X
        #     'lookat': [0, 0, 0],
        #     'up': [0, 0, 1],
        #     'zoom': 1.0
        # }

        # Call the function with view parameters
        visualize_tracking_with_pointcloud(points_original, colors_original, tracked_objects, 
                                        view_params=view_params)
    # After the loop, add remaining tracked objects
    all_tracked_objects.extend(tracked_objects)

    # Visualize after the run
    # Visualize current frame with trajectories
    
    visualize_tracked_objects_histories(all_tracked_objects)