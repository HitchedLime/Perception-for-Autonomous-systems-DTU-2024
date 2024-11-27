import os
import glob
from pointcloud_clustering import *
from tracking import *
from image_processing import *
from ultralytics import YOLO

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

    return seq_path_list

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

    # Avoid duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    plt.show()


if __name__ == "__main__":
    # model = YOLO(r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\fine_tuned_yolo.pt")
    model = YOLO(r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\yolo11x-seg.pt")

    rect_folder = r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\34759_final_project_rect"
    seq = "seq_01"
    frame_count = 50

    # classes = [0,4,8]
    classes = [0,1,2]

    seq_list = getImageSeq(path=rect_folder, seq=seq, frame_count=frame_count)


    # Parameters
    cost_threshold = 50
    class_mismatch_penalty = 1000
    max_age = 5

    # Main tracking loop
    tracked_objects = []
    all_tracked_objects = []
    next_object_id = 0

    for frame_idx, (frame_left, frame_right) in enumerate(zip(seq_list["left"],seq_list["right"])):
        print(f"\nProcessing frame {frame_idx + 1}/{len(seq_list['left'])}")
        
        calibration_file_path = seq_list["calibration"]
        current_detections = cluster_from_stereo(model, classes, frame_left, frame_right, calibration_file_path, conf= 0.7, save_results= False, visualize = False)
        print(f"Number of current detections: {len(current_detections)}")

        # Predict tracked object positions
        for obj in tracked_objects:
            obj.predict()
        
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
            tracked_objects[prev_idx].update(current_detections[curr_idx])
        
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

# After the loop, add remaining tracked objects
all_tracked_objects.extend(tracked_objects)

# Visualize after the run
visualize_tracked_objects_histories(all_tracked_objects)