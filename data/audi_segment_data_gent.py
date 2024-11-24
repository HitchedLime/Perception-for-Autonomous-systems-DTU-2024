import os
import glob
import json
import shutil
import cv2
import numpy as np

def generate_yolo_annotations(class_list_path, label_image_paths, output_dir):
    with open(class_list_path, "r") as class_list:
        class_label_dict = json.load(class_list)

    class_index_dict = {label: idx for idx, label in enumerate(class_label_dict.values())}

    color_to_class = {
        tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)): class_index_dict[label]
        for color, label in class_label_dict.items()
    }

    os.makedirs(output_dir, exist_ok=True)

    for label_image_path in label_image_paths:
        label_image = cv2.imread(label_image_path)
        if label_image is None:
            print(f"Warning: Could not read image {label_image_path}. Skipping.")
            continue

        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        height, width, _ = label_image.shape
        yolo_annotations = []

        for color, class_id in color_to_class.items():
            mask = cv2.inRange(label_image, np.array(color), np.array(color))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 0:
                    # Normalize contour points
                    normalized_points = []
                    for point in contour:
                        x, y = point[0]
                        normalized_x = x / width
                        normalized_y = y / height
                        normalized_points.append(f"{normalized_x} {normalized_y}")

                    # Create the annotation string
                    annotation_line = f"{class_id} " + " ".join(normalized_points)
                    yolo_annotations.append(annotation_line)

        output_file_name = os.path.splitext(os.path.basename(label_image_path))[0] + '.txt'
        output_file_path = os.path.join(output_dir, output_file_name)

        with open(output_file_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')





def draw_segmentation_from_annotations(image_path, annotation_path, class_list_path):
    # Load the class list
    with open(class_list_path, "r") as class_list:
        class_label_dict = json.load(class_list)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return

    height, width, _ = image.shape

    # Read the annotations
    with open(annotation_path, "r") as f:
        annotations = f.readlines()

    # Create a color map for visualization
    color_map = {idx: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                 for idx in range(len(class_label_dict))}

    # Draw each annotation
    for annotation in annotations:
        parts = list(map(float, annotation.split()))
        class_id = int(parts[0])
        points = parts[1:]

        # Convert normalized points back to pixel coordinates
        pixel_points = [(int(x * width), int(y * height)) for x, y in zip(points[::2], points[1::2])]

        # Draw the contour
        if len(pixel_points) > 0:
            cv2.polylines(image, [np.array(pixel_points)], isClosed=True, color=color_map[class_id], thickness=2)

    # Display the image with drawn segments
    cv2.imshow("Segmentation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    base_dir = 'data/part'

    labels_paths = glob.glob(os.path.join(base_dir, '**/label/**/*.png'), recursive=True)[0:5000]

    print(len(labels_paths))
    generate_yolo_annotations(
        class_list_path="data/class_list.json",
        label_image_paths=labels_paths,
        output_dir='labels'
    )
    imgs_paths = glob.glob(os.path.join(base_dir, '**/camera/**/*.png'), recursive=True)[0:5000]
    labels_paths = glob.glob(os.path.join('labels', '*.txt'), recursive=True)[0:5000]



    print(f"Total images found: {len(imgs_paths)}")
    print(f"Total labels found: {len(labels_paths)}")

    # Split into training and validation sets (80% train, 20% val)
    train_size = int(0.8 * len(imgs_paths))
    train_images = imgs_paths[:train_size]
    val_images = imgs_paths[train_size:]
    train_labels = labels_paths[:train_size]
    val_labels = labels_paths[train_size:]


    output_dir = 'audi'
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

    # Move training images and labels
    for img_path, label_path in zip(train_images, train_labels):
        shutil.move(img_path, os.path.join(output_dir, 'images/train', os.path.basename(img_path)))
        shutil.move(label_path, os.path.join(output_dir, 'labels/train', os.path.basename(label_path)))

    # Move validation images and labels
    for img_path, label_path in zip(val_images, val_labels):
        shutil.move(img_path, os.path.join(output_dir, 'images/val', os.path.basename(img_path)))
        shutil.move(label_path, os.path.join(output_dir, 'labels/val', os.path.basename(label_path)))

    print("Files have been moved to the respective train and val directories.")

    # image_path = 'data/part/20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000000091.png'  # Replace with your image path
    # annotation_path = 'labels/20180807145028_label_frontcenter_000000091.txt'  # Replace with your annotation path
    # class_list_path = 'data/class_list.json'  # Replace with your class list path

    # draw_segmentation_from_annotations(image_path, annotation_path, class_list_path)
