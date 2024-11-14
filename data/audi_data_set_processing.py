import os
import sys
import glob
import json
import shutil

import cv2
import ultralytics
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
                    x, y, w, h = cv2.boundingRect(contour)

                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    width_yolo = w / width
                    height_yolo = h / height


                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {width_yolo} {height_yolo}")


        output_file_name = os.path.splitext(os.path.basename(label_image_path))[0] + '.txt'
        output_file_path = os.path.join(output_dir, output_file_name)


        with open(output_file_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')


base_dir = 'data/camera_lidar_semantic/camera_lidar_semantic'
# labels_paths=glob.glob(os.path.join(base_dir, '**/label/**/*.png'), recursive=True)


# print(len(labels_paths))
# generate_yolo_annotations(
#     class_list_path="data/camera_lidar_semantic/camera_lidar_semantic/class_list.json",
#     label_image_paths=labels_paths,
#     output_dir='labels'
# )
imgs_paths = glob.glob(os.path.join(base_dir, '**/camera/**/*.png'), recursive=True)
labels_paths= glob.glob(os.path.join('labels' ,'*.txt'), recursive=True)





print(f"Total images found: {len(imgs_paths)}")
print(f"Total labels found: {len(labels_paths)}")

## Split into training and validation sets (80% train, 20% val)
train_size = int(0.8 * len(imgs_paths))
train_images = imgs_paths[:train_size]
val_images = imgs_paths[train_size:]
train_labels = labels_paths[:train_size]
val_labels = labels_paths[train_size:]

# Create directories for train and val
output_dir = 'output'  # Change this to your desired output directory
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