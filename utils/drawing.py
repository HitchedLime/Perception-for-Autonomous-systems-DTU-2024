import cv2


def draw_bounding_boxes(image_path, annotations_file, display=True, save_path=None):
    """
    Draw bounding boxes on the image based on YOLO-style annotations.

    Parameters:
    - image_path: str, path to the image file.
    - annotations_file: str, path to the annotations file.
    - display: bool, whether to display the image with bounding boxes.
    - save_path: str or None, path to save the annotated image. If None, the image won't be saved.

    Returns:
    - annotated_image: numpy array, the image with bounding boxes drawn.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Get image dimensions
    height, width, _ = image.shape

    # Read the annotations from the file
    with open(annotations_file, 'r') as f:
        annotations = f.readlines()

    # Draw the bounding boxes
    for annotation in annotations:
        class_id, x_center, y_center, box_width, box_height = map(float, annotation.strip().split())

        # Convert YOLO format back to pixel values
        x_center_pixel = int(x_center * width)
        y_center_pixel = int(y_center * height)
        box_width_pixel = int(box_width * width)
        box_height_pixel = int(box_height * height)

        # Calculate top-left corner of the bounding box
        x1 = int(x_center_pixel - (box_width_pixel / 2))
        y1 = int(y_center_pixel - (box_height_pixel / 2))
        x2 = int(x_center_pixel + (box_width_pixel / 2))
        y2 = int(y_center_pixel + (box_height_pixel / 2))

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image, f'Class {int(class_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Optionally display the image
    if display:
        cv2.imshow('Annotated Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally save the annotated image
    if save_path:
        cv2.imwrite(save_path, image)

    return image
if __name__=="__main__":
    path_img ="data/camera_lidar_semantic/camera_lidar_semantic/20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000000091.png"
    path_label="annotations.txt"
# Example usage
    annotated_image = draw_bounding_boxes(path_img, path_label, display=True, save_path='annotated_image.jpg')
