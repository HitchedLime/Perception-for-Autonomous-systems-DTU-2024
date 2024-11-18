import cv2
import numpy as np


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

def generate_point_cloud(disp_map, Q, color_image=None, max_z = None):
    # Avoid division by zero
    disp_map[disp_map == 0] = 0.1
    disp_map[disp_map == -1] = 0.1

    points_3D = cv2.reprojectImageTo3D(disp_map, Q)
    
    # Also, ensure that disparity values are valid (greater than zero)
    valid_disp = disp_map > 0

    if max_z is not None:
        # Create a mask where Z-values are less than or equal to max_z
        mask = points_3D[..., 2] <= max_z

    # Combine the masks
    final_mask = np.logical_and(mask, valid_disp)

    points = points_3D[final_mask]

    # If a color image is provided, apply the mask to get colors for each point
    if color_image is not None:
        colors = color_image[final_mask]
        return points, colors
    else:
        return points

# Save to a PLY file
def save_point_cloud_to_ply(filename, points):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

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

    # Construct the Q matrix based on the revised stereo geometry
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, -fx],  # Note the negative sign here
        [0, 0, -1 / Tx, (cx - cx_right) / Tx]
    ])

    return Q

