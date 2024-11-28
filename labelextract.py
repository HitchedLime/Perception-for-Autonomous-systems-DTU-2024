import numpy as np
def parse_label_file(file_path):
    """
    Parses a text file containing object detections into a list of dictionaries.

    Parameters:
    - file_path (str): Path to the text file.

    Returns:
    - List[Dict]: A list of dictionaries where each dictionary represents a detection.
    """
    detections = []

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.strip().split()
            # Check if the line has the expected number of fields
            if len(fields) < 17:
                print(f"Warning: Line {line_number} has fewer fields than expected. Skipping.")
                continue

            try:
                detection = {
                    "frame": int(fields[0]),
                    "track_id": int(fields[1]),
                    "type": fields[2],
                    "truncated": float(fields[3]),
                    "occluded": int(fields[4]),
                    "alpha": float(fields[5]),
                    "bbox": {
                        "left": float(fields[6]),
                        "top": float(fields[7]),
                        "right": float(fields[8]),
                        "bottom": float(fields[9]),
                    },
                    "dimensions": {
                        "height": float(fields[10]),
                        "width": float(fields[11]),
                        "length": float(fields[12]),
                    },
                    "location": {
                        "x": float(fields[13]),
                        "y": float(fields[14]),
                        "z": float(fields[15]),
                    },
                    "rotation_y": float(fields[16]),
                }
                detections.append(detection)
            except (ValueError, IndexError) as e:
                print(f"Error processing line {line_number}: {line.strip()}")
                print(f"Error details: {e}")
                continue

    return detections


def filter_and_extract_locations(detections, frame=None, track_id=None, obj_type=None):
    """
    Filters detections by `frame`, `track_id`, and `type` and extracts locations as a NumPy array.

    Parameters:
    - detections (List[Dict]): List of detection dictionaries.
    - frame (int or List[int], optional): Frame number(s) to filter by.
    - track_id (int or List[int], optional): Track ID(s) to filter by.
    - obj_type (str or List[str], optional): Object type(s) (e.g., "Pedestrian", "Cyclist") to filter by.

    Returns:
    - np.ndarray: Array of filtered locations (N x 3) where each row is [x, y, z].
    """
    filtered_detections = [
        detection for detection in detections
        if (frame is None or (isinstance(frame, (list, tuple)) and detection["frame"] in frame) or detection["frame"] == frame)
        and (track_id is None or (isinstance(track_id, (list, tuple)) and detection["track_id"] in track_id) or detection["track_id"] == track_id)
        and (obj_type is None or (isinstance(obj_type, (list, tuple)) and detection["type"] in obj_type) or detection["type"] == obj_type)
    ]
    
    # Extract locations
    locations = np.array([list(det["location"].values()) for det in filtered_detections])
    
    return locations
