import sys
sys.path.append(r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\yolov7_BS")
import torch
from detect2 import main
import matplotlib.pyplot as plt
import cv2
img_path = r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\34759_final_project_rect\seq_01\image_02\data"
model_path = r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\yolov7\yolov7.pt"
output_path = r"C:\Users\szakt\Desktop\DTU\Perception\FinalProject\Output"
main()