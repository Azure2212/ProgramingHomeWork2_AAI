import sys
import os
import random
root_project_path = os.getcwd()
# print(root_project_path)
sys.path.append(f'{root_project_path}/YOLOV12')
from ultralytics.models import YOLO
img_path_2_save = f"{root_project_path}/outputs/YOLOV12_demonstrate_output_{random.randint(1, 100000)}.png"

def predict_one_by_YoLoV12(image_path, device = 'cpu'):
    model = YOLO(f'{root_project_path}/YOLOV12/yolov12m.pt').to(device)

    results = model(image_path)

    print(f"Predicted image saved to {img_path_2_save}")
    for r in results:
        r.save(filename=img_path_2_save)
    return img_path_2_save
    