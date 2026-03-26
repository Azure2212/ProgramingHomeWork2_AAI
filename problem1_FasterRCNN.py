import sys
import os
import torch as t
root_project_path = os.getcwd()
# print(root_project_path)
sys.path.append(f'{root_project_path}/FasterRCNN')
from pytorch.FasterRCNN import visualize
from pytorch.FasterRCNN.datasets import image
from pytorch.FasterRCNN.models import resnet
from pytorch.FasterRCNN.models.faster_rcnn import FasterRCNNModel

class_index_to_name = {
    0:  "background",
    1:  "aeroplane",
    2:  "bicycle",
    3:  "bird",
    4:  "boat",
    5:  "bottle",
    6:  "bus",
    7:  "car",
    8:  "cat",
    9:  "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
  }

def predict(model, image_data, image, show_image, output_path):
  image_data = t.from_numpy(image_data).unsqueeze(dim = 0).cuda()
  scored_boxes_by_class_index = model.predict(image_data = image_data, score_threshold = 0.7)
  visualize.show_detections(
    output_path = output_path,
    show_image = show_image,
    image = image,
    scored_boxes_by_class_index = scored_boxes_by_class_index,
    class_index_to_name = class_index_to_name
  )

def predict_one(model, url, show_image, output_path):
  image_data, image_obj, _, _ = image.load_image(url = url, preprocessing = model.backbone.image_preprocessing_params, min_dimension_pixels = 600)
  predict(model = model, image_data = image_data, image = image_obj, show_image = show_image, output_path = output_path)

def predict_one_by_FasterRCNN(image_path):
    num_classes = 21
    backbone = resnet.ResNetBackbone(architecture = resnet.Architecture.ResNet50)
    model = FasterRCNNModel(num_classes = num_classes, backbone=backbone).cuda()
    