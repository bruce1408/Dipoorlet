import onnx
from onnx import utils

# Specify the path to the original ONNX model
model_path = "/mnt/share_disk/bruce_trie/onnx_models/modelv5_0915.onnx"

# Specify the path where the extracted sub-model should be saved
output_head_path = "/mnt/share_disk/bruce_trie/onnx_models/modelv5_extract_head.onnx"
output_backbone_path = "/mnt/share_disk/bruce_trie/onnx_models/modelv5_extract_backbone.onnx"

# Define the input and output tensors of the sub-model
input_names = ["input.1700"
               ]
# output_names = ["output_tensor_name"]
output_names=[
        "heatmap",
        "reg",
        "dim",
        "rot"
      ]

input_names_s1 =  ['img_front_fisheye', 
    'img_right_fisheye', 
    'img_rear_fisheye', 
    'img_left_fisheye', 
    'img_front_short', 
    'indice_front_fisheye', 
    'indice_right_fisheye', 
    'indice_rear_fisheye', 
    'indice_left_fisheye', 
    'indice_front_short'
    ]

outputsnames_s1 =  ['input.1700']

# Extract the sub-model
utils.extract_model(model_path, output_backbone_path, input_names_s1, outputsnames_s1)
utils.extract_model(model_path, output_head_path, input_names, output_names)