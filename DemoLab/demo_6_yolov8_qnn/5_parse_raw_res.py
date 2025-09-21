import cv2
import os, sys
import subprocess
import numpy as np
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
from spectrautils.print_utils import *
cfg = get_cfg_defaults()

class_names = cfg.DIPOORLET.COCO_labels


info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "output_shape": [1, 84, 8400], # 1. 添加缺失的 output_shape
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.5,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
}

def LetterBox(img, new_shape):
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    return img

def preprocess(img_path, info):
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    info.update({"img_height": img_height, "img_width": img_width})
    # img = LetterBox(img, (info["input_width"], info["input_height"]))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.array(img) / 255.0
    # img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, axis=0).astype(np.float32)
    return [img], info
    

def postprocess(outputs, info):
    outputs = np.transpose(np.squeeze(outputs))
    boxes = []
    scores = []
    class_ids = []
    for i in range(len(outputs)):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score > info["confidence_thres"]:
            x, y, w, h = outputs[i][:4]
            x1 = x - w / 2
            y1 = y - h / 2
            boxes.append([x1, y1, w, h])
            scores.append(max_score)
            class_id = np.argmax(classes_scores)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, info["confidence_thres"], info["iou_thres"])

    detections = []
    for i in indices:
        box = boxes[i]
        gain = min(info["input_width"] / info["img_width"], info["input_height"] / info["img_height"])
        pad = (
            round((info["input_width"] - info["img_width"] * gain) / 2 - 0.1),
            round((info["input_height"] - info["img_height"] * gain) / 2 - 0.1),
        )
        x1 = (box[0] - pad[0]) / gain
        y1 = (box[1] - pad[1]) / gain
        w = box[2] / gain
        h = box[3] / gain
        score = scores[i]
        class_id = class_ids[i]
        detection = [class_id, info["class_names"][class_id], score.astype(np.float64), x1, y1, w, h]
        detections.append(detection)
    detections.sort(key=lambda x: x[2], reverse=True)
    if len(detections) > info["max_det"]:
        detections = detections[:info["max_det"]]
    return detections, info


def show_results_single_img(img_path, results, class_names, save_path):
    img = cv2.imread(img_path)
    for result in results:
        class_id, class_name, score, x1, y1, w, h = result
        (label_width, label_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
        cv2.putText(img, class_name, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, img)
        
             
def main(info, mode):
    img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000556000.jpg"
    _, info = preprocess(img_path, info)
    
    raw_file_path = f"{log_dir}/Result_0/output0.raw"
    raw_data = np.fromfile(raw_file_path, dtype=np.float32)
    raw_data = raw_data.reshape(info["output_shape"])
    
    results, info = postprocess(raw_data, info)
    
    show_results_single_img(img_path, results, class_names, f"{log_dir}/test_res_qnn_{mode}_556000.jpg")
    print_colored_text(f"pic saved in :\n{log_dir}/test_res_qnn_{mode}_556000.jpg", "green")

    

    
if __name__ == "__main__":
    # log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_int8_1000_20250920_215500"
    log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_mixed_20250921_190437"
    
    # log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_fp16_20250921_010631"
    img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000556000.jpg"


    # mode = "fp16"
    # mode = "int8"
    mode = "mixed"
    main(info, mode)


