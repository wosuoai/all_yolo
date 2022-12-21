import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from models.experimental import attempt_download, attempt_load
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np

# 自适应填充
def letterbox(img, new_shape=(640, 640), stride=32, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # print("dw, dh============", dw, dh)
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        # print("dw, dh", dw, dh)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape) # return coords = x1,y1,x2,y2
    return coords

device = "cpu"
cur_model = None
cur_model_path =None

def predict(img_path, model_path, conf_value=None, labelList=None, obj_iou_conf=None, shopId=None, branchId=None, batchNo=None,id=None, img_shape=None):
    global cur_model
    global cur_model_path
    # global boxes_detected
    labelList = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
            29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
            48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
            62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    if model_path != cur_model_path:
        model = attempt_load(model_path,device=device)
        model.half()
        cur_model_path = model_path  # 第一次已经加载过模型了，第二次还是调用就不用再加载了
        cur_model = model
    else:
        if cur_model_path is None:
            model = attempt_load(model_path,device=device)
            model.half()
            cur_model_path = model_path
            cur_model = model
        else:
            model = cur_model

    model.eval()

    with torch.no_grad():
        img0 = img_path

        img = letterbox(img0, new_shape=(640, 512), stride=int(max(model.stride)))[0]  # (960, 1280, 3)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]

    if conf_value is None and obj_iou_conf is not None:
        print("conf is default 0.6")
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=obj_iou_conf, agnostic=False)

    elif conf_value is not None and obj_iou_conf is None:
        pred = non_max_suppression(pred, conf_thres=conf_value, iou_thres=0.25, agnostic=False)

    elif obj_iou_conf is None and conf_value is None:
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.55, agnostic=False)

    else:
        pred = non_max_suppression(pred, conf_thres=conf_value, iou_thres=obj_iou_conf, agnostic=False)

    detections = []
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            print("img  ======={}含有{}个目标".format(img_path, int(len(det))))
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):

                xyxy_list = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                cls_name = labelList[cls.item()]
                conf = conf.item()
                detections.append([xyxy_list,cls_name,conf])

    return detections

if __name__ == '__main__':
    pass
    # detections = predict(img_path=,model_path=r"yolov5s.pt")
    # print(detections)


'''
cv2.rectangle(img=img, pt1=(int(xyxy_list[0]), int(xyxy_list[1])), pt2=(int(xyxy_list[2]), int(xyxy_list[3])),
                              color=(0, 255, 0), thickness=2, lineType=2)
                text = "{} - {}".format(cls_name, conf.item())
                cv2.putText(img, text, org=(int(xyxy_list[0]) + 10, int(xyxy_list[1]) + 10), fontFace=1, fontScale=1,
                            color=(0, 0, 255), thickness=2, lineType=2)
'''