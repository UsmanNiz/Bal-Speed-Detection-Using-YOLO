# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""


import argparse
from audioop import bias
import os
import sys
from pathlib import Path
import numpy as np
from pandas import concat

# os.chdir("./yolov5")

# print(os.listdir())

# print(os.getcwd())
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from PyQt5.QtGui import *

@torch.no_grad()
class Detector:
    def __init__(self):
        self.weights="D:/FYP/C3D/yolov5/runs/train/fyp_weights/1_jan_2023/best.pt"  # model.pt path(s)
        self.source='vid.mp4' # file/dir/URL/glob, 0 for webcam
        # print(type(frame))
        # self.source = frame
        # self.frame = frame
        # self

        # ball
        # bat
        # wicket
        # pitch
        # shoe
        self.currBallXAxis = 0
        self.data='data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 480)  # inference size (height, width)
        self.conf_thres=0.25 # confidence threshold
        self.iou_thres=0.45 # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device= 0   # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=True  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False   # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / 'runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        # print(self.half)
        self.dnn=False  # use OpenCV DNN for ONNX inference

        self.stride = None
        self.pt = None
        self.model= None
        self.bs = None
        self.dataset = None
        self.names = None
        self.pred = None

        self.classifcation_point = {}
        self.ballFrames = 0
        self.FlagFrameStart = False
        self.FlagFrameStop = False
        self.previou = 0
        self.videoCapFlag = False
        self.videoCapCounter = 0

        self.ballspeed = 0



    def extractDetectorInformation(self, detections):
        self.classifcation_point = {}

        # print(detections)
        # print(detections.shape[1] == 6)
        # print(len(detections.size()))
        if detections.shape[1] == 6:
            # detections = detections.cpu().detach().numpy()

            for x1, y1, x2, y2, cls_conf, cls_pred in detections:

                cls_pred = int(cls_pred)

                if cls_pred not in self.classifcation_point.keys():
                    self.classifcation_point[cls_pred] = []

                self.classifcation_point[int(cls_pred)].append([x1, y1, x2, y2])

    def load_model(self):
         
        vidcap = cv2.VideoCapture("videos/vid5.mp4")
        self.fps = vidcap.get(cv2.CAP_PROP_FPS)

        print(self.fps)
        self.device = select_device(self.device)
        
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    def data_loader(self):
        # self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        self.bs = 1  # batch_size
        vid_path, vid_writer = [None] * self.bs, [None] * self.bs    
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def inference(self,frame,video_player_obj):
        # self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
        # for path, im, im0s, vid_cap, s in self.dataset:
        
        # self.frame.size()
        t1 = time_sync()
        frame = cv2.resize(frame,(640,480))
        im = torch.from_numpy(frame).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # cv2.resize(im,(640,480))
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

        im = im.permute(0,3,1,2)
        # print(im.shape)
        self.pred = self.model(im, augment=self.augment, visualize=self.visualize)

        t3 = time_sync()
        
    
        dt[1] += t3 - t2
        # NMS

        self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        im0 = frame.copy()
        ballspeed = self.logic()

        if ballspeed > 1:
            self.ballspeed = round(ballspeed)
            # self.ballspeed = round()
            self.videoCapFlag = True

        if self.videoCapFlag:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im0,"Ball Speed="+str(self.ballspeed),(100, 50),font, 1,(0, 255, 255),1,cv2.LINE_4)

            self.videoCapCounter += 1

        if self.videoCapCounter == 10:
            self.videoCapCounter = 0
            self.videoCapFlag= False

        dt[2] += time_sync() - t3
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(self.pred):  # per image
            seen += 1
            
            # p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
            
            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
            # Stream results
            im0 = annotator.result()
            im0 = cv2.resize(im0,(640,480))
            # cv2.imshow("Cricket Frames", im0)
            # cv2.waitKey(1)
            
            cv2.imwrite("frame.jpg",im0)
            video_player_obj.pixmap = QPixmap('frame.jpg')
            video_player_obj.label.setPixmap(video_player_obj.pixmap)
            cv2.waitKey(1)
            
        

    def logic(self):
        pred = self.pred[0].detach().cpu().numpy()
        self.extractDetectorInformation(pred)  
        # print(self.classifcation_point)
        # print("##") 

        pitchLengthInKM = 0.0201168
        # timeInHOURS = sec/360

        # print(c3d_pred)
        if 3 in self.classifcation_point.keys():
            
            # print("here in logic")      
            if 0 in self.classifcation_point.keys():
                ballBox = self.classifcation_point[0][0]
            else:
                ballBox = [0,0,0,0]
            self.prevBallXAxis = self.currBallXAxis
            self.currBallXAxis= ballBox[0]
            
            # print(self.currBallXAxis," AND ", self.prevBallXAxis,self.ballFrames)
            pitchBox = self.classifcation_point[3][0]
            # print(ballBox)
            # print(ballBox)
            # self.ballFrames += 1
            iou = self.handAndObjectIOUWithRespecToHandObject(ballBox,pitchBox)
            Ballspeed = 0
            # print(iou)
            # print(self.previou, ":::",iou)
            if self.previou == 0 and iou > 0.5:
                self.FlagFrameStart = True
                # self.ballFrames += 1
            elif self.previou > 0.5 and iou == 0:
                # print("in ELIFF")
                self.FlagFrameStart= False
                # self.ballFrames = 0

            # print(self.ballFrames)
            if self.ballFrames > 32:
                self.ballFrames = 0
            # if self.FlagFrameStart:
            if self.FlagFrameStart and abs(self.currBallXAxis - self.prevBallXAxis) < 10:
                self.ballFrames += 1
            else:
                if self.ballFrames != 0 and self.ballFrames>8:
                    
                    # if self.ballFrames == 11:
                    #     self.ballFrames += 1
                    # if self.ballFrames == 10:
                    #     self.ballFrames += 2

                        # print(self.ballFrames)
                    timeInSec = (self.ballFrames+3)/self.fps
                    # print(timeInSec)
                    timeInHours = timeInSec/3600
                    # print(timeInHours)
                    Ballspeed = pitchLengthInKM/timeInHours
        
                    print(Ballspeed , " Kmp/hr") 
                    self.ballFrames = 0
                # else:
                #     self.ballFrames = 0

            self.previou = self.handAndObjectIOUWithRespecToHandObject(ballBox,pitchBox)
            return Ballspeed
        
        return 0

        # for cls in self.classifcation_point.keys():
        #     if 

    def handAndObjectIOUWithRespecToHandObject(self, hand, object):
        xA = max(hand[0], object[0])
        yA = max(hand[1], object[1])
        xB = min(hand[2], object[2])
        yB = min(hand[3], object[3])

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        hand_area = (hand[2] - hand[0] + 1) * (hand[3] - hand[1] + 1)
        object_area = (object[2] - object[0] + 1) * (object[3] - object[1] + 1)

        iou = inter_area / float(hand_area)
        return iou

def main(video_player_obj):
    Detect = Detector()
    Detect.load_model()
    Detect.data_loader()
    cap = cv2.VideoCapture("videos/vid5.mp4")
    
    while(cap.isOpened()):    
        suc,frame = cap.read()
        if suc:
            Detect.inference(frame,video_player_obj)
        else:
            break
    # print("")
    # Detect.logic()



# if __name__ == "__main__":
#     # opt = parse_opt()
#     main(video_player_obj)
