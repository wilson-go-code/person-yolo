# %%
# Setup
import json
import subprocess

import cv2 as cv
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchinfo import summary

from config import *
from utils import *

# %%
# Preprocessing, label creation
# process_label(train_label, train_savedir)
# process_label(valid_label, valid_savedir)

# %%
# inference

fp = "datasets/train/train/data/000000000036.jpg"

# sdk
def infer(path):
    imgpath = path
    model = YOLO("yolov8n.yaml")
    model.predict(imgpath, save=True)

# cli
# !yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source="dataset/train/train/data/000000000036.jpg"

def exec_cli(path):
    cmd = [
        "yolo",
        "task=detect",
        "mode=predict",
        "model=yolov8n.pt",
        "conf=0.25",
        f"source={path}"
    ]

    subprocess.run(cmd, check=True)


# %%
# infer(fp)
# exec_cli(fp)

# %%
# optimizer =  SGD, Adam, AdamW, NAdam, RAdam, RMSProp
# dropout
model = YOLO("yolov8n.yaml")
# model.train(
#     data="data.yaml", 
#     epochs=3, 
#     batch=4, 
#     workers=1,
#     device=[0],
#     pretrained=True,
#     amp=False,
#     save=True,
#     resume=True,
# )

# %%
# load
# model = YOLO("yolov8sgdbest.pt")
model = YOLO("yolov8adambest.pt")

# %%
# results = model(fp, save=True)

# %%
summary(model)
