import cv2
import torch
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# exp_name 수정하여 weights Load
exp_name = '0406_fold3_yolov5x6'

model = torch.hub.load('.', 'custom', path= f'./runs/train/{exp_name}/weights/best.pt', source='local')
model.conf = 0.001  # confidence threshold (0-1)
model.iou = 0.6  # NMS IoU threshold (0-1)

prediction_string = ['']  * 4871
image_id = [f'test/{i:04}.jpg' for i in range(4871)]
for i in tqdm(range(4871)):
    img = Image.open(f'../yolodata/images/test/{i:04}.jpg')

    results = model(img, size=1024, augment=True)
    for bbox in results.pandas().xyxy[0].values:
        xmin, ymin, xmax, ymax, confidence, cls, name = bbox
        prediction_string[i] += f'{cls} {confidence} {xmin} {ymin} {xmax} {ymax} '

raw_data ={
    'PredictionString' : prediction_string,
    'image_id' : image_id
}

dataframe = pd.DataFrame(raw_data)

# csv 형태로 저장
dataframe.to_csv(f'./runs/val/submission_tta_{exp_name}.csv', sep=',', na_rep='NaN', index=None)