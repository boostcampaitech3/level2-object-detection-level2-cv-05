# [AI Tech 3기 Level 2 P Stage] Object Detection
<img width="807" alt="화면 캡처 2022-04-13 200809" src="https://user-images.githubusercontent.com/90603530/163167628-4a440bd7-d974-449e-a05f-370d2cc1cfd8.png">

# ConVinsight 🧑‍💻
Convenience + insight : 이용자의 편의를 찾는 통찰력
## Member
| 김나영 | 신규범 | 이정수 | 이현홍 | 전수민 |  
| :-: | :-: | :-: | :-: | :-: |  
|[Github](https://github.com/dudskrla) | [Github](https://github.com/KyubumShin) | [Github](https://github.com/sw930718) | [Github](https://github.com/Heruing) | [Github](https://github.com/Su-minn) |

## Wrap Up Report 📑
💻 [ConvinSight level2-object-detection Notion](https://yummy-angle-b95.notion.site/CV-05-Wrap-Up-Report-3b569e864aee4c3abe90a2a2e5c9b643)

## Final Score 🏆
- Public mAP 0.7221 → Private mAP 0.7101
- Public 4위 → Private 4위
![그림2](https://user-images.githubusercontent.com/90603530/163172908-ac49bb77-5f9f-489a-a68d-273461837be1.jpg)


## Competition Process 🗓️
### Time Line
![간트차트](https://user-images.githubusercontent.com/90603530/163168369-8d26a3fe-8858-4c4f-b136-f43306027e7f.jpg)

### Project Outline 

> Data 
- [x] Data EDA
- [x] Data Argumentation
- [x] Multilabel Stratifiedkfold
- [x] Oversampling
> Model
- [x] Cascade RCNN with Various Backbone
- [x] YOLO (v5, R)
- [x] Soft NMS, NMS
- [x] GIoU, DIoU, CIoU
> Ensemble
- [x] Ensemble (WBF)
- [ ] Classfication
- [x] tile

### Folder Structure 📂
```
📂 detection/
│
├── 📂 baseline
│      │ 
│      ├── 📂 Swin_Transformer_Object_Detection
│      │    └── 📂 configs
│      │    └── 📂 p-stage
│      │         ├── 📂 __base__
│      │         │    ├── 📑 cascade_rcnn_swin_Base_fpn.py
│      │         │    └── 📑 cascade_rcnn_swin_Large_fpn.py
│      │         └── 📑 setup.py
│      │ 
│      ├── 📂 custom_configs
│      │    └── 📂 CNN
│      │         └── 📑 detectors_cascade_rcnn_resnext101_fpn.py
│      │ 
│      ├── 📂 efficientdet
│      │    └── 📂 effdet
│      │         ├── 📂 data
│      │         │    ├── 📑 dataset_config.py
│      │         │    └── 📑 transforms.py
│      │         └── 📑 train.py
│      │ 
│      ├── 📂 utils
│      │    ├── 📂 Compute_mean_std
│      │    ├── 📂 EDA
│      │    ├── 📂 EfficientDet_utils
│      │    ├── 📂 K-Fold
│      │    ├── 📂 inference
│      │    ├── 📂 multilabel_Kfolds
│      │    ├── 📂 oversampling
│      │    ├── 📂 pseudo_label
│      │    ├── 📑 csv2json.py
│      │    └── 📑 label_cleansing.py
│      │ 
│      ├── 📂 yolodata
│      ├── 📂 yolor
│      └── 📂 yolov5
│           ├── 📂 models
│           │    └── 📑 yolo.py
│           ├── 📂 utils
│           │    └── 📑 augmentations.py
│           ├── 📑 inference.py
│           └── 📑 train.sh
│      
└── 📂 dataset
```
