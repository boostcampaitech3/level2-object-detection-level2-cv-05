## 1. EDA
* 팀원 각자가 진행한 EDA 파일

## 2. K-fold
* image_id를 기준으로 나눈 Ramdom Kfold 코드

## 3. Inference
* 대회 규격에 맞게 csv파일과 추후 작업을 위한 pkl 파일을 동시에 출력하는 파일
```
python inference.py [config 경로] [ckpt 경로]
```

## 4. Multilabel Stratfied K-fold
* Annotation의 class별 bbox의 개수 비율을 유지시키면서 K-fold를 수행하는 파일

```
python multilabel_Kfolds.py --path [파일을 저장할 위치] --n_split [n-fold]
```

## 5. Compute mean std
* dataset의 pixel mean과 std를 계산하는 notebook file

## 6. label_clensing
* coco-annotator로 만든 label 데이터의 image, annotation id를 재정렬하고 index를 0번부터 시작하도록 재설정하는 파일
```
python label_cleansing.py [json_file] [new json file]
```

## 7. Augmentation for Oversampling
* Oversampling을 위한 image generate 파일
```
python crop_box_make_patch.py [json file] --middle-range [MIDDLE] --small-ratio [SMALL RATIO] --middle-ratio [MIDDLE RATIO] --eps [epsilon] --name [DIR NAME]
```
* json file : oversampling할 label 파일
* MIDDLE : middle box의 area 범위, Default : 10000
* SMALL RATIO : small box의 oversampling 배수 parameter
* MIDDLE RATIO : middle box의 oversampling 배수 parameter
* epsilon : 다양한 background 이미지를 위한 random 오차
* name : 생성한 이미지를 저장할 폴더

총 Oversampling할 box는 아래와 같이 연산됨

Patch Labels = SMALL RATIO * small boxs + MIDDLE RATIO * middle boxs
 
여기서 7~20개 random sampling을 통해서 임의의 이미지를 생성