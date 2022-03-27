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
