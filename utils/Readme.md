## 1. EDA
* 팀원 각자가 진행한 EDA 파일

## 2. K-fold
* 작성자 : 김나영
* image_id를 기준으로 나눈 Ramdom Kfold 코드

## 3. Inference
* 작성자 : 신규범
* 대회 규격에 맞게 csv파일과 추후 작업을 위한 pkl 파일을 동시에 출력하는 파일
```
python inference.py [config 경로] [ckpt 경로]
```

## 4. Multilabel Stratfied K-fold
* 작성자 : 신규범
* Annotation의 class별 bbox의 개수 비율을 유지시키면서 K-fold를 수행하는 파일

```
python multilabel_Kfolds.py --path [파일을 저장할 위치] --n_split [n-fold]
```

