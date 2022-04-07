# EfficientDet
이 Repository는 EfficientDet-Pytorch을 기반으로 만들어졌습니다

📗 [EfficientDet-Pytorch](https://github.com/rwightman/efficientdet-pytorch)   
📙 Boostcamp AI Tech 3기 CV _ EfficientDet 강의자료 

## **Setup for running**
### **가상환경 생성**
```bash
conda create -n [env name] --clone detection
conda activate [env name]
``` 
ex.  
conda create -n effdet --clone detection  
conda activate effdet

### **필수 Package 설치**
실행은 baseline 에서  
```bash
git clone https://github.com/rwightman/efficientdet-pytorch.git  
cd efficientdet-pytorch  
pip install -r requirements.txt
pip install -v -e .
```

## **Pre treatment**  
### **categories & annotations 수정**
대회 데이터의 label은 0-9까지. 코드 내 label은 1-10    
따라서, 대회에서 제공된 categories과 annotations내 label값을 +1 시켜주었다.  
* [Modify Label](https://github.com/boostcampaitech3/level2-object-detection-level2-cv-05/blob/utils/utils/modify_label.py)      

실행 후 ) eff_train.json & eff_val.json & eff_test.json 생성 

### ***적용할 json 파일 변경***  
사용데이터 coco 2017. 따라서 effdet/data/dataset_config.py파일의 해당 json파일 부분 수정   

```bash  
class Coco2017Cfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename=[train json파일], img_dir='', has_labels=True),
        val=dict(ann_filename=[val json파일], img_dir='', has_labels=True),
        test=dict(ann_filename=[test json파일], img_dir='', has_labels=False)
    ))
```
ex.   
train=dict(ann_filename='./eff_train.json', img_dir='', has_labels=True),   
val=dict(ann_filename='./eff_val.json', img_dir='', has_labels=True),   
test=dict(ann_filename='./eff_test.json', img_dir='', has_labels=False)   

## **Train** 
실행은 efficientdet-pytorch 에서  
`
python train.py [dataset 경로] --model tf_efficientdet_d4_ap --dataset coco -b 4 --amp --lr .008 --opt momentum --model-ema --model-ema-decay 0.9966 --epochs 70 --num-classes 10 --tta 1 --pretrained   
`     
ex.  
python train.py /opt/ml/detection/dataset --model tf_efficientdet_d4_ap --dataset coco 
-b 4 --amp --lr .008 --opt momentum --model-ema --model-ema-decay 0.9966 --epochs 70 
--num-classes 10 --tta 1 --pretrained    

## **Inference**
실행은 efficientdet-pytorch 에서  
`
python validate.py [dataset 경로] --model tf_efficientdet_d4_ap --dataset coco --split test --num-gpu 1 -b 1 --checkpoint [checkpoint 경로] --num-classes 10 --results [결과 생성할 경로]  
`   
ex.  
python validate.py /opt/ml/detection/dataset --model tf_efficientdet_d4_ap --dataset coco 
--split test --num-gpu 1 -b 1 --checkpoint /opt/ml/detection/baseline/efficientdet-pytorch/output/train/20220331-075919-tf_efficientdet_d4_ap/model_best.pth.tar 
--num-classes 10 --results /opt/ml/detection/baseline/efficientdet-pytorch/result.json    

## **After treatment** 
결과파일인 json파일을 -> 대회 제출 형식에 맞는 내용과 형식(csv)으로 변경 필요 
* [Json to Csv](https://github.com/boostcampaitech3/level2-object-detection-level2-cv-05/blob/utils/utils/submit.py)      

실행은 efficientdet-pytorch 에서    
```bash  
python submit.py    
```   
실행 후 ) submission.csv 생성

## 추가 : inference 결과 시각화로 확인   
* [Inference Visualization](https://github.com/boostcampaitech3/level2-object-detection-level2-cv-05/blob/utils/utils/inference_viz.py)   
```bash
python data_viz.py -d [test데이터경로] -a [annotation파일(.json)경로] -p [포트번호]   
```   
ex.   
python data_viz.py -d /opt/ml/detection/dataset/test -a /opt/ml/detection/baseline/efficientdet-pytorch/csv_to_json.json -p 30004
