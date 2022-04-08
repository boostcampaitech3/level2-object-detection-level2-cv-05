# YoloV5
이 Repository는 YoloV5을 기반으로 만들어졌습니다

📗 [YoloV5](https://github.com/ultralytics/yolov5)   

## **Setup for running**
### **가상환경 생성**
```bash
conda create -n yolov5 python=3.8
conda activate yolov5
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
``` 

### **필수 Package 설치**
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# optional 
pip install wandb
```

## **Pre treatment**  
- /opt/ml/yolov5/data/trash.yaml 데이터 yaml 파일 생성 
- /opt/ml/yolov5/data/scripts/download_weights.sh 파일 실행 (pretrained model 다운로드)
```bash
sh /opt/ml/yolov5/data/scripts/download_weights.sh
```

## **Train** 
- /opt/ml/yolov5/train.sh 파일 생성 후 실행
```bash
sh train.sh
```

- train.sh 파일 예시
```
python train.py \
--img 1024 \
--batch 6 \
--epochs 120 \
--data data/trash.yaml \
--hyp data/hyps/hyp.scratch-high.yaml \ # hyps 폴더 내에서 사용할 hyperparameter yaml 파일 지정 # augmentation 정도에 따라 high/medium/low 
--optimizer SGD \
--multi-scale \ # multi-scale 사용 유무 지정
--cfg models/hub/yolov5x6.yaml \ # models 폴더 내에서 사용할 yolo 모델 yaml 파일 지정
--weights yolov5x6.pt \ # pretrained model 사용 시 지정 
--project [wandb project name] \
--name [wandb runs name] \
--entity [wandb id]
```

## **Inference**
> 방법1
- /opt/ml/yolov5/inference.py 파일 생성 후 실행
```bash
python inference.py
```

> 방법2
- /opt/ml/yolov5/inference.sh 파일 생성 후 실행
```bash
sh inference.sh
```
- inference.sh 파일 예시
```bash
python3 detect.py \
--img 640 \
--source /opt/ml/detection/baseline/yolodata/images/test \
--conf-thres 0.001 \
--iou-thres 0.65 \
--device 0 \
--weights /opt/ml/yolov5/runs/train/exp2/weights/best.pt \
--name [folder name] \
--save-conf --save-txt
```
- 실행 결과로 생성된 txt 파일을 csv로 변환해주어야 함
