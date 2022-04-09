# YoloR
이 Repository는 YoloR을 기반으로 만들어졌습니다

📗 [YoloR](https://github.com/WongKinYiu/yolor)   

## **Setup for running**
### **가상환경 생성**
```bash
conda create -n yolor --clone detection
conda activate yolor
``` 

### **필수 Package 설치**
```bash
git clone https://github.com/WongKinYiu/yolor
```

## **Pre treatment**  
- coco.yaml 파일 수정
```bash
# /opt/ml/detection/baseline/yolor/data/coco.yaml 

# train and val datasets (image directory or *.txt file with image paths)
train: ../yolodata/train.txt  # 118k images
val: ../yolodata/val.txt  # 5k images
test: ../yolodata/test.txt  # 20k images for submission to https://competitions.codalab.org/competitions/20794

# number of classes
nc: 10

# class names
names: ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
```

- coco.names 파일 수정
```bash  
# /opt/ml/detection/baseline/yolor/data/coco.names 

General trash
Paper
Paper pack
Metal
Glass
Plastic
Styrofoam
Plastic bag
Battery
Clothing


```
- pretrained 모델 다운로드
```bash
# /opt/ml/detection/baseline/yolor/scripts/get_pretrain.sh 
# 주석 처리한 부분 (맨 앞에 # 표시)의 링크로 접속해서 pt 파일 다운
# pt 두 개의 파일(yolor_p6.pt, yolor_w6.pt)을 yolor 폴더에 옮겨 넣기  

# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76" -o yolor_p6.pt
rm ./cookie

# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1UflcHlN5ERPdhahMivQYCbWWw7d2wY7U" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1UflcHlN5ERPdhahMivQYCbWWw7d2wY7U" -o yolor_w6.pt
rm ./cookie
```

- train.py 파일 수정
```bash
# /opt/ml/detection/baseline/yolor/train.py  

# 63번째 줄 변경
plots = False # not opt.evolve
```

- test.py 파일 수정
```bash
# /opt/ml/detection/baseline/yolor/test.py

# 45번째 줄 변경
plots=False
```

## **Train** 
- /opt/ml/detection/baseline/yolor/train.sh 파일 생성 후 실행
```bash
sh train.sh
```
- train.sh 파일 예시
```bash
# /opt/ml/detection/baseline/yolor/train.sh

python3 train.py --batch-size 8 \
--img 1280 \
--data coco.yaml \
--cfg cfg/yolor_w6.cfg \ # yolor_p6.cfg
--weights ./yolor_w6.pt \ # yolor_p6.pt
--device 0 --name [folder name] \
--hyp hyp.finetune.1280.yaml \
--epochs 900
```

## **Inference**
- /opt/ml/detection/baseline/yolor/inference.sh 파일 생성 후 실행
```bash
sh inference.sh
```
- inference.sh 파일 예시
```bash
python3 test.py --batch-size 32 \
--img 1280 \
--data coco.yaml \
--conf 0.001 \
--iou 0.65 \
--device 0 \
--cfg cfg/yolor_w6.cfg \ # yolor_p6.cfg
--weights ./runs/train/[folder name]/weights/best_ap50.pt \
--name [folder name] \
--task test --verbose --save-conf --save-txt
```

## **After treatment** 
- submit.py 파일 수정
```bash
# submit.py

# 4번째 줄 변경
images = sorted(glob("/opt/ml/detection/baseline/yolor/runs/test/[folder name]/labels/*.txt")) # 파일 경로의 folder name을 변경
```


