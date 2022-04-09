# yolodata
## folder structure
```
📂 yolodata/
│
├── 📂 images 
│	├── 📂 train
│	└── 📂 test
├── 📂 labels
│
├── 📑 train.txt 
├── 📑 val.txt 
└── 📑 test.txt 
```

## labels 폴더 생성 
- [convert2yolo](https://github.com/ssaru/convert2Yolo) 사용
```bash
git clone https://github.com/ssaru/convert2Yolo.git
cd convert2Yolo
```
- names.txt 파일 생성 
```bash
# convert2Yolo/names.txt

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
- convert2yolo 실행 예시
```bash
python3 example.py --datasets COCO \
--img_path /opt/ml/detection/dataset/train/ \
--label /opt/ml/detection/dataset/train.json \
--convert_output_path /opt/ml/detection/yolodata/labels/ \
--img_type ".jpg" \
--manipast_path ./ \
--cls_list_file names.txt
```
- labels 폴더
  - 생성한 labels 폴더를 yolodata 폴더로 이동

## txt 파일 생성

- coco2yolo 파일 생성 후 실행
```bash
python coco2yolo.py [train.json 파일 경로]
python coco2yolo.py [val.json 파일 경로]
python coco2yolo.py [test.json 파일 경로]
```
- [train.json 파일명].txt -> train.txt로 이름 변경 
- [val.json 파일명].txt -> val.txt로 이름 변경 

- txt 파일
  - 세 txt 파일 (train.txt, val.txt, test.txt) 모두 yolodata 폴더로 이동

## images 폴더 생성 
- image 폴더 
	- 1 ) (datasets 폴더에 있는) train, test 이미지 폴더를 복사해서
	- 2 ) yolodata 폴더의 images 폴더 내부에 붙여넣기
