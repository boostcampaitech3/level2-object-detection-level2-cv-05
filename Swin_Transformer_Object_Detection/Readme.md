# object-detection-level2-cv-05
# **Swin Transformer Object Detection**
이 Repository는 Swin Transformer Object Detection을 기반으로 만들어졌습니다

:orange_book: [MMdetection Repository](https://github.com/open-mmlab/mmdetection)  
:green_book: [Swin Transformer Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)  
:open_book: [Documentation](https://mmdetection.readthedocs.io/en/latest/)  

## **Setup for running**
* **본 Repository는 최신 버전이 아닌 mmdetection module을 사용하기 때문에 분리된 가상환경에서 작동시키는것을 권장합니다**

### **가상환경 생성**
```bash
conda create -n [env name] --clone detection
conda activate [env name]
```
* avtivate 에서 init하라는 에러가 발생할 경우
```bash
# root에서 시작
conda init --all
source ./.zshrc
conda activate [env name]
```

### **필수 Package 설치**
```bash
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git
cd Swin-Transformer-Object-Detection
pip install -r requirements/build.txt
pip install -v -e .
pip install mmcv==1.4.0
```

### **(Optional) For using EpochBasedRunnerAmp**
* Nvidia Apex 설치
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

### **Swin Transformer ImageNet Pretrained Model**
* 아래의 링크 참고
* [MicroSoft SwinTransformer](https://github.com/microsoft/Swin-Transformer)
```
wget [model url]
```

### Custom Config
* [Cascade Swin Model](./configs/p_stage/)
