# object-detection-level2-cv-05
# **Swin Transformer Object Detection**
이 Repository는 Swin Transformer Object Detection을 기반으로 만들어졌습니다

:orange_book: [MMdetection Repository](https://github.com/open-mmlab/mmdetection)  
:green_book: [Swin Transformer Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)  
:open_book: [Documentation](https://mmdetection.readthedocs.io/en/latest/)  

## **Setup for running**
* 본 Repository는 mmdetection과 다른 버전의 동일한 package를 사용하기 때문에 다른 환경에서 설치하는것을 권장합니다

### **필수 Package 설치**
```
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git
cd Swin-Transformer-Object-Detection
pip install -r requirements/build.txt
pip install -v -e .
```

### **(Optional) For using EpochBasedRunnerAmp**
* Nvidia Apex 설치
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### **Swin Transformer ImageNet Pretrained Model**
* 아래의 링크 참고
* [MicroSoft SwinTransformer](https://github.com/microsoft/Swin-Transformer)
```
wget [model url]
```

### Custom Config
* [Cascade Swin Model](./configs/p_stage/)