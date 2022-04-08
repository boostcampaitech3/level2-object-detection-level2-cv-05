# Test Time Augmentation




## 사용법


- yolov5/models/yolo.py 파일 내의,  
  forward_augment method 내부를 수정하여 Customize



- 수정 예시

```python
def _forward_augment(self, x):
	img_size = x.shape[-2:]  # height, width
	# s = [1, 0.83, 0.67]  #  기존 scale
	# f = [None, 3, None]  #  기존 flips (2-ud, 3-lr)

	s = [1.33, 1.17, 1, 0.83, 0.67, 1.33, 1.17, 1, 0.83, 0.67] ### TTA 시, Multi-scale 수정 적용
	f = [None, None, None, None, None, 3, 3, 3, 3, 3] ### TTA 시, flip 수정 적용

	y = []  # outputs
	for si, fi in zip(s, f):
		xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
		yi = self._forward_once(xi)[0]  # forward
		# cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
		yi = self._descale_pred(yi, fi, si, img_size)
		y.append(yi)
	y = self._clip_augmented(y)  # clip augmented tails
	return torch.cat(y, 1), None  # augmented inference, train
```





## Reference

- yolov5 Repository Test-Time Augmentation (TTA) Tutorial #303
- https://github.com/ultralytics/yolov5/issues/303
