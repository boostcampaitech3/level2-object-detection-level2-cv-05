import os
import numpy as np
import argparse

import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint

from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    arg = parser.parse_args()
    return arg


def make_submission(cfg, output):
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, 'submission.csv'), index=None)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False)
    checkpoint_path = args.checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    pkl_dir = os.path.join(cfg.work_dir, 'result.pkl')

    print(f'\nwriting results to {pkl_dir}')
    mmcv.dump(output, pkl_dir)

    make_submission(cfg, output)

if __name__ == "__main__":
    main()
