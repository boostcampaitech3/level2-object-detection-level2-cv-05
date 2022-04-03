import os
import json
import math
import random
import argparse
import shutil

from tqdm import tqdm
from glob import glob

import cv2
import numpy as np
from pycocotools.coco import COCO
import albumentations as A

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

SMALL = 32 ** 2 * 4
MIDDLE = 10000
absolute_data_dir = '/opt/ml/detection/dataset'
classes = {"General_trash": 0, "Paper": 1, "Paper_pack": 2, "Metal": 3, "Glass": 4,
           "Plastic": 5, "Styrofoam": 6, "Plastic_bag": 7, "Battery": 8, "Clothing": 9}
classes_invert = {0: "General_trash", 1: "Paper", 2: "Paper_pack", 3: "Metal", 4: "Glass",
                  5: "Plastic", 6: "Styrofoam", 7: "Plastic_bag", 8: "Battery", 9: "Clothing"}
index = 0
bg_path = os.path.join(absolute_data_dir, 'bg_patch')
g_path = None
transform = A.Compose([
    A.RandomRotate90(p=0.2)
])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Code for make a Patched Images')
    parser.add_argument('json', type=str, default=None)
    parser.add_argument('--middle-range', type=int, default=10000)
    parser.add_argument('--small-ratio', type=int, default=3)
    parser.add_argument('--middle-ratio', type=int, default=1)
    parser.add_argument('--eps', type=int, default=20)
    parser.add_argument('--name', type=str, default='g_image')
    args = parser.parse_args()
    global MIDDLE
    MIDDLE = args.middle_range
    return args


def make_dirs(arg) -> None:
    for i in classes_invert.values():
        path = os.path.join(absolute_data_dir, 'cropped_image', i)
        os.makedirs(path + '/middle', exist_ok=True)
        os.makedirs(path + '/small', exist_ok=True)
    global bg_path
    os.makedirs(os.path.join(bg_path, 'image'), exist_ok=True)
    os.makedirs(os.path.join(bg_path, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(absolute_data_dir, 'bg'), exist_ok=True)
    global g_path
    g_path = os.path.join(absolute_data_dir, arg.name)
    os.makedirs(g_path, exist_ok=True)


def check_size(area: int) -> str or None:
    if area > MIDDLE:
        return
    elif area > 32 ** 2 * 4:
        return 'middle'
    else:
        return 'small'


def crop_box(image: np.ndarray, annotations: list) -> None:
    img = image.copy()
    mask = np.ones([1024, 1024, 3]) * 255
    for ann in annotations:
        ann_id = ann['id']
        category = classes_invert[ann['category_id']]

        x, y, w, h = ann['bbox']
        x, y = map(int, [x, y])
        w, h = map(math.ceil, [w, h])

        cropped_img = image[y:y + h, x:x + w]
        img[y:y + h, x:x + w] = 255
        mask[y:y + h, x:x + w] = 0

        area = ann['area']
        size = check_size(area)
        if not size:
            continue

        img_dir = os.path.join(absolute_data_dir, 'cropped_image', category, size, str(ann_id) + '.jpg')
        cv2.imwrite(img_dir, cropped_img)
        global index
    crops = [img[0 + 512 * (i // 2):512 + 512 * (i // 2), 0 + 512 * (i % 2):512 + 512 * (i % 2)] for i in
             range(4)]
    cropped_masks = [mask[0 + 512 * (i // 2):512 + 512 * (i // 2), 0 + 512 * (i % 2):512 + 512 * (i % 2)] for i in
                     range(4)]
    for crop, crop_mask in zip(crops, cropped_masks):
        index += 1 if check_img(crop, crop_mask) else 0


def check_img(img, mask):
    global index
    if np.sum(np.where(img == [255, 255, 255], 1, 0)) < 262144:
        cv2.imwrite(os.path.join(bg_path, 'image', f'{index}.jpg'), img)
        cv2.imwrite(os.path.join(bg_path, 'mask', f'{index}.jpg'), mask)
        return True
    return False


def make_patch_image(root_dir: str):
    path = os.path.join(root_dir, 'bg_patch', 'image', '*.jpg')
    bg_list = glob(path)


def paste_bg_patch(img_dir, x, y, bg_img):
    img = cv2.imread(img_dir)
    mask = cv2.imread(img_dir.replace('image', 'mask'))
    output = transform(image=img, mask=mask)
    img, mask = output['image'], output['mask']
    roi = bg_img[y:y + 512, x:x + 512]
    cv2.copyTo(img, mask, roi)
    return bg_img


def create_rand_len(eps):
    wh = [[], []]
    for i in range(2):
        for _ in range(3):
            wh[i].append(0)
            f = random.randint(100 + eps, 512)
            s = random.randint(512 - f + eps, 512)
            wh[i].append(f)
            wh[i].append(f + s)
    ep_w = random.randint(0, eps)
    ep_h = random.randint(0, eps)
    return [wh[0][0], wh[0][3], wh[0][6], wh[0][1], wh[0][4], wh[0][7], wh[0][2], wh[0][5], wh[0][8]], wh[1], ep_w, ep_h


def make_bg(arg):
    image_list = glob(os.path.join(absolute_data_dir, 'bg_patch/image/*.jpg'))
    random.shuffle(image_list)
    for i in tqdm(range(len(image_list) // 9), total=len(image_list) // 9):
        files = []
        for _ in range(9):
            files.append(image_list.pop())
        bg_img = cv2.imread(os.path.join(absolute_data_dir, 'null.jpg'))
        w, h, ep_w, ep_h = create_rand_len(arg.eps)
        anchors = [[i, j] for (i, j) in zip(w, h)]
        random.shuffle(anchors)
        for file, anchor in zip(files[:9], anchors):
            paste_bg_patch(file, anchor[0], anchor[1], bg_img)
        img = bg_img[0 + ep_w:1024 + ep_w, 0 + ep_h:1024 + ep_h]
        cv2.imwrite(os.path.join(absolute_data_dir, 'bg', str(i) + '.jpg'), img)


def generate_image(args):
    global index

    bg_list = glob(os.path.join(absolute_data_dir, 'bg/*.jpg'))
    small_list = glob(os.path.join(absolute_data_dir, 'cropped_image/*/small/*.jpg'))
    middle_list = glob(os.path.join(absolute_data_dir, 'cropped_image/*/middle/*.jpg'))

    patch_list = small_list * args.small_ratio + middle_list * args.middle_ratio
    random.shuffle(patch_list)
    eps = 1e-5
    index = 4883
    anno_index = 23151
    images_info = []
    annos_info = []
    for i, bg in enumerate(tqdm(bg_list, total=len(bg_list))):
        bg = cv2.imread(bg)
        bboxs = []
        images = []
        for _ in range(min(random.randint(6, 20), len(patch_list))):
            images.append(patch_list.pop())
        annos = []
        for image in images:
            img = cv2.imread(image)
            h, w, _ = img.shape
            for _ in range(50):
                flag = False
                x, y = random.randint(0, 1024 - w), random.randint(0, 1024 - h)
                bbox = [x, y, x + w, y + w]
                if bboxs:
                    ious = bbox_overlaps(np.array([bbox]), np.array(bboxs))
                    if np.sum(ious) > eps:
                        flag = True
                if not flag:
                    bg[y:y + h, x:x + w] = img
                    bboxs.append(bbox)
                    anno = {'id': anno_index, 'image_id': index + i, 'category_id': classes[image.split('/')[6]],
                            'area': w * h, 'bbox': [x, y, w, h], 'iscrowd': False}
                    annos.append(anno)
                    anno_index += 1
                    break
            else:
                patch_list.append(image)

        image_info = {'id': index + i, 'width': 1024, 'height': 1024, 'file_name': f'{args.name}/{index + i}.jpg',
                      'license': 0}
        images_info.append(image_info)
        annos_info.extend(annos)
        cv2.imwrite(os.path.join(g_path, str(index + i) + '.jpg'), bg)
        if not patch_list:
            break
    return images_info, annos_info


def main():
    args = get_args()
    with open(args.json) as f:
        json_data = json.load(f)
    make_dirs(args)
    coco = COCO(args.json)

    print('Start Crop Bounding Box & Make Background Patch')
    for img_id in tqdm(coco.imgs, total=len(coco.imgs)):
        ann_ids = coco.getAnnIds(img_id)
        bboxs = [coco.loadAnns(i)[0] for i in ann_ids]
        img = coco.loadImgs(img_id)[0]['file_name']
        img_dir = os.path.join(absolute_data_dir, img)
        image = cv2.imread(img_dir)
        crop_box(image, bboxs)
    print('End Crop & Get Background Patch')

    print('Start Make Background image')
    if not os.path.exists(os.path.join(absolute_data_dir, 'null.jpg')):
        bg_img = np.zeros([1534, 1536, 3])
        cv2.imwrite('./null.jpg', bg_img)
    make_bg(args)
    print('End Make Background image')

    print('Start generate image')
    images_info, annos_info = generate_image(args)

    annos = json_data['annotations']
    images = json_data['images']

    images.extend(images_info)
    annos.extend(annos_info)

    with open('./new_data.json', 'w') as w:
        json.dump(json_data, w)

    shutil.rmtree(os.path.join(absolute_data_dir, 'cropped_image'))
    shutil.rmtree(os.path.join(absolute_data_dir, 'bg_patch'))
    shutil.rmtree(os.path.join(absolute_data_dir, 'bg'))
    print('Done')

if __name__ == '__main__':
    main()
