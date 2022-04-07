import argparse
import json
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str)
    parser.add_argument('test_path', type=str)
    parser.add_argument('--output-path', type=str, default='./combine.json')
    arg = parser.parse_args()
    return arg


def main():
    args = get_args()
    with open(args.train_path, 'r') as f:
        train = json.load(f)

    with open(args.test_path, 'r') as f:
        test = json.load(f)

    img_index = max([i['id'] for i in train['images']]) + 1

    imgs = test['images']
    annos = test['annotations']

    for image in imgs:
        image['id'] += img_index
        del image['flickr_url']
        del image['coco_url']
        del image['date_captured']

    for anno in annos:
        anno['image_id'] += img_index

    train['images'].extend(imgs)
    train['annotations'].extend(annos)

    for i, anno in enumerate(train['annotations']):
        anno['id'] = i

    with open(args.output_path, 'w') as f:
        json.dump(train, f, indent=4)


if __name__ == '__main__':
    main()