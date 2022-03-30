import json
import pandas as pd
import argparse


def main(args):
    with open(args.json_dir, 'r') as f:
        data = json.load(f)
    with open('/opt/ml/detection/dataset/train.json', 'r') as d:
        train = json.load(d)

    assert len(data['images']) == len(train['images']), '이미지 부족'

    data['info'] = train['info']
    data['licenses'] = train['licenses']
    data['categories'] = train['categories']
    images = data['images']
    annotations = data['annotations']

    images_df = pd.DataFrame.from_dict(images)
    images_df['id'] = images_df.index
    images_df = images_df.drop(columns=['dataset_id', 'category_ids', 'path', 'annotated', 'annotating',
                                        'num_annotations', 'metadata', 'deleted', 'milliseconds',
                                        'events', 'regenerate_thumbnail'])
    images_df['license'] = 0
    images_df['file_name'] = "train/" + images_df['file_name']

    data['images'] = images_df.to_dict(orient='records')

    annotations_df = pd.DataFrame.from_dict(annotations)
    annotations_df = annotations_df.sort_values('image_id', ignore_index=True).drop(columns=['isbbox', 'color', 'metadata', 'segmentation'])
    annotations_df['id'] = annotations_df.index
    annotations_df['image_id'] -= 1
    annotations_df['category_id'] -= 1

    data['annotations'] = annotations_df.to_dict(orient='records')

    with open(f'{args.new_json_dir}', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir', type=str)
    parser.add_argument('new_json_dir', type=str)
    arg = parser.parse_args()
    main(arg)
