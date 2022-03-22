import json
import os
import random


def split_dataset(input_json, output_dir, val_ratio, random_seed): # input_csv param X
    random.seed(random_seed)

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    # annotation json 정보 저장
    images = dataset['images']
    annotations = dataset['annotations']
    categories = dataset['categories']

    # file_name에 prefix 디렉토리까지 포함 (CocoDataset 클래스를 사용하는 경우)
    for image in images:
        image['file_name'] = '{}'.format(image['file_name'])

    image_ids = [x.get('id') for x in images]
    image_ids.sort()
    random.shuffle(image_ids)

    # val, train 개수 설정 
    num_val = int(len(image_ids) * val_ratio)
    num_train = len(image_ids) - num_val

    image_ids_val, image_ids_train = set(image_ids[:num_val]), set(image_ids[num_val:])

    train_images = [x for x in images if x.get('id') in image_ids_train]
    val_images = [x for x in images if x.get('id') in image_ids_val]
    train_annotations = [x for x in annotations if x.get('image_id') in image_ids_train]
    val_annotations = [x for x in annotations if x.get('image_id') in image_ids_val]

    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories,
    }

    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories,
    }

    output_seed_dir = os.path.join(output_dir, f'seed{random_seed}')
    os.makedirs(output_seed_dir, exist_ok=True)
    output_train_json = os.path.join(output_seed_dir, 'train.json')
    output_val_json = os.path.join(output_seed_dir, 'val.json')
    output_train_csv = os.path.join(output_seed_dir, 'train.csv')
    output_val_csv = os.path.join(output_seed_dir, 'val.csv')

    print(f'write {output_train_json}')
    with open(output_train_json, 'w') as train_writer:
        json.dump(train_data, train_writer)

    print(f'write {output_val_json}')
    with open(output_val_json, 'w') as val_writer:
        json.dump(val_data, val_writer)

    # print(f'write {output_train_csv}, {output_val_csv}')
    # with open(input_csv, 'r') as csv_reader, \
    #         open(output_train_csv, 'w') as train_writer, \
    #         open(output_val_csv, 'w') as val_writer:
    #     train_writer.write('ImageId,EncodedPixels,Height,Width,CategoryId\n')
    #     val_writer.write('ImageId,EncodedPixels,Height,Width,CategoryId\n')
    #     for line in csv_reader:
    #         if line.startswith('ImageId'): continue
    #         image_id, encoded_pixels, height, width, category_id = line.strip().split(',')
    #         image_id = int(image_id)
    #         if image_id in image_ids_train:
    #             train_writer.write(line)
    #         elif image_id in image_ids_val:
    #             val_writer.write(line)
    #         else:
    #             raise ValueError(f'unknown image_id: {image_id}')


split_dataset(input_json='/opt/ml/detection/dataset/train.json',
              # input_csv='../dataset/train.csv',
              output_dir='/opt/ml/detection/dataset/',
              val_ratio=0.2, # 8:2 분할 
              random_seed=2022)