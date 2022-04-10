import argparse
import json
import pandas as pd

test_json_path = '/opt/ml/detection/dataset/test.json'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str)
    parser.add_argument('--output-path', type=str, default='./pseudo.json')
    parser.add_argument('--conf-ths', type=float, default=0.3)
    arg = parser.parse_args()
    return arg


def main():
    args = get_args()
    submission = pd.read_csv(args.csv_path)
    _id_cnt = 0
    with open(test_json_path, 'r') as f:
        pseudo_data = json.load(f)

    for i, row in submission.iterrows():
        row = row['PredictionString'].split(" ")
        for j in range(0, len(row) - 1, 6):
            if float(row[j + 1]) > args.conf_ths:
                category_id = int(row[j])
                image_id = i
                bbox = list(map(float, row[j + 2:j + 6]))
                area = round(bbox[2]*bbox[3])
                if area > 100:
                    anno = {'image_id': image_id, "category_id": category_id,
                            "area": area, "iscrowd": 0, "id": _id_cnt, "bbox": bbox}
                    _id_cnt += 1
                    pseudo_data['annotations'].append(anno)

    with open(args.output_path, 'w') as f:
        json.dump(pseudo_data, f, indent=4)


if __name__ == '__main__':
    main()
