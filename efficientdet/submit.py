import json
import pandas as pd

def main():
    id=-1
    Prediction_strings=[]
    file_names=[]

    json_root='/opt/ml/detection/baseline/efficientdet-pytorch/result.json'
    
    with open(json_root,'r') as f:
        results=json.load(f)
    
    Prediction_string = ''
    for i, result in enumerate(results): 
        xmin=str(result['bbox'][0])
        ymin=str(result['bbox'][1])
        xmax=str(result['bbox'][0]+result['bbox'][2])
        ymax=str(result['bbox'][1]+result['bbox'][3])

        if id < result['image_id']: # 같은 id의 첫번째 값 
            # image id 작성 및 추가  
            file_name = 'test/' + str(result['image_id']).zfill(4) + '.jpg'
            file_names.append(file_name)
            id = result['image_id']
             
            # Prediction_string 작성  
            Prediction_string = ''
            Prediction_string += str(result['category_id']-1) + ' ' + str(result['score']) + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' ' 

        elif result['image_id']!= 4870 : # 같은 id의 중간값 들 
            if result['image_id'] < results[i+1]['image_id']: # 같은 id의 마지막번째 값 
                Prediction_string += str(result['category_id']-1) + ' ' + str(result['score']) + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' ' 
                Prediction_strings.append(Prediction_string)
            else:
                Prediction_string += str(result['category_id']-1) + ' ' + str(result['score']) + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' ' 
        elif i==(len(results)-1): # 전체 id 중 맨 마지막 
            Prediction_string += str(result['category_id']-1) + ' ' + str(result['score']) + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' ' 
            Prediction_strings.append(Prediction_string)

    submission = pd.DataFrame()
    submission['PredictionString'] = Prediction_strings
    submission['image_id'] = file_names
    submission.to_csv('/opt/ml/detection/baseline/efficientdet-pytorch/submission.csv', index=None)

if __name__ == "__main__":
    main()