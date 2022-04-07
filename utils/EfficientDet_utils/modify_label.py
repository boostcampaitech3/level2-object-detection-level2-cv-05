# efficientdet categories 1~10 
import json
from signal import alarm

def modify_anno_category(json_file): # choose fold
    # json 파일 읽어오기 
    with open(json_file,'r') as f:
        json_data = json.load(f)
    
    # categories 수정 
    categoriesArray=json_data.get("categories")
    for idx, list in enumerate(categoriesArray):
        list["id"]=list.get("id")+1
        json_data["categories"][idx]=list
        
    # annotations 수정 
    annoCategoriesArray=json_data.get("annotations")
    for idx, list in enumerate(annoCategoriesArray):
        list["category_id"]=list.get("category_id")+1
        json_data["annotations"][idx]=list
    
    # json 파일로 저장 
    if 'train' in json_file:
        with open('/opt/ml/detection/dataset/eff_train.json','w') as makefile:
            json.dump(json_data, makefile, indent="\t") 
    elif 'val' in json_file:
        with open('/opt/ml/detection/dataset/eff_val.json','w') as makefile:
            json.dump(json_data, makefile, indent="\t")
     
        
def modify_category(val_json):
    # json 파일 읽어오기 
    with open(val_json,'r') as f:
        json_data = json.load(f)
    
    # categories 수정 
    categoriesArray=json_data.get("categories")
    for idx, list in enumerate(categoriesArray):
        list["id"]=list.get("id")+1
        json_data["categories"][idx]=list
    
    # json 파일로 저장 
    with open('/opt/ml/detection/dataset/eff_test.json','w') as makefile:
        json.dump(json_data, makefile, indent="\t")  
    
def modify_test(test_json):
    train_json='/opt/ml/detection/dataset/StratifiedKFold/cv_train_3.json'
    val_json='/opt/ml/detection/dataset/StratifiedKFold/cv_val_3.json'
    test_json='/opt/ml/detection/dataset/test.json'
    
    modify_anno_category(train_json) # 기존 label & annotation의 category +1 한 eff_train.json 파일 생성 
    modify_anno_category(val_json) # 기존 label & annotation의 category +1 한 eff_val.json 파일 생성 
    modify_category(test_json) # 기존 label의 category +1 한 eff_test.json 파일 생성 