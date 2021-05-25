import pandas as pd
import pymongo

def save_df ( name , list_, tags) :
    output = pd.DataFrame()
    for feature in list_ :
        extract_data = []
        for i in range(len(tags)):
            extract_data.append(tags[i]['IMM'][int(feature)])
        output[feature] = extract_data
        output.to_csv(name ,index = False)
    

def save_defect(name , df) :
    defect =df.defect
    outcome = list(map(int, defect))
    defect.to_csv(name ,index = False)