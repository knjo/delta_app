import sys
import json
import os
import io
from datetime import datetime
import pandas as pd
import torch
from torch.utils import data
from network import network, training
from datapreprocess import DataPreprocess

class Wos_reply:
    def __init__ (self, Dic):
        self.wos_list = Dic['wos']
        self.sort()
        self.text_list()

    def sort(self):
        self.wos_df = pd.DataFrame(self.wos_list)
        self.wos_df.sort_values(by=['eqpName','itemName'] , ascending=True, axis =0 , inplace=True)
        self.wos_list = self.wos_df.reset_index(drop=True)

    def text_list(self) :
        wos_list = []
        for i in range(len(self.wos_list)):
            time = "日期:" + str(self.wos_list['startTime'][i]).split('T')[0] + " " + str(self.wos_list['startTime'][i]).split('T')[1].split('.')[0] +" ~ " + str(self.wos_list['endTime'][i]).split('T')[0] + " " + str(self.wos_list['endTime'][i]).split('T')[1].split('.')[0]
            name = "機台:" + str(self.wos_list['eqpName'][i]) + "  工單:"+ str(self.wos_list['woName'][i])+ "  產品:"  +str(self.wos_list['itemName'][i]) +"  資料筆數:" + str(self.wos_list['orderQty'][i]) 
            text = {"name" : name , "index" : i , "time" : time}
            wos_list.append(text)
        self.wos_list_html = wos_list
        self.sampleName = str(self.wos_list['eqpName'][i]) + "_"+ str(self.wos_list['woName'][i])+ "_"  +str(self.wos_list['itemName'][i])+ str(datetime.today()).split(" ")[0]

    def to_json(self , index_list) :
        selected = []
        item = self.wos_list['itemId'][int(index_list[0])]
        eqip = self.wos_list['eqpId'][int(index_list[0])]
        self.sameID_flag = True
        for i in index_list:
            name = self.wos_list['woId'][int(i)]
            selected.append ({"wos": [name]})
            if item != self.wos_list['itemId'][int(i)] or eqip != self.wos_list['eqpId'][int(i)] :
                self.sameID_flag = False
        #output = {"wos": selected}
        self.json_file = json.dumps(selected)

class Wos_query:
    def __init__ (self, startTime, endTime):
        self.startTime = str(startTime) + " 00:00:00"
        self.endTime = str(endTime)+ " 00:00:00"
        self.date =  {
            "startTime": str(self.startTime), 
            "endTime": str(self.endTime) }
        self.to_json()

    def to_json(self) :
        self.json_file = json.dumps(self.date)

class Sample_reply:
    def __init__ (self, json , data_path ,name):
        self.wos_list = json
        print (self.wos_list)
        self.KPIV , self.KPIV_name = self.KP_extract(self.wos_list[0]['KPIV'] , "I")
        self.KPOV, self.KPOV_name = self.KP_extract(self.wos_list[0]['KPOV'] , "O")
        self.extract_data(data_path, name)

    def KP_extract(self , kp , io ):
        list_ = []
        list_name = []
        for i in range(len(kp)) :
            try : 
                IMM = str(kp[i]['tagPath']).split('[')[1].split(']')[0]
            except :
                IMM = io + "_"+ str(i)
            list_.append(IMM)
            list_name.append(kp[i]['name'])
        return list_, list_name

    def extract_data (self ,data_path ,name ):
        feature = []
        defect = []
        self.KP = self.KPIV
        self.KP.extend(self.KPOV)
        for i in range(len(self.wos_list)):
            for j in range(len(self.wos_list[i]['tagData'])):
                kp = self.wos_list[i]['tagData'][j]["KPIV"]
                kp.extend(self.wos_list[i]['tagData'][j]["KPOV"])
                feature.append(kp)
                isDefect = 0
                if str(self.wos_list[i]['tagData'][j]["qcResult"]["isDefect"]) != "False" :
                    isDefect = 1
                defect.append(isDefect)
        self.feature = pd.DataFrame(feature , columns=self.KP)
        self.Defect =  pd.DataFrame(defect , columns=["defect"])
        self.feature_path = data_path + name + ".csv"
        self.defect_path = data_path + name + "_defect.csv"
        self.feature.to_csv( self.feature_path, index=False )
        self.Defect.to_csv( self.defect_path, index=False )


if __name__ == "__main__":

    data_path = "data\\" 
    model_path = "model\\"
    with open('wos_api.json',encoding="utf-8") as f:
        wos = json.load(f)
    wos_reply = Wos_reply(wos)
    print (wos_reply.wos_df)
    
    """
    sample_reply = Sample_reply(wos , data_path , "test")
    split_size = 0.1

    data = DataPreprocess(sample_reply.feature_path, sample_reply.defect_path, sample_reply.KP, split_size)
    net = training(data.X_train,data.y_train,data.X_valid,data.y_valid , 3)
    Model_path = model_path +  "test"
    print (net)
    torch.save(net, Model_path)
    """