import sys
import json
import os
import io
from datetime import datetime

class Wos_reply:
    def __init__ (self, Dic):
        self.wos_list = Dic['wos']
        self.text_list()

    def text_list(self) :
        wos_list = []
        for i in range(len(self.wos_list)):
            name = self.wos_list[i]['woName'] + ", " + self.wos_list[i]['itemName'] + ", " +self.wos_list[i]['eqpName']+ ", " +self.wos_list[i]['startTime']
            text = {"name" : name , "index" : i }
            wos_list.append(text)
        self.wos_list_html = wos_list

    def to_json(self , index_list) :
        selected = []
        item = self.wos_list[int(index_list[0])]['itemId']
        eqip = self.wos_list[int(index_list[0])]['eqpId']
        self.sameID_flag = True
        for i in index_list:
            name = self.wos_list[int(i)]['woId']
            selected.append (name)
            if item != self.wos_list[int(i)]['itemId'] or eqip != self.wos_list[int(i)]['eqpId'] :
                self.sameID_flag = False
        output = {"wos": selected}
        self.json_file = json.dumps(output)

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

        


if __name__ == "__main__":
    startTime = datetime.now()
    endTime = datetime.now()
    j = Wos_reply(startTime ,endTime)
    print(j.json_file)
