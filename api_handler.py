import sys
import json
import os
import io
from datetime import datetime

class Wos_reply:
    def __init__ (self, startTime, endTime):
        self.startTime = str(startTime)
        self.endTime = str(endTime)
        self.date = [
             {
            "woId": "GUID",
            "woName": "”woName”",
            "itemId": "GUID",
            "itemName": "itemName",
            "eqpId": "GUID",
            "eqpName": "eqpName",
            "orderQty": 14000,
            "startTime": str(startTime), 
            "endTime": str(endTime)   }
        ]
        self.to_json()

    def to_json(self) :
        output = {"wos": self.date}
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
