from pymongo import MongoClient
from bson.objectid import ObjectId #這東西再透過ObjectID去尋找的時候會用到
import api_handler as api

test = pd.DataFrame()
# connection
conn = MongoClient() # 如果你只想連本機端的server你可以忽略，遠端的url填入: mongodb://<user_name>:<user_password>@ds<xxxxxx>.mlab.com:<xxxxx>/<database_name>，請務必既的把腳括號的內容代換成自己的資料。
db = conn.delta_test
collection = db.delta_test

# test if connection success
print(collection.stats) # 如果沒有error，你就連線成功了。pytho
collection.delete_many({})
dic = {"equipID":6, "mold":680 , "name":"test","order":[36220011620]}
dic2 = {"equipID":11, "mold":600 , "name":"test2","order":[36220011620,36219038317]}
dic3 = {"equipID":6, "mold":700 , "name":"test3","order":[36219038317,362200012691]}
collection.insert_one(dic)
collection.insert_one(dic2)
collection.insert_one(dic3)
#cursor = collection.find({},{'equip': 6})

all_data = [row for row in collection.find({'equipID': 6},{'equipID','mold',"name","order"})]

for row in all_data:
    print(row)