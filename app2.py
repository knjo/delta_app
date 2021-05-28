from flask import Flask, request, render_template, redirect, url_for, flash
import sys
import pandas as pd
import json
import handler
import os
import io
from tqdm import tqdm as tqdm #pip install tqdm
#import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from network import network, training
from datapreprocess import DataPreprocess
from api_handler import Sample_reply, Wos_query , Wos_reply
import requests
import time


app = Flask(__name__)

# Set KPIV&KPOV
"""
KPIV = pd.read_csv(r'ini_/KPIV.csv',sep=',')
KPOV = pd.read_csv(r'ini_/KPOV.csv',sep=',')
KP = KPIV.append(KPOV)
KP = KP.reset_index(drop = True)
"""
data_path = "data\\"
model_path = "model\\"


defects = [0,1]
split_size = 0.4
# global variable
wos = None
wos_reply = None
sample_reply = None
load_flag = True
headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
wos_url = "http://e8ef412f72e5.ngrok.io/api/v1/ai/getWOs"
sample_url = 'http://e8ef412f72e5.ngrok.io/api/v1/ai/getSamples'
# funtion 
def input_check(data1, data2):
    if data1 != '' and data2 != '':
        return True
    else:
        return False


# about route
@app.route('/BuildModel', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if input_check(request.form['equip'], request.form['mold']):
            name = request.form['equip'] + request.form['mold']+"_"+ request.form['name']
            data = pd.read_json(request.files['data'])
            Data_path = data_path + name +".csv"
            Defect_path = data_path + name + "_defect.csv"
            
            handler.save_df(Data_path, KP , data.tags)
            handler.save_defect(Defect_path, data)
            ##training
            Model_path = model_path + name
            data = DataPreprocess(Data_path,Defect_path,KP,defects,split_size)
            net = training(data.X_train,data.y_train,data.X_valid,data.y_valid)
            torch.save(net.state_dict(), Model_path)
            flash('Updload Success!')
            flash( name + ' Training Success!')
            return render_template('login.html')
        else :
            return 'bye'
    return render_template('login.html')

@app.route('/date', methods=['GET', 'POST'])
def date():
    global wos ,headers ,wos_reply
    if request.method == 'POST':
        if input_check(request.form['startTime'], request.form['endTime']):
            wos_query = Wos_query(request.form['startTime'],request.form['endTime'])
            print (wos_query.json_file)
            
            r = requests.post( wos_url, headers = headers , data= wos_query.json_file ,timeout=3)
            
            print(r)
            flash('Updload Success!')
            wos = r.json()
            wos_reply = Wos_reply(wos)
            return redirect(url_for('work_order'))
        else :
            return 'bye'
    return render_template('date.html')
    
@app.route('/work_order', methods=['GET', 'POST'])
def work_order():
    global wos, load_flag , wos_reply ,headers ,sample_reply ,model_path

    wos  =  wos_reply.wos_list_html

    if request.method == 'POST':
        if len(request.form.getlist('work_order')) > 0 :
            wos_reply.to_json(request.form.getlist('work_order'))
            if wos_reply.sameID_flag == False :
                flash('Plese work orders in same itemID and eqipID !')
            else :
                flash('Successed in selecting work order !')
                r = requests.post( sample_url , headers = headers , data= wos_reply.json_file ,timeout=300)            
                print(r)
                with open('sample2.json', 'w') as outfile:
                    json.dump(r.json(), outfile)
                
                sample_reply = Sample_reply(r.json() , data_path , wos_reply.sampleName )
                data = DataPreprocess(sample_reply.feature_path, sample_reply.defect_path, sample_reply.KP, defects, split_size)
                net = training(data.X_train,data.y_train,data.X_valid,data.y_valid)
                Model_path = model_path +  wos_reply.sampleName 
                torch.save(net.state_dict(), Model_path)
                flash('Traning Completed!')
            print (wos_reply.json_file)

        else :
            flash('Plese select at least one work order!')
    return render_template('work_order.html', wos = wos)



if __name__ == '__main__':
    app.debug = True
    app.secret_key = "Your Key"
    app.run()