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
from api_handler import Wos_query , Wos_reply


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

KP = ['603','601','602','600','599','291','221','114','28','96','98','99','102','128','132','255','461','462','463','464','467','468','469','470','482','481','487','488','493','494','565','566','567','568','564']
defects = [0,5,6,7]
split_size = 0.4


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
    if request.method == 'POST':
        if input_check(request.form['startTime'], request.form['endTime']):
            wos_query = Wos_query(request.form['startTime'],request.form['endTime'])
            print (wos_query.json_file)
            flash('Updload Success!')
            with open('wos_api.json') as f:
                data = json.load(f)
            return redirect(url_for('work_order'))
        else :
            return 'bye'
    return render_template('date.html')
    
@app.route('/work_order', methods=['GET', 'POST'])
def work_order():
    wos = [
        {
            'name':'test1',
            'place': 'kaka',
        },
        {
            'name': 'test2',
            'place': 'Goa',
        }
    ]
    if request.method == 'POST':
        if input_check(request.form['startTime'], request.form['endTime']):
            wos_query = Wos_query(request.form['startTime'],request.form['endTime'])
            print (wos_query.json_file)
            flash('Updload Success!')
            
            #return render_template('work_order.html', wos = wos)
        else :
            return 'bye'
    return render_template('work_order.html', wos = wos)


@app.route('/hello/<username>')
def hello(username):
    return render_template('hello.html', username=username)

if __name__ == '__main__':
    app.debug = True
    app.secret_key = "Your Key"
    app.run()