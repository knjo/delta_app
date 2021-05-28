from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm #pip install tqdm
import torch
from sklearn.metrics import classification_report,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import torch.utils.data as data
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold


class network(nn.Module):
    def __init__(self,input_layer,output_layer):
        super(network,self).__init__()
        self.L1 = nn.Linear(input_layer,input_layer*2)
        self.L2 = nn.Linear(input_layer*2,input_layer)
        self.L3 = nn.Linear(input_layer,10)
        self.output = nn.Linear(10,output_layer)
    def forward(self , x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = self.output(x)
        return x
    
    


def training(X_train,y_train,X_valid,y_valid,epoch):
    tensor_x = torch.from_numpy(X_train)
    tensor_y = torch.from_numpy(y_train)
    tensor_y = torch.squeeze(tensor_y)
    tensor_x_test = torch.from_numpy(X_valid)
    tensor_y_test = torch.from_numpy(y_valid)
    tensor_y_test = torch.squeeze(tensor_y_test)

    train_set = data.TensorDataset(tensor_x,tensor_y)
    test_set = data.TensorDataset(tensor_x_test,tensor_y_test)
    train_dataset = data.DataLoader(dataset =  train_set,batch_size=100,shuffle=True)
    test_dataset = data.DataLoader(dataset = test_set , batch_size=100)

    use_cuda = torch.cuda.is_available()
    input_l = X_train.shape[1]
    output_l = len(np.unique(np.concatenate((y_train,y_valid.T.reshape(-1)))))
    net = network(input_l,output_l)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters())
    loss_history = []

    if use_cuda:
        net = net.cuda()
    for e in tqdm(range(epoch),ncols=80):
        epoch_loss_sum = 0
        for x , y in tqdm(train_dataset,ncols=80):
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            batch_size = x.shape[0]
            x = x.view(batch_size,-1)
            net_out = net(x.float())
            loss = loss_fn(net_out , y)
            epoch_loss_sum += float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append(epoch_loss_sum)

        correct_count = 0
        total_testing = len(test_set)
    for x,y in test_dataset:
        if use_cuda:
            x = x.cuda()
            y = y.cuda().detach()
        batch_size = x.shape[0]
        x = x.view(batch_size,-1)
        output = net(x.float()).max(1)[1] #output出來是一個向量
        correct_count += torch.sum(output==y).item()
        #print(torch.sum(output==y))
        #print("output = {}".format(output))
        #print("y = {}".format(y))
        #print("======")
    #print(correct_count, total_testing)
    #correct_count = correct_count
    print('accuracy rate',correct_count/total_testing)
    
    y_pred = net(tensor_x_test.float()).max(1)[1]
    y_prediction = y_pred.detach().cpu().numpy() 
    y_prediction

    y_test2 = tensor_y_test.numpy()
    print("Confusion Matrix : ")
    print(confusion_matrix(y_test2,y_prediction))

    print(classification_report(y_test2, y_prediction))

    a = precision_recall_fscore_support(y_test2, y_prediction)
    labels = np.unique(np.concatenate((y_train,y_valid.T.reshape(-1))))
    precision = a[0]
    recall = a[1]
    fscore = a[2]
    supports = a[3]

    return net, precision,recall,fscore,supports,labels


def ten_fold(df_select,df_defect,epoch,k_folds):

    #configuration options

    k_folds = k_folds
    num_epochs = epoch
    results = {}

    kfold = KFold(n_splits=k_folds, shuffle = True)

    use_cuda = torch.cuda.is_available()
    loss_fn = nn.CrossEntropyLoss()
    loss_history = []

    
    for fold,(train_index, test_index) in enumerate(kfold.split(df_select.values,df_defect.values)):
   

        #print("fold = {}".format(fold))
        #rint("train_index = {}".format(train_index))
        #rint("test_index = {}".format(test_index))
        ### Dividing data into folds
        x_train_fold = df_select.values[train_index]
        x_test_fold = df_select.values[test_index]
        y_train_fold = df_defect.values[train_index]
        y_test_fold = df_defect.values[test_index]
        #print("123")
        #print(sorted(Counter(y_test_fold).items()))
        
        input_l = x_train_fold.shape[1]
        output_l = len(np.unique(np.concatenate((y_train_fold,y_test_fold))))
        net = network(input_l,output_l)
        optimizer = torch.optim.Adam(params=net.parameters())
        

        #ros = RandomOverSampler(random_state=42)
        #x_train_fold, y_train_fold = ros.fit_resample(x_train_fold,y_train_fold)
        #x_test_fold, y_test_fold = ros.fit_resample(x_train_fold,y_train_fold)
        #x_train_fold = StandardScaler().fit_transform(x_train_fold)
        #x_test_fold = StandardScaler().fit_transform(x_test_fold)

        """
        print("x_train_fold shape = {}".format(x_train_fold.shape))
        print("y_train_fold shape = {}".format(y_train_fold.shape))
        print("x_test_fold shape = {}".format(x_test_fold.shape))
        print("y_test_fold shape = {}".format(y_test_fold.shape))
        """
        tensor_x = torch.from_numpy(x_train_fold)
        tensor_y = torch.from_numpy(y_train_fold)
        tensor_y = torch.squeeze(tensor_y)
        tensor_x_test = torch.from_numpy(x_test_fold)
        tensor_y_test = torch.from_numpy(y_test_fold)
        tensor_y_test = torch.squeeze(tensor_y_test)

        train_set = data.TensorDataset(tensor_x,tensor_y)
        test_set = data.TensorDataset(tensor_x_test,tensor_y_test)
        
        train_dataset = data.DataLoader(dataset =  train_set,batch_size=100,shuffle=True)
        test_dataset = data.DataLoader(dataset = test_set , batch_size=100)

        for e in tqdm(range(num_epochs),ncols=80):
            epoch_loss_sum = 0
            for x , y in tqdm(train_dataset):
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                batch_size = x.shape[0]
                x = x.view(batch_size,-1)
                net_out = net(x.float())
                loss = loss_fn(net_out , y)
                epoch_loss_sum += float(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_history.append(epoch_loss_sum)

        correct_count = 0
        total_testing = len(test_set)
    
        print("len of test_set = {}".format(total_testing))
        
        for x,y in test_dataset:
            if use_cuda:
                x = x.cuda()
                y = y.cuda().detach()
            batch_size = x.shape[0]
            x = x.view(batch_size,-1)
            output = net(x.float()).max(1)[1] #output出來是一個向量
            correct_count += torch.sum(output==y).item()
            #print(torch.sum(output==y))
            #print("output = {}".format(output))
            #print("y = {}".format(y))
            #print("======")
        #print(correct_count, total_testing)
        #correct_count = correct_count
        print('accuracy rate',correct_count/total_testing)
        
        results[fold] = 100.0 * (correct_count/total_testing)

        y_pred = net(tensor_x_test.float()).max(1)[1]
        y_prediction = y_pred.detach().cpu().numpy() 
        y_test2 = tensor_y_test.numpy()


        print("Confusion Matrix : ")
        print(confusion_matrix(y_test2,y_prediction))
        print(classification_report(y_test2, y_prediction))
        a = precision_recall_fscore_support(y_test2, y_prediction)
        labels = np.unique(df_defect)
        precision = a[0]
        recall = a[1]
        fscore = a[2]
        supports = a[3]

    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

    return net, precision,recall,fscore,supports,labels


def closest_point(point,points):
    index = cdist([point],points).argmin()
    return points[index],index

def closest_points(point,points):
    index = cdist([point],points).argsort().reshape(-1)[:20]
    return points[index],index

    
def recommend_parameter(parameter,set,sc,kpiv,kpov):
    #print(parameter)
    #print(set)
    a , ind= closest_points(parameter[:kpov],set[:,:kpov])
    print(a)
    print(ind)
    rcmd , ind2= closest_point(parameter,set[ind])
    print(rcmd)
    print(ind2)
    rcmd_ind = ind[ind2]
    c = set[ind[ind2]]
    c = sc.inverse_transform(c).reshape(1,-1)
    d = sc.inverse_transform(parameter).reshape(1,-1)
    print("c = {}".format(c))
    print("d = {}".format(d))
    print("dist = {}".format(cdist(d,c)))
    original = sc.inverse_transform(rcmd)
    print(rcmd)
    print(rcmd_ind)
    print(original)

    return original