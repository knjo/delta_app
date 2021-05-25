import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.functional import split
from sklearn.preprocessing import StandardScaler
from collections import Counter



class DataPreprocess:
    def __init__(self,data_path,defect_path,select_columns,defects,split_size=0.4):
        self.data_path = data_path
        self.defect_path = defect_path
        self.select_columns = select_columns
        self.defects = defects
        self.split_size = split_size
        self.load()
        self.defect_relabel()
        self.data_split()
        self.ROS()

    def load(self):
        self.df = pd.read_csv(self.data_path)
        self.df_defect = pd.read_csv(self.defect_path)
        self.df_new = self.df
        self.df_new.reset_index(drop = True)
        self.df_defect.reset_index(drop= True)
        self.df_select = self.df_new[self.select_columns]
        print("Data size = {}".format(self.df_select.shape))
        print("====================")

    def ROS(self):
         ros = RandomOverSampler(random_state=41)
         self.X_train, self.y_train = ros.fit_resample(self.X_train, self.y_train)
         print("Data after resampling = {}".format(sorted(Counter(self.y_train).items())))
    
    def Standardization(self):
        self.sc = StandardScaler()
        self.sc.fit(self.X_train)
        self.X_train = self.sc.transform(self.X_train)




        
    def defect_relabel(self):
        for i in range(len(self.defects)):
            self.df_defect[self.df_defect.values == self.defects[i]] = i
        
        self.df_defect[self.df_defect.values > (len(self.defects)-1)] = 0
        print("Defect relabel Value counts = \n{}".format(self.df_defect['defect'].value_counts()))
        print("====================")


        
    def data_split(self):
        train_end_idx = len(self.df_defect)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.df_select.values[:train_end_idx,:],self.df_defect.values[:train_end_idx], test_size=0.4)
        print("X_train.shape = {}, y_train.shape = {}\nX_valid.shape = {}, y_valid.shape={}".format(self.X_train.shape, self.y_train.shape,self.X_valid.shape, self.y_valid.shape))
        print("====================")





if __name__ == "__main__":
    data_path = "E11M47_0405.csv"
    defect_path = "E11M47_defect_0405.csv"
    defects = [0,5,6,7]
    select_col = ['603','601','602','600','599','291','221','114','28','96','98','99','102','128','132','255','461','462','463','464','467','468','469','470','482','481','487','488','493','494','565','566','567','568','564']
    split_size = 0.4
    data = DataPreprocess(data_path,defect_path,select_col,defects,split_size)
