import pandas as pd
import numpy as np
from math import log
import warnings
import os
import matplotlib.pyplot as plt
from tqdm import trange
import datetime, time
warnings.filterwarnings('ignore')

def get_all_gx_files():
    files_0 = os.listdir('./data/money_laundrying_dataset/0/')
    files_1 = os.listdir('./data/money_laundrying_dataset/1/')
    files_0 = ['./data/money_laundrying_dataset/0/'+file for file in files_0]
    files_1 = ['./data/money_laundrying_dataset/1/'+file for file in files_1]
    all_files = files_0 + files_1
    
    return all_files

def read_gx_data(filename):
    data = pd.read_excel(filename)
    data.drop('户籍地', axis = 1, inplace=True)
    data.drop('性别', axis = 1, inplace=True)
    data.drop('年龄', axis = 1, inplace=True)
    data.drop('交易网点名称', axis = 1, inplace=True)
    data = data.rename(columns={'交易卡号':'nameOrig',
                            '交易对手账卡号':'nameDest',
                            '交易时间':'date',
                            '交易金额':'amount',
                            '交易余额':'balance',
                            '收付标志':'type'})
#     data.date = [val[:10] for val in data.date.tolist()] #save Y-m-d only
#     data = data.drop_duplicates(['date']) #duplicated
    data.type[data.type=='进'] = 'in'
    data.type[data.type=='出'] = 'out'
    #data = data[~np.isnan(data['nameDest'])]
    data['amount'][data['type']=='out'] = data['amount'][data['type']=='out']*-1
    data.reindex()
    return data

def get_all_files_ts():
    dict_ts = dict()
    all_files = get_all_gx_files()
    for i in trange(len(all_files)):
    #for i in trange(10):
        file = all_files[i]
        df = read_gx_data(file)
        #dropna
        df.dropna(subset=['amount', 'balance', 'nameDest'],inplace=True)
        df.reset_index(inplace=True)
        df.drop(columns='index',inplace=True)
        
        if df.shape[0] == 0:
            continue
            
        account, amounts, balances, dests = get_ts(df)
        dict_ts[account] = dict()
        dict_ts[account]['amounts'] = amounts
        dict_ts[account]['balances'] = balances
        dict_ts[account]['dests'] = dests
    #add label
    files_0 = os.listdir('./data/money_laundrying_dataset/0/')
    files_1 = os.listdir('./data/money_laundrying_dataset/1/')
    for key in dict_ts.keys():
        filename = str(key)+'.xlsx'
        if filename in files_0:
            dict_ts[key]['label'] = 0
        else:
            dict_ts[key]['label'] = 1

    return dict_ts

def get_all_files_ts():
    dict_ts = dict()
    all_files = get_all_files()
    for i in trange(len(all_files)):
    #for i in trange(10):
        file = all_files[i]
        df = read_gx_data(file)
        #dropna
        df.dropna(subset=['amount', 'balance'],inplace=True)
        df.reset_index(inplace=True)
        df.drop(columns='index',inplace=True)
        
        if df.shape[0] == 0:
            continue
            
        account, amounts, balances = get_ts(df)
        dict_ts[account] = dict()
        dict_ts[account]['amounts'] = amounts
        dict_ts[account]['balances'] = balances
    #add label
    files_0 = os.listdir('./data/money_laundrying_dataset/0/')
    files_1 = os.listdir('./data/money_laundrying_dataset/1/')
    for key in dict_ts.keys():
        filename = str(key)+'.xlsx'
        if filename in files_0:
            dict_ts[key]['label'] = 0
        else:
            dict_ts[key]['label'] = 1

    return dict_ts

def load_ts():
    d = np.load('./data/money_laundrying_dataset/TS_amt_bal_dest.npy',allow_pickle=True)
    dict_ts = d[()]
    return dict_ts
def load_train_test():
    d = np.load('./data/money_laundrying_dataset/dataset_train_test.npy', allow_pickle=True)
    d = d[()]
    return d

def load_shp_loc_ts():
    shp_loc_ts = np.load('./data/money_laundrying_dataset/shp_20_200/shp_loc_ts.npy', allow_pickle=True)
    shp_loc_ts = shp_loc_ts[()]
    #shp_loc_ts.keys()
    return shp_loc_ts

def str2datetime(string, form='%Y-%m-%d'):
    year, month, day = time.strptime(string, form)[:3]
    return datetime.date(year, month, day)

#计算从开始日期的往后几天的日期
def add_delta_time(begin_date, delta):
    return begin_date + datetime.timedelta(days=delta)

#为出现的转账对象编号
def get_dest_code(df):
    dests = df['nameDest'].tolist()
    #print(len(dests), len(set(dests)))
    code = 0
    d_dest = dict()
    for dest in (set(dests)):
        d_dest[dest] = code
        code+=1
    return d_dest

#获得某个账户的转账时间序列、余额时间序列、转账对象时间序列
def get_ts(df):
    begin_date = datetime.date(2010,5,1)
    end_date = datetime.date(2020,4,1)
    amounts, balances, dests = list(), list(), list()  #转账金额  余额  转账对象
    date = begin_date
    delta = datetime.timedelta(days=1)
    i = 0
    account = df.nameOrig.iloc[0]
    #dests = df['nameDest'].tolist()
    d_dest = get_dest_code(df)
    while date <= end_date:
        #一天内有多个转账记录
        if  i < df.shape[0] and i != 0 and str2datetime(df.date.iloc[i][:10]) == str2datetime(df.date.iloc[i-1][:10]):
            amounts[len(amounts)-1] += df.amount.iloc[i]
            balances[len(balances)-1] = df.balance.iloc[i]
            date -= delta
            i+=1
        elif  i < df.shape[0] and str2datetime(df.date.iloc[i][:10]) == date:
            #print(df.amount.iloc[i])
            amounts.append(df.amount.iloc[i])
            balances.append(df.balance.iloc[i])
            #print(account)
            dests.append(d_dest[df.nameDest.iloc[i]])
#             print(df.amount.iloc[i])
            i+=1    
        else:
#             print('i=',i)
            amounts.append(0)
            balances.append(balances[len(balances)-1] if len(balances)!=0 else 0)
            dests.append(-1)
        date += delta
    return account, amounts, balances, dests

def asymmetricKL(P,Q):
    return np.sum(P * np.log(P / Q)) #calculate the kl divergence between P and Q
 
def symmetricalKL(P,Q):
    return (asymmetricKL(P,Q)+asymmetricKL(Q,P))/2.00

def JS_divergency(P, Q):
    M = (P + Q)/2
    js = 0.5*np.sum(P*np.log(P/M))+0.5*np.sum(Q*np.log(Q/M))
    return js

# def precision_score(y_train, pred_labels):
#     pred_loc = np.where(pred_labels == 1)[0].tolist()
#     true_loc = np.where(y_train == 1)[0].tolist()
#     cnt = 0
#     for i in pred_loc:
#         if i in true_loc:
#             cnt += 1
#     # print(cnt/len(true_loc))
#     return cnt/len(true_loc)

def dtw_distance(ts_a, ts_b, d=lambda x,y: abs(x-y), mww=10000):
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])
#         cost[0, i] = cost[0, i-1] + d(ts_a[0], ts_b[i])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window 
    return cost[-1, -1]