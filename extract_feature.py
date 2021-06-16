import pandas as pd
import numpy as np
import os
from datetime import datetime


# 计算两个时间的差值，time_after-time_before，单位s，如果两个时间不是同一天，返回-1

def time_diff(time_before, time_after):
    try:
        time_before = datetime.strptime(time_before, "%Y-%m-%d %H:%M:%S")
        time_after = datetime.strptime(time_after, "%Y-%m-%d %H:%M:%S")
        if time_before.year == time_after.year and \
                time_before.month == time_after.month and \
                time_before.day == time_after.day:
            seconds = int((time_after - time_before).total_seconds())
            return seconds
        else:
            return -1
    except Exception:
        return -1


def in_out_flag(_x):
    if _x == "借":
        return "进"
    if _x == "贷":
        return "出"
    return _x


'''
提取账户的特征，包括
1.总交易金额
2.月均交易金额：总交易金额/时间跨度
3.交易金额离散系数
4.月交易金额离散系数
5.出账/入账笔数
6.出账/入账频率
7.交易笔数：总的交易笔数
8.交易对手个数：总的交易对手个数
9.出账/入账金额
10.大额交易笔数
11.出入金额差：出账金额与入账金额的差值
12.交易金额一千整：交易金额为一千整的交易笔数
13.交易金额一万整
14.交易时间间隔均值：同一天的交易中，两笔交易之间的时间间隔均值
'''

# 设置交易金额大于5W元的为大额交易
BIG_VALUE = 50000


def extract_feature(filename_xlsx):
    file = filename_xlsx
    account = file.split(".")[0]
    df = pd.read_excel(file, converters={"交易卡号": str, "交易对手账卡号": str})
    df.sort_values(by=["交易时间"], inplace=True)
    df["year"] = pd.to_datetime(df["交易时间"].values).year
    df["month"] = pd.to_datetime(df["交易时间"].values).month
    df["day"] = pd.to_datetime(df["交易时间"].values).day

    # 有些数据收付标识是"借 贷"，有些是"进 出"，需要统一
    df["收付标志"] = df.apply(lambda x: in_out_flag(x["收付标志"]), axis=1)
    df["交易金额"] = df.apply(lambda x: -x["交易金额"] if x["交易金额"] < 0 else x["交易金额"], axis=1)

    feature = dict()
    feature["卡号"] = account
    # 总交易金额
    feature["总交易金额"] = df["交易金额"].sum().round(decimals=5)
    transaction_value_mean = df["交易金额"].mean()
    transaction_value_std = df["交易金额"].std()
    # 时间跨度
    months = (df.iloc[df.shape[0] - 1, :]["year"] - df.iloc[0, :]["year"]) * 12 + (
            df.iloc[df.shape[0] - 1, :]["month"] - df.iloc[0, :]["month"]) + 1
    feature["月均交易金额"] = feature["总交易金额"] / months
    month_trans_value = list(df.groupby(by=["month"])["交易金额"].sum())
    # 没有交易的月份视为交易金额为0
    month_trans_value += [0] * (months - len(month_trans_value))
    # 交易金额离散系数 = 交易金额标准差/交易金额均值
    feature["交易金额离散系数"] = transaction_value_std / transaction_value_mean
    feature["月交易金额离散系数"] = np.std(month_trans_value, ddof=1) / feature["月均交易金额"]
    # 如果只有一笔交易，离散系数为0
    if df.shape[0] <= 1:
        feature["交易金额离散系数"] = feature["月交易金额离散系数"] = 0.0
    feature["出账次数"] = df[df["收付标志"] == "出"].shape[0]
    feature["入账次数"] = df[df["收付标志"] == "进"].shape[0]
    feature["出账频率"] = feature["出账次数"] / len(df)
    feature["入账频率"] = feature["入账次数"] / len(df)
    feature["交易笔数"] = df.shape[0]
    feature["交易对手个数"] = len(set(df["交易对手账卡号"].values))
    feature["入账金额"] = df[df["收付标志"] == "进"]["交易金额"].sum()
    feature["出账金额"] = df[df["收付标志"] == "出"]["交易金额"].sum()
    feature["大额交易笔数"] = df[df["交易金额"] >= BIG_VALUE].shape[0]
    feature["出入金额差"] = abs(feature["出账金额"] - feature["入账金额"])
    # 交易金额为一千的整数倍的交易笔数
    feature["交易金额一千整"] = df["交易金额"].map(lambda x: True if x % 1000 == 0 else False).sum()
    feature["交易金额一万整"] = df["交易金额"].map(lambda x: True if x % 10000 == 0 else False).sum()

    # 同一天内的交易时间间隔的均值
    df["交易时间_shift"] = df["交易时间"].shift(-1)
    df["交易时间间隔"] = df.apply(lambda x: time_diff(x["交易时间"], x["交易时间_shift"]), axis=1)
    feature["交易时间间隔均值"] = df[df["交易时间间隔"] >= 0]["交易时间间隔"].mean()
    if not feature["交易时间间隔均值"] >= 0:
        feature["交易时间间隔均值"] = 0.0
    return feature


'''
example:
feature = extract_feature("14369101040007057.xlsx")
print(feature)
'''
