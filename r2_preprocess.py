import pandas as pd
import numpy as np
import pickle
import os.path

def get_holidays(fpath):
    # holidays are from http://www.timeanddate.com/holidays/us/ , holidays and some observances
    
    f = open(fpath)
    lines = f.readlines()
    lines = [line.split(" ")[:3] for line in lines]
    lines = ["{} {} {}".format(line[0], line[1], line[2]) for line in lines]
    lines = pd.to_datetime(lines)
    return pd.DataFrame({"date2":lines})

def get_holiday_names(fpath):
    # holiday_names are holidays + around Black Fridays
    
    f = open(fpath)
    lines = f.readlines()
    lines = [line.strip().split(" ")[:4] for line in lines]
    lines_dt = ["{} {} {}".format(line[0], line[1], line[2]) for line in lines]
    lines_dt = pd.to_datetime(lines_dt)
    lines_hol = [line[3] for line in lines]
    return pd.DataFrame({"date2":lines_dt, "holiday_name":lines_hol})

def to_float(series, replace_value_for_M, replace_value_for_T):
    series = series.map(lambda s : s.strip())
    series[series == 'M'] = replace_value_for_M
    series[series == 'T'] = replace_value_for_T
    return series.astype(float)

def preprocess(_df, is_train):
    
    df = _df.copy()

    # log1p
    if is_train: 
        df['log1p'] = np.log(df['units'] + 1)

    # date
    df['date2'] = pd.to_datetime(df['date'])

    # weather features
    wtr['date2'] = pd.to_datetime(wtr.date)
    wtr["preciptotal2"] = to_float(wtr["preciptotal"], 0.00, 0.005)
    wtr["preciptotal_flag"] = np.where(wtr["preciptotal2"] > 0.2, 1.0, 0.0)

    wtr["depart2"] = to_float(wtr.depart, np.nan, 0.00)
    wtr["depart_flag"] = 0.0
    wtr["depart_flag"] = np.where(wtr["depart2"] < -8.0, -1, wtr["depart_flag"])
    wtr["depart_flag"] = np.where(wtr["depart2"] > 8.0 ,  1, wtr["depart_flag"])
    df = pd.merge(df, key, on='store_nbr')
    df = pd.merge(df, wtr[["date2", "station_nbr", "preciptotal_flag", "depart_flag"]], 
                      on=["date2", "station_nbr"])
    
    # weekday
    df['weekday'] = df.date2.dt.weekday
    df['is_weekend'] = df.date2.dt.weekday.isin([5,6])
    df['is_holiday'] = df.date2.isin(holidays.date2)
    df['is_holiday_weekday'] = df.is_holiday & (df.is_weekend == False)
    df['is_holiday_weekend'] = df.is_holiday &  df.is_weekend

    # bool to int (maybe no meaning)
    df.is_weekend = np.where(df.is_weekend, 1, 0)
    df.is_holiday = np.where(df.is_holiday, 1, 0)
    df.is_holiday_weekday = np.where(df.is_holiday_weekday, 1, 0)
    df.is_holiday_weekend = np.where(df.is_holiday_weekend, 1, 0)
    
    # day, month, year
    df['day'] = df.date2.dt.day
    df['month'] = df.date2.dt.month
    df['year'] = df.date2.dt.year
    
    # around BlackFriday
    df = pd.merge(df, holiday_names, on='date2', how = 'left')
    df.loc[df.holiday_name.isnull(), "holiday_name"] = ""

    around_BlackFriday = ["BlackFridayM3", "BlackFridayM2", "ThanksgivingDay", "BlackFriday",
                          "BlackFriday1", "BlackFriday2", "BlackFriday3"]
    df["around_BlackFriday"] = np.where(df.holiday_name.isin(around_BlackFriday), 
                                        df.holiday_name, "Else")

    return df

# read dataframes
key = pd.read_csv("data/key.csv")
wtr = pd.read_csv("data/weather.csv")
holidays = get_holidays("holidays.txt")
holiday_names = get_holiday_names("holiday_names.txt")

store_item_nbrs_path = 'model/store_item_nbrs.csv'
store_item_nbrs = pd.read_csv(store_item_nbrs_path)
valid_store_items = set(zip(store_item_nbrs.store_nbr, store_item_nbrs.item_nbr))

# preprocess 
df_train = pd.read_csv("data/train.csv")
mask_train = [(sno_ino in valid_store_items) for sno_ino in zip(df_train['store_nbr'], df_train['item_nbr']) ]
df_train = df_train[mask_train].copy()
preprocess(df_train, True).to_pickle('model/train2.pkl')

df_test =  pd.read_csv("data/test.csv")
mask_test = [(sno_ino in valid_store_items) for sno_ino in zip(df_test['store_nbr'], df_test['item_nbr']) ]
df_test =  df_test[mask_test].copy()
preprocess(df_test,  False).to_pickle('model/test2.pkl')