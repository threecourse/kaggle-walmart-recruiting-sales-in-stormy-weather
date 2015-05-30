import pandas as pd
import numpy as np
import pickle

def create_rollingmean(_df):

    dfs = []

    for sno, ino in store_items:

        exclude_date = pd.to_datetime("2013-12-25")

        df = _df[(_df.store_nbr == sno) & (_df.item_nbr == ino)].copy()
        df = df.set_index('date2', drop=False)
        df = df.sort_index()
        df = df[df.date2 != exclude_date] # exclude 2013-12-25

        # calculate rolling mean
        window = 21
        df['rmean'] = pd.rolling_mean(df.log1p, window, center=True)
        df['rmean'] = df['rmean'].interpolate()
        df['rmean'] = df['rmean'].ffill()
        df['rmean'] = df['rmean'].bfill()

        # alldates
        alldates = pd.date_range('2012-01-01', '2014-10-31', freq='D')
        alldates.name = 'date2'
        df2 = pd.DataFrame(None, index = alldates)

        df2['store_nbr'] = sno
        df2['item_nbr'] = ino

        df2['log1p'] = df.log1p
        df2['rmean'] = df.rmean
        df2['rmean'] = df2['rmean'].interpolate()
        df2['rmean'] = df2['rmean'].ffill()
        df2['rmean'] = df2['rmean'].bfill()
        df2 = df2.reset_index()

        EPS = 0.000001
        df2['include1'] = (df2.rmean > EPS)
        # exclude 2013-12-25
        df2['include2'] = df2['include1'] & (df2.date2 != exclude_date) 
        
        dfs.append(df2)

    return pd.concat(dfs, ignore_index=True)

df_train = pd.read_pickle("model/train2.pkl")

store_item_nbrs_path = 'model/store_item_nbrs.csv'
store_item_nbrs = pd.read_csv(store_item_nbrs_path)
store_items = zip(store_item_nbrs.store_nbr, store_item_nbrs.item_nbr)

df_rollingmean = create_rollingmean(df_train)
df_rollingmean.to_pickle('model/df_rollingmean.pkl')