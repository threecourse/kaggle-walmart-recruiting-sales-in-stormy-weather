import pandas as pd
import numpy as np
import pickle

def create_features():

    dfs = []

    for sno, ino in store_items:

        alldates = pd.date_range('2012-01-01', '2014-10-31', freq='D')
        alldates.name = 'date2'
        df = pd.DataFrame(None, index = alldates)
        df = df.reset_index()
        df['store_nbr'] = sno
        df['item_nbr'] = ino
        df['date2j'] = (df.date2 - pd.to_datetime("2012-01-01")).dt.days

        df = df.merge(df_baseline[['item_nbr', 'store_nbr', 'date2j', 'ppr_fitted']],
                      how = 'left',
                      on = ['item_nbr', 'store_nbr', 'date2j'])

        df = df.merge(df_rollingmean[['item_nbr', 'store_nbr', 'date2', 'rmean', 'include1', 'include2']],
                      how = 'left',
                      on = ['item_nbr', 'store_nbr', 'date2'])
        
        df = df.merge(df_zeros[['item_nbr', 'store_nbr', 'date2', 'include_zeros']],
                      how = 'left',
                      on = ['item_nbr', 'store_nbr', 'date2'])

        df['include3'] = (df.include2 &  df.include_zeros)

        df = df.reset_index(drop = True)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

df_baseline = pd.read_csv("model/baseline.csv",  sep=",")
df_rollingmean = pd.read_pickle('model/df_rollingmean.pkl')
df_zeros = pd.read_pickle("model/df_zeros.pkl")

store_item_nbrs_path = 'model/store_item_nbrs.csv'
store_item_nbrs = pd.read_csv(store_item_nbrs_path)
store_items = zip(store_item_nbrs.store_nbr, store_item_nbrs.item_nbr)

df_features = create_features()
df_features.to_pickle('model/df_features.pkl')