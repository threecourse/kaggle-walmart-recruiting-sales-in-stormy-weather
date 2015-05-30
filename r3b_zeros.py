import pandas as pd
import numpy as np
import pickle
import os.path

def create_zeros_parameters(_df):

    df = _df[['date2','item_nbr','store_nbr','log1p']].copy()
    df = df.set_index('date2', drop=True)
    df['is_zero'] = np.where(_df.log1p == 0.0, 1, 0)

    loop_range = range(1, 11)

    # calculate forward successive zeros
    # start/end of the dates is treated as non-zero
    cols_forward = ['f{}'.format(i) for i in loop_range]
    for i in loop_range:
        col = cols_forward[i-1]
        colp = cols_forward[i-2]
        df[col] = df.is_zero.shift(-i)
        df[col] = df[col].fillna(0)
        if (i-2 >= 0) : df[col] = df[colp] * df[col]
    df['forward_zeros'] = df[cols_forward].sum(axis=1)

    # calculate back successive zeros
    # start/end of the dates is treated as non-zero
    cols_back = ['b{}'.format(i) for i in loop_range]
    for i in loop_range:
        col = cols_back[i-1]
        colp = cols_back[i-2]
        df[col] = df.is_zero.shift(i)
        df[col] = df[col].fillna(0)
        if (i-2 >= 0) : df[col] = df[colp] * df[col]
    df['back_zeros'] = df[cols_back].sum(axis=1)

    df['min_zeros'] = np.minimum(df.back_zeros, df.forward_zeros)

    g =  df[df.is_zero == 0].groupby(['min_zeros'])
    max_bothside_zeros = g['min_zeros'].count().index.max()
    max_bothside_zeros = np.min([9, np.max([1, max_bothside_zeros])])

    df['max_bothside_zeros'] = max_bothside_zeros
    df = df.drop(cols_forward, axis=1)
    df = df.drop(cols_back, axis=1)
    return df


def create_zeros(_df):
    
    dfs = []

    for sno, ino in store_items:
        df = _df[ (_df.item_nbr == ino) & (_df.store_nbr == sno) ]
        dfn = create_zeros_parameters(df)

        # all dates
        alldates = pd.date_range('2012-01-01', '2014-10-31', freq='D')
        alldates.name = 'date2'
        dfn2 = pd.DataFrame(dfn, index = alldates)

        # fill same values
        dfn2[['item_nbr', 'store_nbr']] = dfn2[['item_nbr', 'store_nbr']].ffill()
        dfn2[['item_nbr', 'store_nbr']] = dfn2[['item_nbr', 'store_nbr']].bfill()
      
        dfn2[['max_bothside_zeros']] = dfn2[['max_bothside_zeros']].ffill()
        dfn2[['max_bothside_zeros']] = dfn2[['max_bothside_zeros']].bfill()

        # calculate previous and next is zero or not
        not_train = dfn2.log1p.isnull()

        dfn2['is_zero_prev'] = dfn2['is_zero']
        dfn2['is_zero_prev'] = dfn2['is_zero_prev'].ffill()
        dfn2['is_zero_prev'] = dfn2['is_zero_prev'].bfill()
        dfn2['is_zero_next'] = dfn2['is_zero']
        dfn2['is_zero_next'] = dfn2['is_zero_next'].bfill()
        dfn2['is_zero_next'] = dfn2['is_zero_next'].ffill()

        dfn2['back_zeros'] = dfn2['back_zeros'].interpolate(method='ffill')
        dfn2['back_zeros'] = dfn2['back_zeros'].bfill()
        dfn2.loc[not_train, 'back_zeros'] = np.where(dfn2.loc[not_train, 'is_zero_prev'], 
                                                     dfn2.loc[not_train, 'back_zeros'] + 1, 0)
        dfn2['back_zeros'] = np.minimum(dfn2['back_zeros'], 10)

        dfn2['forward_zeros'] = dfn2['forward_zeros'].interpolate(method='bfill')
        dfn2['forward_zeros'] = dfn2['forward_zeros'].ffill()
        dfn2.loc[not_train, 'forward_zeros'] = np.where(dfn2.loc[not_train, 'is_zero_next'], 
                                                        dfn2.loc[not_train, 'forward_zeros'] + 1, 0)
        dfn2['forward_zeros'] = np.minimum(dfn2['forward_zeros'], 10)

        dfn2['min_zeros'] = np.minimum(dfn2.back_zeros, dfn2.forward_zeros)

        dfn2['include_zeros'] = (dfn2.min_zeros <= dfn2.max_bothside_zeros)
        dfn2 = dfn2.reset_index(drop = False)

        dfs.append(dfn2)
    
    return pd.concat(dfs, ignore_index=True)

df_train = pd.read_pickle("model/train2.pkl")

store_item_nbrs_path = 'model/store_item_nbrs.csv'
store_item_nbrs = pd.read_csv(store_item_nbrs_path)
store_items = zip(store_item_nbrs.store_nbr, store_item_nbrs.item_nbr)

df_zeros = create_zeros(df_train)
df_zeros.to_pickle("model/df_zeros.pkl")
