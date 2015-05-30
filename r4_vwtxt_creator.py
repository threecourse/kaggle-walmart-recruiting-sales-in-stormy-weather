
import pandas as pd
import numpy as np
import pickle

class VWtxtCreator(object):

    def write_train(self, _df_train, fname):
        df = self.create_df_vw(_df_train, True)
        self.write_txt(df, fname)
        return df

    def write_test(self, _df_test, fname):
        df = self.create_df_vw(_df_test, False)
        self.write_txt(df, fname)
        return df

    def create_df_vw(self, _df, is_train):

        df = _df.copy()
        df['datestr'] = map(lambda d:d.strftime('%Y%m%d'), df.date2) # it's slow..
        df['id2'] = ( np.char.array(df.item_nbr) + "_" + np.char.array(df.store_nbr) + "_" + df.datestr )
        
        df = df.merge(
              df_features[ ['item_nbr', 'store_nbr', 'date2',
                            'ppr_fitted', 'rmean', 'include1', 'include2', 'include3'] ],
              how = 'left',
              on = ['item_nbr', 'store_nbr', 'date2']
              )

        df['baseline'] = df.ppr_fitted
        df['include'] = df.include2
        df['include_prediction'] = df.include3 # use for training, but predict as zero

        # set index again
        df = df.set_index(_df.index)

        # drop not merged rows 
        df = df.dropna()

        # set y (only when train)
        if is_train:
             df['y'] = df.log1p - df.baseline
        else:
             df['y'] = 0.0 

        # exclude dates not effective for linear regression
        df = df[df.include]

        return df
        
    def write_txt(self, _df, fname):
        import csv

        f = open( fname, 'wb' )
        wtr = csv.writer(f)
        for i, row in _df.iterrows():
            newline = "{}".format(row.y)
            newline += (" |A wd{} we:{} hol:{} holwd:{} holwe:{}"
                         .format(row.weekday, row.is_weekend, 
                                 row.is_holiday, row.is_holiday_weekday, row.is_holiday_weekend))
            newline += " |B ino{}".format(int(row.item_nbr))
            newline += " |C sno{}".format(int(row.store_nbr))
            newline += " |D date{}".format(int(row.datestr))
            newline += " |F {}".format(row.holiday_name)
            newline += " |K {}".format(row.around_BlackFriday)
            newline += " |M day{} month{} year{}".format(row.day, row.month, row.year)
            newline += " |W isRS:{} departF:{}".format(row.preciptotal_flag, row.depart_flag)
            newline += " |I id {} avl4 {} rmean {}".format(row.id2, row.include_prediction, row.baseline)
            wtr.writerow( [newline] )

        f.close()

vwtxt_creator = VWtxtCreator()

df_features = pd.read_pickle("model/df_features.pkl")
df_train = pd.read_pickle('model/train2.pkl')
vwtxt_creator.write_train(df_train, "model/vwdata.vwtxt")
print "finished write train"

df_test = pd.read_pickle('model/test2.pkl')
vwtxt_creator.write_test(df_test, "model/vwdata_test.vwtxt")
print "finished write test"



