
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PATH=/usr/local/bin:$PATH

cd model

# linear regression by vowpal wabbit
# unexpected features might be used for test data prediction, because I forgot adding --ignore for test data prediction.

vw -d vwdata.vwtxt -c -k -P 1000000 --passes 650 -q AB -q AC -q BM -q CM -q BK -q CK --ignore F --ignore I -f vwdata.vwmdl --l1 0.0000001
vw -d vwdata.vwtxt -t -i vwdata.vwmdl --invert_hash vwdata.vwih

vw -d vwdata_test.vwtxt -t -i vwdata.vwmdl -p vwdata.predict.txt

cd ..
