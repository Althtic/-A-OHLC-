import tushare as ts

pro = ts.pro_api()

df = pro.index_member(index_code='801001.SI', start_date='20180901', end_date='20180930')

print(df)