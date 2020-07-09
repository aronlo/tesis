# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:29:17 2020

@author: Aron
"""


from __future__ import print_function
import datetime
import sys

import numbers
import sqlite3
import numpy as np
import pandas as pd



def convert_dotnet_tick(ticks):
    """Convert .NET ticks to formatted ISO8601 time
    Args:
        ticks: integer
            i.e 100 nanosecond increments since 1/1/1 AD"""
    _date = datetime.datetime(1, 1, 1) + \
        datetime.timedelta(microseconds=(int(ticks)) // 10)
    if _date.year < 1900:  # strftime() requires year >= 1900
        _date = _date.replace(year=_date.year + 1900)
    return _date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3]

# if __name__ == "__main__":
#     try:
#         print(convert_dotnet_tick(int(sys.argv[1])))
#     except:
#         #print("Missing or invalid argument; use, e.g.:"
#               #" python ticks.py 636245666750411542")
#         print("Date: %s " % convert_dotnet_tick(633729059336920080))
    
        
     
path = "./data/grey/keystroke.db"
conn = sqlite3.connect(path)

#Se hace la lectura y se almacena los datos extraídos en la variable df ("dataframe")
df = pd.read_sql_query('select * from keystroke_datas', conn, parse_dates=['date'])

#Se eliminan los registros de los usuarios que no hallan escrito la palabra 'greyc laboratory'
df.drop(df[df['password'] != 'greyc laboratory'].index, inplace = True)

rp = df["rawPress"].iloc[0].split()
rr = df["rawRelease"].iloc[0].split()


rp_values =  []
for i in range(len(rp)):
    if i % 2 != 0:
        rp_values.append(convert_dotnet_tick(rp[i]))
        
rr_values = []

for i in range(len(rr)):
    if i % 2 != 0:
        rr_values.append(convert_dotnet_tick(rr[i]))
        


#Se hace un cierre de la conexión con la base de datos SQLite
conn.close()