# -*- coding:utf-8 -*-  
from __future__ import division
import pandas as pd
from pymongo import MongoClient

con_info = ['192.168.259.200', '192.168.250.208']

def con_mongo():
    client = MongoClient(con_info[0])
    db = client.ada
    ced_id = db.ced_indicator
    ced_data = db.ced_indicator_data













