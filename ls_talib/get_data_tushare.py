# coding:utf-8



import tushare as ts


## 生成交易日日期列表

class GetPriceData(object):

    def __init__(self, sid):
        self.security_id = sid

    def get_nbars_backward(self, bars_type= ):



