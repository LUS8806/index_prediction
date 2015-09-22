from dateutil.parser import parse
from gmsdk import md
import pandas as pd
import numpy as np
import time

# st = time.time()

md.init('cloud.myquant.cn:8000', '18616622990', '034863342')


class GMPrice(object):
    def __init__(self, symbol=None, bar_types=None, begin_time=None, end_time=None):
        self.symbol = symbol
        self.bar_type = bar_types
        self.begin_time = begin_time
        self.end_time = end_time

    def __var_df(self, symbol=None, bar_type=None, begin_time=None, end_time=None, method=''):
        symbol = symbol if symbol else self.symbol
        bar_type = bar_type if bar_type else self.bar_type
        begin_time = begin_time if begin_time else self.begin_time
        end_time = end_time if end_time else self.end_time
        a = md.__getattribute__(method)
        if method in ['get_bars']:
            r = a(symbol, bar_type, begin_time, end_time)
        elif method in ['get_dailybars', 'get_ticks']:
            r = a(symbol, begin_time, end_time)
        elif method in ['get_last_ticks', 'get_last_dailybars']:
            r = a(symbol)
        # ret = self.__df_trans(r)
        return r

    def __df_trans(r):
        times = []
        close_p = []
        open_p = []
        high_p = []
        low_p = []
        volume = []
        r_bar_type = r[0].bar_type
        for i in r:
            if i.bar_time != 0:
                times.append(parse(str(i.bar_time)))
            else:
                times.append(parse(str(i.bar_time)))
            close_p.append(i.close)
            open_p.append(i.open)
            high_p.append(i.high)
            low_p.append(i.low)
            volume.append(i.volume)
        price = np.transpose([open_p, high_p, low_p, close_p, volume])
        ret = pd.DataFrame(data=price, index=times, columns=['open', 'high', 'low', 'close', 'volume'])
        return ret

    # get between time

    def get_ticks(self, symbol=None, begin_time=None, end_time=None):
        ret = self.__var_df(symbol=symbol,
                            begin_time=begin_time,
                            end_time=end_time,
                            method='get_ticks')
        return ret


    def get_bars(self, symbol=None, bar_type=60, begin_time=None, end_time=None):
        self.bar_type = 'min'
        ret = self.__var_df(symbol=symbol,
                            bar_type=bar_type,
                            begin_time=begin_time,
                            end_time=end_time,
                            method='get_bars')
        return ret

    def get_daily_bars(self, symbol=None, begin_time=None, end_time=None):
        self.bar_type = ''
        ret = self.__var_df(symbol=symbol,
                            begin_time=begin_time,
                            end_time=end_time,
                            method='get_dailybars')
        return ret

    # get_last

    def get_last_ticks(self, symbol=None):
        ret = self.__var_df(symbol=symbol, method='get_last_ticks')
        return ret

    def get_last_bars(self, symbol=None, bar_type=None):
        ret = self.__var_df(symbol=symbol, bar_type=bar_type, method='get_last_bars')
        return ret


if __name__ == '__main__':
    PP = GMPrice()
    ans = PP.get_daily_bars(symbol='CFFEX.IF1506', begin_time='2015-06-01 09:16:00', end_time='2015-06-03 09:18:00')

    print 'Done!'
    print 'Done!'