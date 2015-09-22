# coding:utf-8
__author__ = 'song.lu'

import time
import MySQLdb
import numpy as np
import pandas as pd

ip_list = [('192.168.250.208', {'username': 'ada_user', 'password': 'ada_user'}),
           ('122.144.134.3', {'username': 'fd_user', 'password': 'fd_user!@#'})]


class GetPriceData(object):
    """
    数据读取的种类：
    (1) 给定日期的前n天
    (2) 给定日期的后几天
    (3) 两个给定日期之间
    (4) 最后一个日期的收盘价

    两种返回类型：
    (1) pandas.DataFrame
    (2) csv文件
    """

    def __init__(self,
                 sid,
                 index=False,
                 out='DataFrame',
                 sql_adr=ip_list[1][0],
                 user_name=ip_list[1][1]['username'],
                 password=ip_list[1][1]['password'],
                 database='ada-fd'):
        """
        :param sid: the security ID of the stock
        :param field: the price field list, such as ['open','high','low','close']
        """
        self.cnx = MySQLdb.connect(host=sql_adr, user=user_name, passwd=password, db=database)
        self.mysql_cursor = self.cnx.cursor()
        self.security_id = sid
        self.out_type = out
        self.__table = 'hq_index' if index else 'hq_price'

    def __getdata(self, sql):
        """

        :rtype : object
        """
        mysql_cursor = self.cnx.cursor()
        mysql_cursor.execute(sql)
        price_tuple = mysql_cursor.fetchall()
        price_array = np.array(price_tuple)
        ret = pd.DataFrame(data=np.double(price_array[:, 1:6]),
                           index=price_array[:, 0],
                           columns=['open', 'high', 'low', 'close', 'volume'])
        ret = ret.sort_index(ascending="True")
        ret1 = pd.DataFrame(data=ret.values, index=pd.DatetimeIndex(ret.index), columns=ret.columns)

        if self.out_type == 'DataFrame':
            ans = ret1
        else:
            ans = ret1.to_csv()

        return ans

    def get_ndays_backward(self, end_date, n_days):
        """
        this function get the price data of n_days backward from the end_date to the end_date
        :param end_date: the querying end date
        :param n_days: n_days backward
        :return: the DataFrame
        """
        sql = "SELECT dt,open,high,low,close,vol FROM `ada-fd`.%s WHERE " \
                    "tick = '%s' and dt <= '%s' and vol != 0 " \
                    "order by dt DESC LIMIT %d" % (self.__table, self.security_id, end_date, n_days)

        ret = self.__getdata(sql)

        return ret

    def get_ndays_forward(self, start_date, n_days):
        """
        this function get the price data of n_days forward from the start_date to the end_date
        :param start_date: the querying end date
        :param n_days: n_days forward
        :return: the DataFrame
        """
        sql = "SELECT dt, open,high,low,close,vol FROM `ada-fd`.%s WHERE " \
              "tick = '%s' and dt >= '%s' and vol != 0 " \
              "order by dt ASC LIMIT %d" % (self.__table, self.security_id, start_date, n_days)

        ret = self.__getdata(sql)

        return ret

    def get_between_dates(self, start_date, end_date):
        """
        this function get the price data between start_date and end_date
        :param start_date: the querying start date
        :param end_date: the querying end date
        :return: the DataFrame
        """
        sql = "SELECT dt,open,high,low,close,vol FROM `ada-fd`.%s " \
              "WHERE tick = '%s' and dt between '%s' and '%s' and vol != 0 order by dt ASC" \
              % (self.__table, self.security_id, start_date, end_date)

        ret = self.__getdata(sql)

        return ret

    def get_last_price(self):

        sql = "SELECT dt,open,high,low,close,vol as volume " \
              "FROM `ada-fd`.%s " \
              "WHERE tick = '%s' and vol != 0 order by dt DESC LIMIT 1" \
              % (self.__table, self.security_id)

        ret = self.__getdata(sql)

        return ret

if __name__ == '__main__':
    ans = GetPriceData('000806')
    price = ans.get_between_dates('2015-01-01', '2015-07-03')
    price.to_csv('D:\song_code\MLPM\demo_price_2.csv')

    print price

    print 'Done!'
    print 'Done!'




