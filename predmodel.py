# coding:utf-8

"""
=====================================================
指数涨跌预测程序（沪深300，上证综指，深证成指，中小板，创业板）
=====================================================
"""

from __future__ import division
import pandas as pd
import talib as ta
import tushare as ts
import MySQLdb
import numpy as np
import os

from sqlalchemy import create_engine
from multiprocessing.dummy import Pool as ThreadPool
from datetime import timedelta, datetime
from dateutil.parser import parse
from talib.abstract import Function
from ls_talib import get_data_sql
from ls_talib.ls_talib import TSSB_TA

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline

class MLPreTrade(object):
    """
    this class is a machine learning model used to forecast the move direction of the security
    """

    def __init__(self,
                 sid=None,
                 index=True,
                 price_data=None,
                 start_date=None,
                 end_date=None,
                 list_ta=None,
                 look_back=True,
                 ta_lags=0,
                 pred_type=1,
                 data_source=['mysql'],
                 window_size=100):
        """
        :param sid: 证券或期货的代码
        :param start_date: 开始预测的日期
        :param end_date: 结束预测的日期
        :param list_ta: 所选的指标（来自TA_Lib，或者LS_TALib）
        :param pred_type: 预测的类型{0：当天开盘价与收盘价比较，1：隔天收盘价与当天收盘价比较}
        :return:

        """
        self.n_accu = {}
        self.maxn_factors = None
        self.maxn_labels = None
        self.start_date = start_date
        self.end_date = end_date
        self.dic_factors = list_ta
        self.predict_type = pred_type
        self.data_source = data_source
        self.security_id = sid
        self.isIndex = index
        self.ta_lags = ta_lags
        self.window_size = window_size
        self._ta_look_back = self._ta_look_back() if list_ta else 0
        self.look_back = self._ta_look_back + self.window_size + self.ta_lags if look_back else 0
        self.security_type = self._security_type() if sid else sid
        if sid is not None and price_data is None:
            self.price_data = self.get_price_data()
        elif sid is None and price_data is None:
            pass
        elif price_data is not None:
            self.price_data = price_data

    def _security_type(self):
        """
        根据证券代码确定是股票还是期货
        :return:
        """
        if self.security_id[0] in ['0', '3', '6', '9']:
            ret = 'Stock'
        else:
            ret = 'Future'
        return ret

    def _ta_look_back(self):
        """
        获取所需回溯历史数据的长度，计算技术指标时需要n天前的数据，同时预测涨跌时又要windows_size的历史指标
        :return:
        """
        ta_fun_names = ta.get_functions()
        lk_list = []
        for j in self.dic_factors:
            if j[0] in ta_fun_names:
                ta_fun = Function(j[0])
                if j[1] == {}:
                    continue
                else:
                    dic_param = dict(ta_fun.parameters)
                    for key1 in dic_param:
                        dic_param[key1] = j[1][key1]
                        ta_fun.parameters = dic_param
                lk = ta_fun.lookback
                lk_list.append(lk)
            else:
                pass
        ret = max(lk_list)
        return ret

    def set_window_size(self, n_window):
        """
        重新设置回测窗口的长度为n_window
        当窗口长度变化时，self.look_back 与 self.price_data 跟着变化
        """
        self.window_size = n_window
        self.look_back = self._ta_look_back + self.window_size + self.ta_lags
        self.price_data = self.get_price_data()

    def get_price_data(self, sid=None, start=None, end=None, is_index=None, data_source=['mysql']):
        """
        从选取的数据源获取价格数据
        :param data_source:
        'mysql: 公司MySQL数据库', 'csv: 本地csv文件', 'wind: Wind数据库', 'TuShare：TuShare财经数据接口'
        :return: OHLCV价格数据
        """
        code = sid if sid else self.security_id
        start_date = start if start else self.start_date
        end_date = end if end else self.end_date
        is_index = is_index if is_index else self.isIndex
        data_source = self.data_source if self.data_source else data_source
        ret = []
        if data_source[0] == "mysql":
            if start_date == end_date:
                ret = get_data_sql.GetPriceData(code, index=is_index).get_ndays_backward(end_date, self.look_back)
            else:
                ret1 = get_data_sql.GetPriceData(code, index=is_index).get_between_dates(start_date, end_date)
                ret2 = get_data_sql.GetPriceData(code, index=is_index).get_ndays_backward(start_date, self.look_back)
                ret = ret2.combine_first(ret1)

        elif data_source[0] == 'csv':
            ret = pd.read_csv(data_source[1])

        elif data_source[0] == 'wind':
            pass

        elif data_source[0] == 'TuShare':
            if start_date == end_date:
                start_date1 = str((parse(end) - timedelta(self.look_back + 150)).date())
                ret1 = ts.get_hist_data(code, start=start_date1, end=end_date).ix[:, 0:5]
            else:
                start_date1 = str((parse(start_date) - timedelta(self.look_back + 150)).date())
                ret1 = ts.get_hist_data(self.security_id, start=start_date1, end=end_date).sort_index().ix[:, 0:5]
            date_index = pd.DatetimeIndex(ret1.index)
            ret2 = pd.DataFrame(data=ret1.values, index=date_index, columns=ret1.columns)
            start = ret2.index.searchsorted(start_date)-self.look_back
            ret = ret2.ix[start:, 0:5]

        self.price_data = ret
        return ret

    def set_factors_labels(self, price=None, ta_list=None, lags=None):
        """
        根据价格数据及选定的指标计算特征
        :return: the factors array DataFrame
        """
        ta_fun_names = ta.get_functions()
        df_price = price if price else self.price_data
        ta_list = ta_list if ta_list else self.dic_factors
        lags = lags if lags else self.ta_lags
        ta_factors = df_price.copy()
        ii = 1
        for j in ta_list:
            if j[0] in ta_fun_names:
                ta_fun = Function(j[0])
                if j[1] == {}:
                    ta_fun = ta_fun
                else:
                    dic_param = dict(ta_fun.parameters)
                    for key1 in dic_param:
                        dic_param[key1] = j[1][key1]
                    ta_fun.parameters = dic_param
                ta_value = ta_fun(df_price)
                if ta_value.size > len(ta_value):
                    ta_factors["%s_%d" % (j[0], ii)] = ta_value.ix[:, 0]
                else:
                    ta_factors["%s_%d" % (j[0], ii)] = ta_value
            ii += 1

        ta_factors['ACC'] = TSSB_TA(df_price).ACC()
        ta_factors['ADOSC'] = TSSB_TA(df_price).ADOSC()

        # 计算lags指标

        if lags > 0:
            ret = pd.DataFrame(index=ta_factors.index)
            columns = ta_factors.columns
            for col in columns:
                for lag in xrange(0, lags):
                    ret[col + "_%s" % str(lag+1)] = ta_factors[col].shift(lag)
        else:
            ret = ta_factors

        # 定义标签: 上涨为1， 下跌为-1
        if self.predict_type == 1:
            labels = df_price['close'].pct_change().shift(-1)  # 预测明收盘与今收盘的涨跌
        else:
            labels = (df_price['close'] - df_price['open']).shift(-1)  # 预测明收盘与明开盘的涨跌

        labels[labels >= 0] = 1
        labels[labels < 0] = 0

        ret = ret.dropna()
        labels = labels[ret.index]
        self.ta_factors, self.labels = ret, labels
        return ret, labels

    def machine_predict(self, clf='RSVM', factors=None, ta_labels=None, n_windows=None):
        """
        机器学习预测模型，clf定义模型，一般选用’RSVM‘支持向量机，或者’RF‘随机森林
        返回DataFrame：pre，pre['pre_label']为基于当日数据预测下一交易日的涨跌，
        pre['pre_actual']为下一交易日实际涨跌的情况；1为涨，-1为跌。最后一个交易
        日没有下一交易日实际的涨跌情况，所以pre_actual = NaN
        """
        if factors is None and ta_labels is None:
            ta_factors, labels = self.set_factors_labels()
        else:
            ta_factors, labels = factors.copy(), ta_labels.copy()
        n_s = n_windows if n_windows else self.window_size
        models = {
            'LR': LogisticRegression(),
            'LDA': LDA(),
            'QDA': QDA(),
            'LSVC': LinearSVC(),
            'RSVM': SVC(C=10, gamma=30),
            'RF': RandomForestClassifier()
        }

        model = models[clf]
        min_max_scaler = preprocessing.MinMaxScaler()
        pca = PCA(0.9)
        pre = pd.DataFrame(index=ta_factors.index[n_s:], columns=['pre_label', 'pre_actual'])

        for num in range(0, len(ta_factors)-n_s):
            ta_factors_scaled = min_max_scaler.fit_transform(ta_factors.ix[num:num+n_s+1])
            ta_factors_scaled_pca = pca.fit_transform(ta_factors_scaled)
            x_train = ta_factors_scaled_pca[:-1]
            x_test = ta_factors_scaled_pca[-1:]
            y_train = labels[num:num+n_s]
            y_test = labels[num+n_s]
            pre_model = model.fit(x_train, y_train)
            pre['pre_label'][num] = pre_model.predict(x_test).item()
            pre['pre_actual'][num] = y_test

        pre['pre_acu'] = pre['pre_label'] == pre['pre_actual']
        # self.prediction_results = pre

        return pre

    def op_machine_predict(self):
        """
        与machine_predict的区别在于，op_machine_predict在每个window中都通过grid_search
        的方法确定最后的参数。该模型的训练及预测步骤如下：
        对于每一个窗口的数据
        1) 对输入的ta_factors进行标准化的处理
        2) Feature selection：方法可选择
        3) PCA降维
        4) 训练并Grid_Search
        """
        ta_factors, labels = self.set_factors_labels()
        svc = SVC(kernel='linear')
        min_max_scaler = preprocessing.MinMaxScaler()
        pre = pd.DataFrame(index=ta_factors.index[self.window_size:], columns=['pre_label', 'pre_actual'])
        Cs = range(10, 100, 10)
        gammas = range(5, 100, 5)
        n_s = self.window_size
        for num in range(0, len(ta_factors)-n_s):
            ta_factors_scaled = min_max_scaler.fit_transform(ta_factors.ix[num:num+n_s+1])
            x_train = ta_factors_scaled[:-1]
            x_test = ta_factors_scaled[-1:]
            y_train = labels[num:num+n_s]
            y_test = labels[num+n_s]
            # ta_factors_scaled_pca = pca.fit_transform(ta_factors_scaled)
            rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y_train, 2))
            clf = Pipeline([('feature_select', rfecv), ('svm', SVC())])
            # estimator = GridSearchCV(clf, dict(svm__C=Cs, svm__gamma=gammas))
            pre_model = clf.fit(x_train, y_train)
            pre['pre_label'][num] = pre_model.predict(x_test).item()
            pre['pre_actual'][num] = y_test

        pre['pre_acu'] = pre['pre_label'] == pre['pre_actual']
        self.prediction_results = pre

        return pre

    def pool_bw(self, n_s):
        try:
            # self.set_window_size(n)
            # self.maxn_factors, self.maxn_labels = self.set_factors_labels()
            start = self.maxn_factors.index.searchsorted(self.start_date)-n_s
            factors, labels = self.maxn_factors.ix[start:, :], self.maxn_labels.ix[start:]
            pre = self.machine_predict(factors=factors, ta_labels=labels, n_windows=n_s)
            self.n_accu[n_s] = sum(pre['pre_acu']) / len(pre['pre_acu'])
        except Exception as e:
            self.n_accu[n_s] = 0
        return pre

    def best_windows(self, n_range=None):
        """
        穷举法在n_range范围内寻找预测准确率最高的n，返回字典n_accu, 不同n时的准确率
        及 准确率最高的best_n = (acc, n)
        """
        n_range = range(200, 10, -2) if n_range is None else n_range
        self.set_window_size(200)
        self.maxn_factors, self.maxn_labels = self.set_factors_labels()
        pool = ThreadPool(8)
        pool.map(self.pool_bw, n_range)
        pool.close()
        best_n = max(zip(self.n_accu.values(), self.n_accu.keys()))
        n_accu = self.n_accu
        return n_accu, best_n





def best_ns():
    pre_lst = index_list
    i = 1
    for tick in pre_lst:
        try:
            ans, _ = MLPreTrade(sid=tick[0:6],
                                start_date='2014-10-01',
                                end_date='2015-07-08',
                                list_ta=tas).best_windows()
            print "Success: %s" % tick

        except UnboundLocalError:
            print "Failed tick: %s" % tick
            ans = dict(zip(range(200, 10, -2), np.nan*np.ones(95)))

        df_n_acc = pd.DataFrame(ans, index=[tick])
        if i == 1:
            df_n_acc.to_csv("D:\song_code\MLPM\index_best_n1.csv")
        else:
            df_n_acc.to_csv("D:\song_code\MLPM\index_best_n1.csv", header=False, mode='a')

        i += 1

def today_predict():
    args = list()
    for i in xrange(len(index_list)):
        a = MLPreTrade(sid=ts_index_list[i],
                       start_date='2015-01-01',
                       end_date=today,
                       list_ta=tas,
                       window_size=n[i],
                       data_source=['TuShare'])
        print "Today's Price of index %s is" % index_list[i]
        print a.price_data.ix[-1, :]
        print'\n'

        pre = a.machine_predict().ix[-2:, 0:2]
        if i >= 40:
            args.append((index_list[i], today, 0, int(pre['pre_actual'][0]),dt_today))
            args.append((index_list[i], tom, int(pre['pre_label'][1]), None, dt_today))
        else:
            args.append((index_list[i], today, int(pre['pre_label'][0]), int(pre['pre_actual'][0]),dt_today))
            args.append((index_list[i], tom, int(pre['pre_label'][1]), None, dt_today))
    print args
    return args

def write_to_sql(inputs, host=None, user=None, passwd=None):
    cnx = MySQLdb.connect(host=host, user=user, passwd=passwd, db='ada-fd')
    cursor = cnx.cursor()

    sql = '''insert into index_prediction (secu, dt, pred, fact, crt) values (%s,%s,%s,%s,%s)'''
    sql_del = '''delete from index_prediction where dt in ('%s','%s') ''' % (today, tom)

    try:
        cursor.execute(sql_del)
        cnx.commit()
        cursor.executemany(sql, inputs)
    except Exception as e:
        print("执行Mysql: %s 时出错：%s" % (sql, e))
    finally:
        cursor.close()
        cnx.commit()
        cnx.close()

def index_predict_hist():
    engine = create_engine('mysql://ada_user:ada_user@192.168.250.208/ada-fd')
    engine1 = create_engine('mysql://ada_user:ada_user@122.144.134.3/ada-fd')

    fail_list = []
    # trade_cal = pd.read_csv('D:\song_code\MLPM\\trade_cal.csv', header=None, index_col=0)
    for i in xrange(len(ts_index_list)):
        try:
            ans = MLPreTrade(sid=ts_index_list[i],
                             start_date='2014-01-01',
                             end_date='2014-12-31',
                             list_ta=tas,
                             window_size=n[i],
                             data_source=['TuShare'])
            pre = ans.machine_predict()
            dt1 = pre.index[1:].tolist()
            dt = [str(ii.date()) for ii in dt1]
            dt.append('2015/01/05')
            pred = pre['pre_label'].tolist()
            crt = [datetime.now()]*len(pred)
            fact = pre['pre_actual'].tolist()
            fact[-1] = None
            secu = [index_list[i]]*len(pred)
            sum_dict = {'dt': dt, 'secu': secu, 'pred': pred, 'fact': fact, 'crt': crt}
            sum_df = pd.DataFrame(sum_dict, columns=['dt', 'secu', 'pred', 'fact', 'crt'])
            # sum_df.to_sql('index_prediction', engine, index=False, if_exists='append')
            sum_df.to_sql('index_prediction', engine1, index=False, if_exists='append')
            # print sum_df
            print 'Success: %s' % index_list[i]
        except Exception as e:
            print("执行: %s 时出错：%s" % (index_list[i], e))




if __name__ == '__main__':

    tas = [
        ("EMA", {'timeperiod': 7}),
        ("EMA", {'timeperiod':50}),
        ("MACD", {}),
        ("RSI", {'timeperiod':14}),
        ("MOM", {}),
        ("STOCH", {}),
        ("WILLR", {})
        ]

    index_list = ['000001_SH_IX', '000300_SH_IX', '399001_SZ_IX', '399005_SZ_IX', '399006_SZ_IX']
    ts_index_list = ['sh', 'hs300', 'sz', 'zxb', 'cyb']
    pwd = os.getcwd()
    trade_cal = pd.read_csv(os.path.join(pwd,'trade_cal.csv'), header=None, index_col=0)
    n = [60, 60, 60, 60, 60]
    dt_today = datetime.today()
    today = str(dt_today.date())
    t_id = trade_cal.index.searchsorted(today)+1
    tom = trade_cal.index[t_id]

    args = today_predict() # 3:15之后
    # write_to_sql(args, host='192.168.250.208', user='ada_user', passwd='ada_user')
    # write_to_sql(args, host='122.144.134.3', user='ada_user', passwd='ada_user')