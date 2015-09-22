# coding:utf-8
__author__ = 'song.lu'
from dateutil.parser import parse
from gmsdk import md
import pandas as pd
import numpy as np
import time




md.init('cloud.myquant.cn:8000', '18616622990', '034863342')

r = md.get_bars(
    'CFFEX.IF1506',
    60,
    '2015-06-13 09:15:00',
    '2015-06-13 15:30:00',
)


st = time.time()
times = []
closep = []
openp = []
highp = []
lowp = []
volume = []
for i in r:
    times.append(parse(str(i.bar_time)))
    closep.append(i.close)
    openp.append(i.open)
    highp.append(i.high)
    lowp.append(i.low)
    volume.append(i.volume)
price = np.transpose([openp, highp, lowp, closep, volume])
df = pd.DataFrame(data=price, index=times, columns=['open','high','low','close','volume'])

print df.head()
print 'all need:', time.time() - st



