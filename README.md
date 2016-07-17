# pthon-
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:48:28 2016

@author: i7
import matplotlib.pyplot as plt

"""
###    cho2
from __future__ import division
import os
import numpy as np
import pandas as pd
from numpy.random import randn
from pandas import Series, DataFrame
from pandas.lib import to_timestamp
import matplotlib.pyplot as plt
home
path0 = 'F:\Python\python for data analysis\pydata-book-master'

#os.getcwd()
os.chdir(path0)
path = 'ch02\usagov_bitly_data2012-03-16-1331923249.txt'
#f = open(path0+path)
f = open(path)
f.readline()              #F:\Python\python for data analysis\pydata-book-master
#list(f)
#len(f.readline())        #327
len(f.readlines())        #3557
f.seek(0)
"""
count = []
for i in f:
    count = append([i])  #~~
    count =len(count)
print count
count = len(open(r"d:\123.txt",'rU').readlines())
print count
open(path).readline()
"""

#df = pd.read_excel('C:\Users\i7\Desktop\cu1704.xlsx')  #~~
#df0 = pd.read_table('C:\Users\i7\Desktop\cu1704.xlsx')
#df = pd.read_csv('C:\Users\i7\Desktop\cu1704.csv')
import json
records = [json.loads(line) for line in open(path)]
records = [json.loads(line) for line in open(path)]
type(records)
records = []
records[0]
records[0]['tz']
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
time_zones[:10]
len(time_zones)
def get_counts(sequence):
    counts={}
    for x in sequence:
        if x in counts:
            counts[x] +=1
        else:
            counts = 1
    return counts
def get_counts(sequence):
    counts = {}
    for x in seuqence:
        if x in counts:
            counts[x] += 1
        else:
            counts = 1
    return counts
from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] +=1
    return counts
def get_counts2(sequnce):
    counts = defaultdict(int)
    for x in sequnce:
        counts[x] += 1
    return counts
get_counts2(time_zones)

len(time_zones)

from pandas import DataFrame,Series
import pandas as pd ;import numpy as np
frame = DataFrame(records)
frame.head()
frame['tz'][:10]
tz_counts = frame['tz'].value_counts()    #  Series 的 value_counts() # chong fu xiang ji shu 

tz_counts = frame['tz'].value_counts()
len(tz_counts )
tz_counts[:10]
clean_tz = frame['tz'].fillna('Missing')
clean_tz = frame['tz'].fillna('Missing')
clean_tz[:10]
clean_tz.head()

clean_tz[clean_tz =='']= 'Unknown' #~~

clean_tz[:10]
clean_tz.head()
tz_counts[:10]
tz_counts[:10].plot(kind = 'barh',rot = 0)
tz_counts[:10].plot(kind = 'barh',rot = 0)
frame['a'][1]
len(frame['a'])
resaults = Series([x.split()[0] for x in frame.a.dropna()])

resaults  = Series([x.split()[0] for x in frame.a.dropna()])
resaults[:5]
resaults.value_counts()[:8]

cframe =frame[frame.a.notnull()]
cframe.head()
cframe.info()
operating_system =np.where(cframe['a'].str.contains('windows'),'windows','Not windows')
#operating_system['windows'].value_counts()
#cframe['windows'].value_counts()
#运转
print operating_system[:5]
#operating_system.info()
by_tz_os=cframe.groupby(['tz',operating_system])
type(by_tz_os)
frame.groupby('tz',oprating_system)
%pwd
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]
agg_counts.info()

# Use to sort in ascending order
indexer = agg_counts.sum(1).argsort()
indexer = agg_counts.sum(1).argsort() #~~ shu xiang xiangjia
indexer[:10]
count_subset = agg_counts.take(indexer)[-10:]
count_subset
import matplotlib.pyplot as plt
plt.figure()

count_subset.plot(kind='barh')
count_subset.plot(kind = 'barh',stacked = True)
plt.figure()

normed_subset = count_subset.div(count_subset.sum(1), axis=0)  # BAI FEN BI 
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)

import pandas as pd
import os
encoding = 'latin1'

upath = os.path.expanduser('F:\Python\python for data analysis\pydata-book-master\ch02\movielens\users.dat')
rpath = os.path.expanduser('F:\\Python\\python for data analysis\\pydata-book-master\\ch02\\movielens\\ratings.dat')
mpath = os.path.expanduser('F:\Python\python for data analysis\pydata-book-master\ch02\movielens\movies.dat')

#mpath = os.path.expanduser('ch02/movielens/movies.dat')

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
mnames = ['movie_id', 'title', 'genres']
#f = open('F:\\Python\\python for data analysis\\pydata-book-master\\ch02\\movielens\\ratings.dat')
#f.close()
users = pd.read_csv(upath, sep='::', header=None, names=unames, encoding=encoding)
ratings = pd.read_table(rpath, sep='::',header=None, names=rnames, encoding=encoding)
ratings = pd.read_csv(rpath, sep='::', header=None, names=rnames,encoding=encoding)
movies = pd.read_csv(mpath, sep='::', header=None, names=mnames, encoding=encoding)
users[:5]
ratings[:5]
movies[:5]

data = pd.merge(pd.merge(ratings, users), movies)

data[:10]
data.info()
data.ix[0]  #.head()  #  扼要概括
data.ix[0]  #  di yi hang
data.ix[:,0] # 第一列
mean_ratings = data.pivot_table('rating', index='title',
                                columns='gender', aggfunc='mean')
mean_ratings = data.pivot_table('rating',rows = 'title',cols='gender', aggfunc='mean')
mean_ratings[:5]
ratings_by_title = data.groupby('title').size()
ratings_by_title[:5]

active_titles = ratings_by_title.index[ratings_by_title >= 250]
active_titles[:10]
type(active_titles)
active_titles[:10]
mean_ratings = mean_ratings.ix[active_titles]
mean_ratings

mean_ratings= mean_ratings.rename(index={'Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)':
                           'Seven Samurai (Shichinin no samurai) (1954)'})

#  了解女性观众  喜欢的电影
top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
top_female_ratings[:10]

                     Measuring rating disagreement

mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']

sorted_by_diff = mean_ratings.sort_index(by='diff')   #  通过某个键排序  sort()
sorted_by_diff = mean_ratings.sort_index(by='diff')
sorted_by_diff[:15]

# Reverse order of rows, take first 15 rows
sorted_by_diff[::-1][:15]  #  qu fan  15  hang

# Standard deviation of rating grouped by title
rating_std_by_title = data.groupby('title')['rating'].std()
# Filter down to active_titles
rating_std_by_title = rating_std_by_title.ix[active_titles]
# Order Series by value in descending order
rating_std_by_title.order(ascending=False)[:10]   #  根据值进行降序排列
###################################################
                      US Baby Names 1880-2010

from __future__ import division
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt ##  为什么在3 的环境中会报错

plt.rc('figure', figsize=(12, 5))
np.set_printoptions(precision=4)

#   %pwd
http://www.ssa.gov/oact/babynames/limits.html

!head -n 10 ch02/names/yob1880.txt
type  ch02/names/yob1880.txt

import pandas as pd
names1880 = pd.read_csv('F:\\Python\\python for data analysis\\pydata-book-master\\ch02\\names\\yob1880.txt', names=['name', 'sex', 'births'])
names1880
names1880.groupby('sex').births.sum()
names1880.groupby('sex').sum()
# 2010 is the last available year right now
years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = 'F:\\Python\\python for data analysis\\pydata-book-master\\ch02\\names\\yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)   # 读到一个数据框中
for year in years:
    path ='F:\\Python\\python for data analysis\\pydata-book-master\\ch02\\names\\yob%d.txt' % year
    frame = pd.read_csv(path,names = columns)
    
    frame['year'] = year
    pieces.append(frame)
    
# Concatenate everything into a single DataFrame

names = pd.concat(pieces, ignore_index=True)

total_births = names.pivot_table('births', rows='year',
                                 cols='sex', aggfunc=sum)
total_births.tail()

total_births.plot(title='Total births by sex and year')

def add_prop(group):
    # Integer division floors
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group

names = names.groupby(['year', 'sex']).apply(add_prop)
names[:5]

np.allclose(names.groupby(['year', 'sex']).prop.sum(),1)

def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]   #  降序  默认升序

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)

pieces = []

for year, group in names.groupby(['year', 'sex']):
    pieces.append(group.sort_index(by='births', ascending=False)[:1000])
    
top1000 = pd.concat(pieces, ignore_index=True)
top1000 = pd.concat(pieces,ignore_index=  True)
top1000.index = np.arange(len(top1000))

import pandas
top1000

                              #################################################################################

    ###   Analyzing naming trends
boys  = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
top1000[top1000.sex == 'M']
total_births = top1000.pivot_table('births', rows='year', cols='name',
                                   aggfunc=sum)
total_births

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False,
            title="Number of births per year")

Measuring the increase in naming diversity

plt.figure()

table = top1000.pivot_table('prop', rows='year',
                            cols='sex', aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex',
           yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))

df = boys[boys.year == 2010]
df

prop_cumsum = df.sort_index(by='prop', ascending=False).prop.cumsum()
prop_cumsum[:10]

prop_cumsum.values.searchsorted(0.5)
df = boys[boys.year == 1900]
in1900 = df.sort_index(by='prop', ascending=False).prop.cumsum()
in1900.values.searchsorted(0.5) + 1

def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')

diversity.head()

diversity.plot(title="Number of popular names in top 50%")

The "Last letter" Revolution

# extract last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'

table = names.pivot_table('births', rows=last_letters,
                          cols=['sex', 'year'], aggfunc=sum)

subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
subtable.head()
subtable.sum()

letter_prop = subtable / subtable.sum().astype(float)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female',
                      legend=False)

plt.subplots_adjust(hspace=0.25)

letter_prop = table / table.sum().astype(float)

dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
dny_ts.head()

plt.close('all')

dny_ts.plot()

Boy names that became girl names (and vice versa)

all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like

filtered = top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()

table = filtered.pivot_table('births', index='year',
                             columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table.tail()

plt.close('all')
table.plot(style={'M': 'k-', 'F': 'k--'})

###########################################

#   unit5   pandas 入门
Getting started with pandas

np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4)


# 缩短小数，简明扼要 # 四位小数
%pwd
Introduction to pandas data structures
Series
obj = Series([4, 7, -5, 3])
obj

obj.values
obj.index

obj2 =Series([4,7,-5,3],index=['d','b','a','c'])
obj2.index

obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]
obj[obj==-5]=-6
obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)
'b' in obj2
'e' in obj2

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4

pd.isnull(obj4)  #  检测缺失数据  obj4.isnull()
pd.notnull(obj4)

obj3
obj4
obj3 + obj4
obj4.name = 'population'   #  列的名字（可能不准确）
obj4.index.name = 'state'  #  行索引的名字
obj4
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj
#    DataFrame

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}#  这是一个字典
frame = DataFrame(data)
frame

DataFrame(data, columns=population)  #按照 columns的顺序排‘列’

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
frame2

frame2.columns
frame2.index
frame2['state']

frame2.ix['three']  # 如果要索引行 应该加一个ix[] ； 函数 列的话 直接句点加列名即可
frame2.state
frame2['debt'] = 16.5
frame2

frame2['debt'] = np.arange(5.)
frame2

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
frame2['eastern']=frame2.state  == 'Ohio'
#  计算机语言有时候他会很简练
frame2['eastern']=frame2.state  #  赋值

frame2

del frame2['eastern']

frame2.columns

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
pop
frame3 = DataFrame(pop)
DataFrame(pop, index=[2001, 2002, 2003])

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3
frame3.values
frame2.values


Index objects

obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
index[1:]
#index[1] = 'd'   #   不可改变的
#index=pd.Index(np.arange(3))    #  这里为什么会是大写和小写有什么不同
index = pd.Index(np.arange(3))   #  通过这种方式改变 index 的内容的索引
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index

frame3

'Ohio' in frame3.columns
2003 in frame3.index

Essential functionality
Reindexing

obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2

obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')   #  使值向前填充

frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])
frame
frame.reindex(columns=['Texas', 'California','Ohio' ])


frame2 = frame.reindex(['a', 'b', 'c', 'd'])  #  添加‘c’行
frame2   

states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)

frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill',
              columns=states)                 #  metfod = 'ffill' 空值向后取，即所说的向前填充
                                              #  bfill 向后 即取前边的值
frame.ix[['a', 'b', 'c', 'd'], states]        #  ix[],重新索引，变得简洁
Dropping entries from an axis

obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
new_obj

obj.drop(['d', 'c'])

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])

data.drop(['Colorado', 'Ohio'])     # drop()  默认删除行，

data.drop('two', axis=1)            #  是  " a x i s "
data.drop('two', axis=0)            #  是  " a x i s " 为什么不能=0

data.drop(['two', 'four'], axis=1)  # aix=1 删除列
Indexing, selection, and filtering

obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'd']]  #  1 个以上的必须加[]

obj[[1, 3]]           #  1 个以上的必须加[]

obj[obj < 2]          #  布尔值索引

obj['b':'c']

obj['b':'c'] = 5
obj

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data
data.ix[:,0]
data[:0]
data['two']

data[['three', 'one']]

data[:2]

data[data['three'] > 5]    #  'three'  列大于5 的
data[data > 5]
data > 5
data[data < 5] = 0
data
data.ix['Colorado', ['two', 'three']]     #  行列的索引  ix[] 先行后列
data.ix[['Colorado', 'Utah'], [3, 0, 1]]  #  后边的是下标
data.ix[2]   #  第二行
data.ix[:'Utah', 'two']
data.ix[data.three > 5, :3]   大于5 的前3 列
data.ix[:,:]
Arithmetic and data alignment

s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1
s2
s1 + s2

df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
df1 + df2    #  相应位置上的书相加
df1.add(df2,fill_value=0)     #   如果没有则会被填充
Arithmetic methods with fill values

df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1
df2
df1 + df2
df1.add(df2, fill_value=0)   #

df1.reindex(columns=df2.columns, fill_value=0)

Operations between DataFrame and Series

arr = np.arange(12.).reshape((3, 4))
arr = Series(range(4), index=list('abcd'))    #arr = Series(range(4.),index=list('abcd'))
#range() 不能接受小数
arr

arr[0]

arr - arr[0]  # 广播

frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]    #  第一行  #  默认是从行的下标索引的
frame
series
frame - series

series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2

series3 = frame['d']
series3
frame
frame-series3  # 默认情况 广播 按照排的方式自上而下广播
frame.sub(series3, axis=0)    #  传入轴号在列上广播

Function application and mapping

frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame
np.abs(frame)

f = lambda x: x.max() - x.min()

frame.apply(f)   #  竖的方向上  # b d e # 默认在列的方向上用函数f 即求差

frame.apply(f, axis=1)  #  行的方向上

def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)
#sys.path.append()
format = lambda x: '%.2f' % x
frame.applymap(format)   #  applymap()  添加浮点数

frame['e'].map(format)    # map 属于Series的方法
Sorting and ranking

obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()

frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])
frame.sort_index()  #   按照航 标题的书序排列

frame.sort_index(axis=1)   #  列标题的书序

frame.sort_index(axis=1, ascending=False) #  降序

obj = Series([4, 7, -3, 2])
obj.order()    #   依据值  排序（默认升）
obj.order(ascending=False)
obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.order() #  NaN 值 置于尾

frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame

frame.sort_index(by='b')
frame.sort_index(by=['a','b'])
b = frame.pop('b')

df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
                 'b': [np.nan, 2., np.nan, 6.],
                 'c': range(2, 18, 4)})
df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
                 'b': [np.nan, 3., 4., 6., 8.]})

df1
b = df1.pop('b')
['b','a'] = df1.drop(['b','a'])
fd=df1[['a','b']]
fd
df1 = df1.drop(['a','b'],axis = 1)
df1
a = fd['a'][:] 
df1.insert(0,a,df1['c'])

b
df1
df1.insert(2,'b',df1['c']) 
df1




obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()

obj.rank(method='first')

obj.rank(ascending=False, method='max')

frame = DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
                   'c': [-2, 5, 8, -2.5]})
frame

frame.rank(axis=1)
Axis indexes with duplicate values

obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj

obj.index.is_unique
obj['a']

obj['c']

df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df

df.ix['b']
Summarizing and computing descriptive statistics

df = DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
               index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
df

df.sum()                       #  不加入NaN 的计算

df.sum(axis=1)

df.mean(axis=1, skipna=False)  #  skipna 排除缺失值  加入NaN 的计算

df.idxmax()

df.cumsum()                    # 连加

df.describe()

obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()
Correlation and covariance

import pandas.io.data as web

all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker)
type(all_data);#head(all_data)
all_data.values();all_data.keys()
all_data.items();
type(all_data['AAPL'].head())
all_data['AAPL'].head().count()
all_data['AAPL'].count()
all_data['AAPL'].info()

price = DataFrame({tic: data['Adj Close']
                   for tic, data in all_data.iteritems()})
price = DataFrame({tic:data['Adj Close'] for tic,data in all_data.items()})

volume = DataFrame({tic: data['Volume']
                    for tic, data in all_data.iteritems()})

price.pct_change()
returns = price.pct_change()
type(returns)
returns.head()
returns.tail()
returns.MSFT.corr(returns.IBM)

returns.MSFT.cov(returns.IBM)

returns.corr()
returns.cov()

returns.corrwith(returns.IBM)

returns.corrwith(volume)  returns.corrwith(volume) 

    # 计算百分比的 变化和成交量的相关性系数  #  aixs = 1 按行计算

Unique values, value counts, and membership

obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj.count()
obj.count
uniques = obj.unique()
uniques
obj.value_counts()
obj.value_counts

pd.value_counts(obj.values, sort=False)
pd.value_counts(obj.values,sort= True)    #  前后没变化
mask = obj.isin(['b', 'c'])
mask
obj[mask]
data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                  'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
data

result = data.apply(pd.value_counts).fillna(0)   # 得到相关列的柱状图
result
Handling missing data

string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data
string_data.isnull()

string_data[0] = None
string_data
string_data.isnull()
Filtering out missing data

from numpy import nan as NA
data = Series([1, NA, 3.5, NA, 7])
data.dropna()  ;  也可通过不了值索引  data[data.notnull()]

data = DataFrame([[1., 6.5, 3.], [1., NA, NA],
                  [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
data
cleaned
data[4] = NA                 #  添加新列
data.dropna(how='all')  #  之丢弃全为 NA 的行
data.dropna(how='all',axis = 1)  #  之丢弃全为 NA 的列

data[4] = NA    #  添加新列
data

df = DataFrame(np.random.randn(7, 3))
df.ix[:4, 1] = NA; df.ix[:2, 2] = NA  #  包含末尾  是闭区间
df

df.dropna(thresh=3)                   #  删除[:4]  的行
Filling in missing data
df.fillna(0)                       #所有NaN 的地方都填充上0
df.fillna({1: 0.5, 3: -1})
df.fillna({1: 0.5, 2: -1})
# always returns a reference to the filled object
_ = df.fillna(0, inplace=True)  #  所有NaN 的地方都填充上0
df

df = DataFrame(np.random.randn(6, 3))
df.ix[2:, 1] = NA; df.ix[4:, 2] = NA
df

df.fillna(method='ffill')  #  forword向前填充

df.fillna(method='ffill', limit=2)  # forword向前填充 2 个空格的位置

data = Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())    #  填充为  mean 值

Hierarchical indexing  #  层次化索引

data = Series(np.random.randn(10),
              index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                     [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data

data.index

data['b']

data['b':'c']

data.ix[['b', 'd']]

data[:, 2]  #   在2列位置上的所有数  内层选取

data.unstack()  #  透视图

data.unstack().stack()

frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=[['Ohio', 'Ohio', 'Colorado'],
                           ['Green', 'Red', 'Green']])
frame
df_obj.sort(columns = ‘’)   #按列名进行排序
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame

frame['Ohio']
MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'],
                        ['Green', 'Red', 'Green']], names=['state', 'color'])

Reordering and sorting levels
# 重排分级排序
frame.swaplevel('key1', 'key2')   #  交换标题名

frame.sortlevel(1)

frame.swaplevel(0, 1).sortlevel(0)

Summary statistics by level

frame.sum(level='key2')    #  上下求和

frame.sum(level='color', axis=1)  #  左右求和 #根据级别汇总统计
Using a DataFrame's columns

frame = DataFrame({'a': range(7), 'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})
frame

frame2 = frame.set_index(['c', 'd'])   #‘c’,'d'  转化为行标签
frame2

frame.set_index(['c', 'd'], drop=False)   #'c','d'列维持原来不变

frame2.reset_index('c')     # 只还原“c”  
frame2.reset_index()        # frame.set_index()  的逆

Other pandas topics
Integer indexing

ser = Series(np.arange(3.))
ser.iloc[-1]
ser
ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
ser2
ser2[-1]
ser.ix[:1]   #  ix[]  进行切片
ser3 = Series(range(3), index=[-5, 1, 3])
#                   Series  的 iget_value()
#                       DataFrame 的  irow() icol()  方法
ser3
ser3.iloc[2]

frame = DataFrame(np.arange(6).reshape((3, 2)), index=[2, 0, 1])
frame
frame.iloc[0]
frame.irow(0)   # o 行0 列
Panel data

import pandas.io.data as web

#Panel 创建 panel  对象

pdata = pd.Panel(dict((stk, web.get_data_yahoo(stk))
                       for stk in ['AAPL', 'GOOG', 'MSFT', 'DELL']))
pdata
pdata = pdata.swapaxes('items', 'minor')

pdata['Adj Close'][-10:]

pdata.ix[:, '6/1/2012', :]   #  截面数据

pdata.ix['Adj Close', '5/22/2012':, :]

stacked = pdata.ix[:, '5/30/2012':, :].to_frame()
#  to_panel  是to_frame()  的逆运算
stacked

stacked.to_panel()
###############################################

#F:\Python\python for data analysis\pydata-book-master\ch06
Data loading, storage, and file formats
# unit6
from __future__ import division
from numpy.random import randn
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))

np.set_printoptions(precision=4)    

%pwd
Reading and Writing Data in Text Format

!cat ch06/ex1.csv

os.chdir('F:\Python\python for data analysis\pydata-book-master')
os.getcwd()

os.listdir('F:\Python\python for data analysis\pydata-book-master')
fls = os.listdir('F:\Python\python for data analysis\pydata-book-master')

fch06 = os.listdir('F:\Python\python for data analysis\pydata-book-master\ch06')
for i in fch06:
    print i,

!type cho6/ex1.csv    ####  ~~~

df = pd.read_csv('ch06/ex1.csv')
df

pd.read_table('ch06/ex1.csv', sep=',')

!cat ch06/ex2.csv
!type ch06 / ex2.csv
pd.read_csv('ch06/ex2.csv', header=None)
pd.read_csv('ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])

names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('ch06/ex2.csv', names=names, index_col='message')
#pd.read_csv('ch06/ex2.csv', names=names, index_col='mesage')

!cat ch06/csv_mindex.csv
                              f =os.listdir('ch06')
                              #sorted(f,key=csv)
list(open('ch06/csv_mindex.csv'))

parsed = pd.read_csv('ch06/csv_mindex.csv', index_col=['key1', 'key2'])
#parsed = pd.read_csv('ch06/csv_mindex.csv', index_col=['key1','key2'])
parsed

list(open('ch06/ex3.txt'))
list(open('ch06/ex3.txt'))
result = pd.read_table('ch06/ex3.txt', sep='\s+')

result = pd.read_table('ch06/ex3.txt',sep='\s+')
result

# !cat ch06/ex4.csv encoding=utf-8
# !type ch06/ex4.csv encoding=utf-8
list(open('ch06/ex4.csv'))
pd.read_csv('ch06/ex4.csv', skiprows=[0, 2, 3])

!cat ch06/ex5.csv
list(open('ch06/ex5.csv'))
result = pd.read_csv('ch06/ex5.csv')
result
pd.isnull(result)
pd.notnull(result)
result.head()
#result.irow()
list(open('ch06/ex5.csv'))  na_value=['NULL']  #  其中NULL等于NaN
result = pd.read_csv('ch06/ex5.csv', na_values=['NULL'])
result

sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('ch06/ex5.csv', na_values=sentinels)


Reading text files in pieces
ls = list(open('ch06/ex6.csv'))
len(ls)
result = pd.read_csv('ch06/ex6.csv')
result
result.index
len(result.index)
result.columns
len(result.columns)
pd.read_csv('ch06/ex6.csv', nrows=5)
#pd.read_csv('ch06/ex6.csv', n)
chunker = pd.read_csv('ch06/ex6.csv', chunksize=1000)
chunksize  的 get_chunk  任意大小的快
chunker
chunker.get_chunk(101)
type(chunker)

chunker = pd.read_csv('ch06/ex6.csv', chunksize=1000)

tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

    tot = tot.order(ascending=False);   tot.sort()
 len(tot)
 tot
tot[:10]

Writing data out to text format

data = pd.read_csv('ch06/ex5.csv')

data

data.to_csv('ch06/out.csv')
data.to_csv('ch06/out1.csv')
pd.read_csv('ch06/out1.csv')
!cat ch06/out.csv
!type ch06 / out.csv

data.to_csv(sys.stdout, sep='|')        # sys.stdout 直接打印出文本以竖线为分隔符

data.to_csv(sys.stdout, na_rep='NULL')  #指定缺失值
data.to_csv(sys.stdout, na_rep='#')

data.to_csv(sys.stdout, index=False, header=False)

data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])

pd.date_range('1/1/2000',periods=7)
dates = pd.date_range('1/1/2000', periods=7)
ts = Series(np.arange(7), index=dates)

ts.to_csv('ch06/tseries.csv')

!cat ch06/tseries.csv
list(open('ch06/tseries.csv'))

pd.read_csv('ch06/tseries.csv')
Series.from_csv('ch06/tseries.csv', parse_dates=True)

Manually working with delimited formats

!cat ch06/ex7.csv
list(open('ch06/ex7.csv'))

import csv
f = open('ch06/ex7.csv')
f.close()
reader = csv.reader(f)

f.seek(0)
for line in reader:
    print line
zip()
lines = list(csv.rea
der(open('ch06/ex7.csv')))
lines
lines[0];  lines[1] ;  lines[2]

header, values = lines[0], lines[1:]
data_dicts = {h: v for h, v in zip(header, zip(*values))}
data_dicts

class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL

os.chdir('F:\Python\python for data analysis\pydata-book-master')
 k = open('ch06/mydata.csv','w')
 k.close()
# pd.read_csv('ch06/mydata.csv')
 k.write('abc')
 csv.writer?
with open('mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))
with open('mydata.csv','w') as f:
    m = csv.writer(f,delimiter='!')
    m.writerow(('one', 'two', 'three'))
    m.writerow (('1' , '2' , '3'))
    m.writerow (('4' , '5' , '6'))
    m.writerow (('7' , '8' , '9'))
pd.read_csv('mydata.csv')
%close mydata.csv
JSON data

obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""

import json
result = json.loads(obj)
result  # type(result)
result.items()

for i in result.keys():
    print '%s<------->%s' % (i, result[i])
# 判断一个键是不是属于某个字典

json.dumps(result)
asjson = json.dumps(result)
type(asjson)        #str

siblings = DataFrame(result['siblings'], columns=['name', 'age'])
#siblings = DataFrame(asjson['siblings'],columns = ['name','age'])
siblings

XML and HTML, Web scraping
NB. The Yahoo! Finance API has changed and this example no longer works


from lxml.html import parse
from urllib2 import urlopen

parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
type(parsed)        # lxml.etree._ElementTree

doc = parsed.getroot()
type(doc)           #  lxml.html.HtmlElement
links = doc.findall('.//a')
links[15:20]
lnk = links[28]
lnk
lnk.get('href')             # get (针对URL)
lnk.text_content()          # text_content （针对 文本)

urls = [lnk.get('href') for  lnk in links ]
urls = [lnk.get('href') for lnk in doc.findall('.//a')]
urls[-10:]

tables = doc.findall('.//table')   ####   table 放置 的 表格
type(tables)   ;     len(tables)

                                 calls = tables[9]
                                 puts = tables[13]
calls0 = tables[0]
calls = tables[1]
puts  = tables[2]
#DataFrame(calls)[:2]
rows = calls.findall('.//tr') #  每个表格都有一个标题行
                              #  然后才是数据行
 # 获得文本标题行就是  “  th  ”单元格；  对于数据行则是“  td  ”
 cols = calls.findall('.//td')
 type(rows)
 type(cols)

def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]
d1 = rows[0].text_content()
d2 = cols[0].text_content()
def _unpack(row, kind = 'td'):
    elts = row.findall('.//%s % kind')
    return [val.text_content() for val in elts]

_unpack(rows[0], kind='th')
_unpack(rows[1], kind='td')

from pandas.io.parsers import TextParser  ##~~

d1 = rows[0]    #.text_content()
d2 = cols[0]    #.text_content()
TextParser(d1[1:]).get_chunk()

def parse_options_data(table):
    rows = table.findall('.//tr')
    header = _unpack(rows[0], kind='th')
    data = [_unpack(r) for r in rows[1:]]
    return TextParser(data, names=header).get_chunk()

call_data = parse_options_data(calls)
put_data = parse_options_data(puts)
call_data[:10]
put_data[:1]
Parsing XML with lxml.objectify

%cd ch06/mta_perf/Performance_XML_Data

!head -21 Performance_MNR.xml

from lxml import objectify

path = 'Performance_MNR.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()

data = []

skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ',
               'DESIRED_CHANGE', 'DECIMAL_PLACES']

for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)

perf = DataFrame(data)
perf

from StringIO import StringIO
tag = '<a href="http://www.googgle.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()
root.get('href')

root.text

Binary data formats  # 二进制数据格式

cd ../..

frame = pd.read_csv('ch06/ex1.csv')
frame
frame.save('ch06/frame_pickle1')
frame.to_pickle('ch06/frame_pickle2')
pd.read_pickle('ch06/frame_pickle')
pd.load('ch06/frame_pickle')

Using HDF5 format
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
store

store['obj1']  # HDF5 文件可以像 字典一样通过 键 访问
store['obj1_col']
store.close()
os.remove('mydata.h5')

Interacting with HTML and Web APIs

import requests
url = 'https://api.github.com/repos/pydata/pandas/milestones/28/labels'
resp = requests.get(url)
resp
type(resp)
#len(resp)
data = json.loads(resp.text)       # 许多web API 返回的都是JSON 字符串
type(data)
data[:5]                #~~
data[:]
tweet_fileds = ['created_at','from_uesr','id','txt']
tweets = DataFrame(data['results'],columns=tweet_fileds)
tweets
tweets.ix[7]
issue_labels = DataFrame(data)
issue_labels

Interacting with databases

import sqlite3

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);"""

con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()

data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

con.executemany(stmt, data)
con.commit()

cursor = con.execute('select * from test')
rows = cursor.fetchall()
rows

cursor.description

DataFrame(rows, columns=zip(*cursor.description)[0])
import pandas.io.sql as sql
sql.read_sql('select * from test', con)

import xlrd
import openpyx1

import pymongo
con = pymongo.Connection('localhost',port = 27017)
tweets =  con.db.tweets
import requests,json
url = 'http://search.twitter.com/serach.json?q=python%2opandas'
data = json.loads(requests.get(url).text)
for tweet in data['results']:
    tweets.save(tweet)
cursor = tweets.find({'feom_user':'wesmckinn'})
tweets_fields=['created_at','from_user','id'.'text']
result = DataFrame(list(cursor),columns = tweets_fields)
########################################################################################

unit7
Data Wrangling: Clean, Transform, Merge, Reshape

from __future__ import division         #~~
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas
import pandas as pd
#~~
np.set_printoptions(precision=4, threshold=500)
pd.options.display.max_rows = 100

%matplotlib inline

Combining and merging data sets
Database-style DataFrame merges

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'],
                 'data2': range(3)})
df1
df2
pd.merge(df1, df2)  #  多对一的合并，重叠列的列名当做键  #取交集

pd.merge(df1, df2, on='key')

df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                 'data2': range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey')  # inner连接（交集） #默认是键的连接

pd.merge(df1, df2, how='outer')  # 外连接  并集

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                 'data2': range(5)})

df1
df2

pd.merge(df1, df2, on='key', how='left')  #  多对多连接产生行的笛卡尔积
pd.merge(df1, df2, on='key', how='right')
pd.merge(df1, df2, how='inner')           #  多对多连接产生行的笛卡尔积

left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
pd.merge(left, right, on=['key1', 'key2'], how='outer')#追加到重叠列名的末尾
                                                      #  相同的部分合并

pd.merge(left, right, on='key1')   #  但对一个作为‘键’

pd.merge(left, right, on='key1', suffixes=('_left', '_right'))

Merging on index# 索引上的合并

left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
                  'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])

left1
right1

pd.merge(left1, right1, left_on='key', right_index=True)

pd.merge(left1, right1, left_on='key', right_index=True, how='outer')

#层次化索引
lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                   'key2': [2000, 2001, 2002, 2001, 2002],
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
lefth
righth

pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)#交集

pd.merge(lefth, righth, left_on=['key1', 'key2'],
         right_index=True, how='outer')  #并集

left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],
                 columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                   index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])

left2
right2

pd.merge(left2, right2, how='outer', left_index=True, right_index=True)
left2.join(right2, how='outer')

left1.join(right1, on='key')

another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                    index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])

left2.join([right2, another])
left2.join([right2, another], how='outer')

Concatenating along an axis # 轴向连接

arr = np.arange(12).reshape((3, 4))
arr
#  concatenate()
np.concatenate([arr, arr], axis=1)

s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])

pd.concat([s1, s2, s3])             #   竖向连接
pd.concat([s1, s2, s3], axis=1)     ##  横向连接  aixs= 1 # 逐列显示，变成DataFrame

s4 = pd.concat([s1 * 5, s3])
s4
pd.concat([s1, s4], axis=1)
pd.concat([s1, s4], axis=1,join='inner')
pd.concat([s1, s4], axis=1, join='inner')

pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])

result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])#   竖向排列
result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'],axis=1)#['one', 'two', 'three']  横向排列
result

# Much more on the unstack function later

result.unstack()

pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                columns=['three', 'four'])
df1
df2
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
pd.concat([df1, df2],  keys=['level1', 'level2'])

pd.concat({'level1': df1, 'level2': df2}, axis=1)

pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],
          names=['upper', 'lower'])

df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])

df1
df2

pd.concat([df1, df2], ignore_index=True)#  ignor_index = True 竖向 向下 添加
Combining data with overlap

a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64),
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
a
b
np.where(pd.isnull(a), b, a)

b[:-2].combine_first(a[2:])     # 微数据b[:2] 打补丁
                                # 有值则相应位置不补 维持原状
                                # 没有则补
df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
                 'b': [np.nan, 2., np.nan, 6.],
                 'c': range(2, 18, 4)})
df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
                 'b': [np.nan, 3., 4., 6., 8.]})
df1.combine_first(df2)

Reshaping and pivoting
Reshaping with hierarchical indexing

data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
data
result = data.stack()       #轴向旋转
result

result.unstack()

result.unstack(0)           # 0 轴列对换

result.unstack('state')  #  将   ‘state’    返回到列

s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2.unstack()                     # unsatck stack  的操作是最内层的

data2.unstack().stack()              #  ‘stack’ 默认过滤确实数据

data2.unstack().stack(dropna=False)   #     过程是可逆的

df = DataFrame({'left': result, 'right': result + 5},
               columns=pd.Index(['left', 'right'], name='side'))
df
df.unstack('state')    #    将'state'返回列

df.unstack('state').stack('side')   #  将 ‘side’ 放在行
Pivoting "long" to "wide" format

data = pd.read_csv('ch07/macrodata.csv')

# print (data.index.size,data.columns.size)
    def dfs (data):
        i = data.index.size
        c = data.columns.size
        return i,c

type(data)
[u'year', u'quarter', u'realgdp', u'realcons', u'realinv', u'realgovt', u'realdpi', u'cpi', u'm1', u'tbilrate', u'unemp', u'pop', u'infl', u'realint']
data1=data.stack()
data1[:4]
periods = pd.PeriodIndex(year=data1.year, quarter=data1.quarter, name='date')
periods = pd.PeriodIndex(year=data.year,quarter=data.quarter,name='date')
#~~
data = DataFrame(data.to_records(),
                 columns=pd.Index(['realgdp', 'infl', 'unemp'], name='item'),
                 index=periods.to_timestamp('D', 'end'))

ldata = data.stack().reset_index().rename(columns={0: 'value'})
wdata = ldata.pivot('date', 'item', 'value')

ldata[:10]

pivoted = ldata.pivot('date', 'item', 'value')
pivoted.head()

ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]

pivoted = ldata.pivot('date', 'item')
pivoted[:5]

pivoted['value'][:5]

unstacked = ldata.set_index(['date', 'item']).unstack('item')
unstacked[:7]

Data transformation
Removing duplicates

data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
data
data.duplicated()
data.drop_duplicates()
                    #duplicated  duplicates  默认返回第一个出现的值
data['v1'] = range(7)
data.drop_duplicates(['k1'])      # 只希望通过  ‘K1’ 列 滤重复项

data.drop_duplicates(['k1', 'k2'], take_last=True) #传入take_last=True则最后一个出现的
data.drop_duplicates(['k1', 'k2'], take_last=False)

Transforming data using a function or mapping

data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham',
                           'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data

meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}

data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data

data['food'].map(lambda x: meat_to_animal[x.lower()])

Replacing values

data = Series([1., -999., 2., -999., -1000., 3.])
data

data.replace(-999, np.nan)

data.replace([-999, -1000], np.nan)

data.replace([-999, -1000], [np.nan, 0])

data.replace({-999: np.nan, -1000: 0})

Renaming axis indexes

data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data
data.index.map(str.upper)

data.index = data.index.map(str.upper)
data

data.rename(index=str.title, columns=str.upper)

data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})  # rename() 实现轴标签的更新

# Always returns a reference to a DataFrame

_ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data

Discretization and binning

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats

cats.labels

cats.levels

pd.value_counts(cats)

pd.cut(ages, [18, 26, 36, 61, 100], right=False)# Z左闭右开
pd.cut(ages, [18, 26, 36, 61, 100], right=True)# 左开右闭
pd.cut(ages, [18, 26, 36, 61, 100]) # 默认左开右闭

group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)

data = np.random.rand(20)
 k  = pd.cut(data, 4, precision=2)    # 等距离而不是等个数 即：区间等差
pd.value_counts(k)

qcut

data = np.random.randn(1004) # Normally distributed
cats = pd.qcut(data, 4) # Cut into quartiles # 按四分位数切割 得到等长
cats
pd.value_counts(cats)
# 设置自定义分位数
 h = pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
 h = pd.qcut(data, 10)
 h
pd.value_counts (h)
pd.value_counts(h)

Detecting and filtering outliers
#   检测和过滤异常值
np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
data.describe()
dfs(data)
col = data[3]
col[np.abs(col) > 3]

data[(np.abs(data) > 3).any(1)]# 选出全部含有  “超过3或-3 的  ‘值’  ” 的行
#data[(np.abs(data) > 3).any(0)]

data[np.abs(data) > 3] = np.sign(data) * 3  # abs() 大于3 的数会被设为 等于 3
data.describe()
# 排列和随机采样
Permutation and random sampling
随机排列
df = DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
sampler
df
df.take(sampler)

df.take(np.random.permutation(len(df))[:3])

bag = np.array([5, 7, -1, 6, 4])
sampler = np.random.randint(0, len(bag), size=10) # random.randint() 得到一组随机整数
sampler
bag
draws = bag.take(sampler)  #  通过用 .tkae()  随机整数 的方式获取随机样本
draws

Computing indicator / dummy variables
#哑变量

df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': range(6)})
pd.get_dummies(df['key'])

dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ch02/movielens/movies.dat', sep='::', header=None,
                        names=mnames)
type(movies)
movies[:10] # import pandas
movies.genres[:10]

genre_iter = (set(x.split('|')) for x in movies.genres) #
# len(genre_iter)
#设置成set 可以得到唯一值
type(genre_iter)  #  generator  (发生器)
genres = sorted(set.union(*genre_iter))
genres
dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)
type(dummies)
dummies.head()
for i, gen in enumerate(movies.genres):
    dummies.ix[i, gen.split('|')] = 1
dummies.ix[2,movies.genres[5].split('|')] =1 #将各行的项设置为  1
dummies.ix[2,movies.genres[5].split('|')]

movies_windic = movies.join(dummies.add_prefix('Genre_'))# add_pre_fix()
movies_windic.ix[0]


np.random.seed(12345)
values = np.random.rand(10)
values

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))

String manipulation
String object methods

val = 'a,b,  guido'
val.split(',')

pieces = [x.strip() for x in val.split(',')]
pieces

first, second, third = pieces
first + '::' + second + '::' + third

'::'.join(pieces)

'guido' in val

val.index(',')
val.find(':')
val.index(':')
val.count(',')
val.replace(',', '::')
val.replace(',', '')

Regular expressions

import re
text = "foo    bar\t baz  \tqux"
re.split('\s+', text)

regex = re.compile('\s+')
# 用 re.compile() 自己编译  compile

regex
regex.split(text)

regex.findall(text)
# 得到匹配 regex 的所有模式
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
# re.IGNORECASE makes the regex case-insensitive 对大小写不敏感
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)

regex.findall(text)

m = regex.search(text)  #  search(text)  返回第一个匹配项
m
print m
m.start()
m.end()
text[m.start():m.end()]  #  注意 m.start()\m.end()巧妙运用
text
print(regex.match(text))
#match 只匹配出现在字符串开头的模式
print(regex.sub('REDACTED', text))
# sub 将匹配到的模式替换为指定的字符串
用圆括号裹起来
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
regex.findall(text)
m = regex.match('wesm@bright.net')
m.groups()   #  group 的方法 返回模式各段组成的元祖
findall（返回所有匹配项） ; sub（替换为指定字符串） ;match （返回第一个匹配项的首部）  ;search('只返回第一个匹配项')
regex.findall(text)   # 返回元组
print regex.search(text)

print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))

regex = re.compile(r"""
    (?P<username>[A-Z0-9._%+-]+)
    @
    (?P<domain>[A-Z0-9.-]+)
    \.
    (?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE|re.VERBOSE)

m = regex.match('wesm@bright.net')
m.groupdict()

Vectorized string functions in pandas
字符串矢量化
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = Series(data)
data
data.isnull()
data.str.contains('gmail')  # 检查各电子邮件是否含有  ‘gmail’
pattern

data.str.findall(pattern, flags=re.IGNORECASE)
matches = data.str.match(pattern, flags=re.IGNORECASE)
matches
type(matches)
matches.str.get(1)
matches.str[0]
data.str[:5]

Example: USDA Food Database
{ "id": 21441, "description": "KENTUCKY FRIED CHICKEN, Fried Chicken, EXTRA CRISPY, Wing, meat and skin with breading", "tags": ["KFC"], "manufacturer": "Kentucky Fried Chicken", "group": "Fast Foods", "portions": [ { "amount": 1, "unit": "wing, with skin", "grams": 68.0 }, ... ], "nutrients": [ { "value": 20.8, "units": "g", "description": "Protein", "group": "Composition" }, ... ] }

import json
db = json.load(open('ch07/foods-2011-10-03.json'))
type(db)   #   list
len(db)
type(db[0])
db[0].keys()

db[0]['nutrients']
db[0]['nutrients'][0]

nutrients = DataFrame(db[0]['nutrients'])
nutrients[:7]
nutrients.describe()
nutrients.index.size
nutrients.columns.size
nutrients.head()

info_keys = ['description', 'group', 'id', 'manufacturer']
info = DataFrame(db, columns=info_keys)
info[:5]
info
pd.value_counts(info.group)[:10]

nutrients = []
for rec in db:                          # 将各营养成分的  “ 列表 ”  转化为 DataFrame
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)
nutrients

nutrients.duplicated().sum()
nutrients = nutrients.drop_duplicates()

col_mapping = {'description' : 'food',
               'group'       : 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
info.head()
info

col_mapping = {'description' : 'nutrient',
               'group' : 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
nutrients

ndata = pd.merge(nutrients, info, on='id', how='outer')
ndata

ndata.ix[30000]

result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].order().plot(kind='barh')

by_nutrient = ndata.groupby(['nutgroup', 'nutrient'])

get_maximum = lambda x: x.xs(x.value.idxmax())
get_minimum = lambda x: x.xs(x.value.idxmin())

max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]

# make the food a little smaller
max_foods.food = max_foods.food.str[:50]

max_foods.ix['Amino Acids']['food']

###################################################################################
unit8
Plotting and Visualization
matplotlib--SourceForge      #最佳项目，python的2D绘图库
mayavi2--python              #    的3D绘图库
sympy---python               #符号计算库
numpy 和 scipy --python      #数值计算库
#!cmd
from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
from sympy import symbols
from sympy.plotting import plot
np.random.seed(12345)

plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4)
plot(np.arange(10))
%matplotlib inline

%pwd
A brief matplotlib API primer

import matplotlib.pyplot as plt
Figures and Subplots

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

from numpy.random import randn
plt.plot(randn(50).cumsum(), 'k--')

_ = ax1.hist(randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))
plt.close('all')
plt.plot([10, 20, 30])
plt.xlabel('tiems')
plt.ylabel('numbers')
plt.show()
fig, axes = plt.subplots(2, 3)
axes

#--------------------------------------------------------------------------------
t = np.arange(0., 5., 0.2)
# red dashes, blue squares and green triangles
plt.figure()
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
    #--------
lines = plt.plot(x1, y1, x2, y2)
# use key<a href="http://www.it165.net/edu/ebg/" target="_blank" class="keylink">word</a> args
plt.setp(lines, color='r', linewidth=2.0)
# or MATLAB style string value pairs
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)

#--------------------------------------------------------------------

######## 横向图形
from matplotlib import pyplot as plt
from numpy import sin, exp,  absolute, pi, arange
from numpy.random import normal


def f(t):
    s1 = sin(2 * pi * t)
    e1 = exp(-t)
    return absolute((s1 * e1)) + .05


t = arange(0.0, 5.0, 0.1)
s = f(t)
nse = normal(0.0, 0.3, t.shape) * s

fig = plt.figure(figsize=(12, 6))
vax = fig.add_subplot(121)
hax = fig.add_subplot(122)

vax.plot(t, s + nse, 'b^')
vax.vlines(t, [0], s)
vax.set_xlabel('time (s)')
vax.set_title('Vertical lines demo')

hax.plot(s + nse, t, 'b^')
hax.hlines(t, [0], s, lw=2)
hax.set_xlabel('time (s)')
hax.set_title('Horizontal lines demo')
plt.show()
# 点状分布图
import numpy as np
import matplotlib.pyplot as plt

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
plt.scatter(x, y, s=area, alpha=0.5)        ##点状分布图
plt.show()
#----
x = np.arange(0,5,0.1)
 lines = plt.plot(x, np.sin(x), x, np.cos(x))
#---
x = np.arange(0., np.e, 0.01)
y1 = np.exp(-x)
y2 = np.log(x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1);
ax1.set_ylabel('Y values for exp(-x)');
ax2 = ax1.twinx()           # this is the important function
ax2.plot(x, y2, 'r');
ax2.set_xlim([0, np.e]);
ax2.set_ylabel('Y values for ln(x)');
ax2.set_xlabel('Same X for both exp(-x) and ln(x)');
plt.show()
#---
X1 = range(0, 50)
Y1 = [num**2 for num in X1] # y = x^2
X2 = [0, 1]
Y2 = [0, 1]  # y = x
Fig = plt.figure(figsize=(8,4))                      # Create a `figure' instance
Ax = Fig.add_subplot(111)               # Create a `axes' instance in the figure
Ax.plot(X1, Y1, X2, Y2)                 # Create a Line2D instance in the axes
Fig.show()
Fig.savefig("F:/test.pdf")
os.remove("F:/test.pdf")
#---
######
Adjusting the spacing around subplots

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=None)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)
Colors, markers, and line styles

plt.figure()

plt.plot(randn(30).cumsum(), 'ko--')

plt.close('all')

data = randn(30).cumsum()
plt.plot(data, 'k--', label='Default')
plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
plt.legend(loc='best'
           , labels, and legends)

Setting the title, axis labels, ticks, and ticklabels

fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum())

ticks = ax.set_xticks([0, 250, 500, 750, 1000])   #  设置横坐标
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                            rotation=30, fontsize='small')

ax.set_title('My first matplotlib plot')
ax.set_xlabel('Stages')

Adding legends
#在一个图中画多个数据
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(randn(1000).cumsum(), 'm', label='one')
ax.plot(randn(1000).cumsum(), 'r--', label='two')
ax.plot(randn(1000).cumsum(), 'y.', label='three')

ax.legend(loc='best')

Annotations and drawing on a subplot

from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data = pd.read_csv('ch08/spx.csv', index_col=0, parse_dates=True)
spx = data['SPX']

spx.plot(ax=ax, style='m-')

crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 50),
                xytext=(date, spx.asof(date) + 200),
                arrowprops=dict(facecolor='black'),
                horizontalalignment='left', verticalalignment='top')

# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in 2008-2009 financial crisis')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color='g', alpha=0.5)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)

Saving plots to file

fig

fig.savefig('figpath.svg')

fig.savefig('figpath.png', dpi=400, bbox_inches='tight')

from io import BytesIO
buffer = BytesIO()
plt.savefig(buffer)
plot_data = buffer.getvalue()

matplotlib configuration

plt.rc('figure', figsize=(10, 10))
Plotting functions in pandas
Line plots

plt.close('all')

s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()
df = DataFrame(np.random.randn(10, 4).cumsum(0),
               columns=['A', 'B', 'C', 'D'],
               index=np.arange(0, 100, 10))
df.plot()
Bar plots

fig, axes = plt.subplots(2, 1)
data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)
data.plot(kind='barh', ax=axes[1], color='k', alpha=0.7)

df = DataFrame(np.random.rand(6, 4),
               index=['one', 'two', 'three', 'four', 'five', 'six'],
               columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df
df.plot(kind='bar')

plt.figure()

df.plot(kind='barh', stacked=True, alpha=0.5)

tips = pd.read_csv('ch08/tips.csv')
party_counts = pd.crosstab(tips.day, tips.size)
party_counts
# Not many 1- and 6-person parties
party_counts = party_counts.ix[:, 2:5]
party_counts
# Normalize to sum to 1
party_pcts = party_counts.div(party_counts.sum(1).astype(float), axis=0)
party_pcts

party_pcts.plot(kind='bar', stacked=True)
Histograms and density plots

plt.figure()

tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips['tip_pct'].hist(bins=50)

plt.figure()

tips['tip_pct'].plot(kind='kde')

plt.figure()

comp1 = np.random.normal(0, 1, size=200)  # N(0, 1)
comp2 = np.random.normal(10, 2, size=200)  # N(10, 4)
values = Series(np.concatenate([comp1, comp2]))
values.hist(bins=100, alpha=0.3, color='k', normed=True)
values.plot(kind='kde', style='k--')

Scatter plots

macro = pd.read_csv('ch08/macrodata.csv')
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
trans_data[-5:]

plt.figure()

plt.scatter(trans_data['m1'], trans_data['unemp'])
plt.title('Changes in log %s vs. log %s' % ('m1', 'unemp'))

pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)
Plotting Maps: Visualizing Haiti Earthquake Crisis data

data = pd.read_csv('ch08/Haiti.csv')
data.info()

data[['INCIDENT DATE', 'LATITUDE', 'LONGITUDE']][:10]

data['CATEGORY'][:6]

data.describe()

data = data[(data.LATITUDE > 18) & (data.LATITUDE < 20) &
            (data.LONGITUDE > -75) & (data.LONGITUDE < -70)
            & data.CATEGORY.notnull()]

def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]

def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))

def get_english(cat):
    code, names = cat.split('.')
    if '|' in names:
        names = names.split(' | ')[1]
    return code, names.strip()

get_english('2. Urgences logistiques | Vital Lines')

all_cats = get_all_categories(data.CATEGORY)
# Generator expression
english_mapping = dict(get_english(x) for x in all_cats)
english_mapping['2a']
english_mapping['6c']

def get_code(seq):
    return [x.split('.')[0] for x in seq if x]

all_codes = get_code(all_cats)
code_index = pd.Index(np.unique(all_codes))
dummy_frame = DataFrame(np.zeros((len(data), len(code_index))),
                        index=data.index, columns=code_index)

dummy_frame.ix[:, :6].info()

for row, cat in zip(data.index, data.CATEGORY):
    codes = get_code(to_cat_list(cat))
    dummy_frame.
    [row, codes] = 1

data = data.join(dummy_frame.add_prefix('category_'))

data.ix[:, 10:15].info()

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def basic_haiti_map(ax=None, lllat=17.25, urlat=20.25,
                    lllon=-75, urlon=-71):
    # create polar stereographic Basemap instance.
    m = Basemap(ax=ax, projection='stere',
                lon_0=(urlon + lllon) / 2,
                lat_0=(urlat + lllat) / 2,
                llcrnrlat=lllat, urcrnrlat=urlat,
                llcrnrlon=lllon, urcrnrlon=urlon,
                resolution='f')
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    return m

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

to_plot = ['2a', '1', '3c', '7a']

lllat=17.25; urlat=20.25; lllon=-75; urlon=-71

for code, ax in zip(to_plot, axes.flat):
    m = basic_haiti_map(ax, lllat=lllat, urlat=urlat,
                        lllon=lllon, urlon=urlon)

    cat_data = data[data['category_%s' % code] == 1]

    # compute map proj coordinates.
    x, y = m(cat_data.LONGITUDE.values, cat_data.LATITUDE.values)

    m.plot(x, y, 'k.', alpha=0.5)
    ax.set_title('%s: %s' % (code, english_mapping[code]))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

to_plot = ['2a', '1', '3c', '7a']

lllat=17.25; urlat=20.25; lllon=-75; urlon=-71

def make_plot():

    for i, code in enumerate(to_plot):
        cat_data = data[data['category_%s' % code] == 1]
        lons, lats = cat_data.LONGITUDE, cat_data.LATITUDE

        ax = axes.flat[i]
        m = basic_haiti_map(ax, lllat=lllat, urlat=urlat,
                            lllon=lllon, urlon=urlon)

        # compute map proj coordinates.
        x, y = m(lons.values, lats.values)

        m.plot(x, y, 'k.', alpha=0.5)
        ax.set_title('%s: %s' % (code, english_mapping[code]))


make_plot()

shapefile_path = 'ch08/PortAuPrince_Roads/PortAuPrince_Roads'
m.readshapefile(shapefile_path, 'roads')
plt.close('all')
#############################################################

unit9
Data Aggregation and Group Operations

from __future__ import division
from numpy.random import randn
import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4)

pd.options.display.notebook_repr_html = False

%matplotlib inline
GroupBy mechanics

df = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                'key2' : ['one', 'two', 'one', 'two', 'one'],
                'data1' : np.random.randn(5),
                'data2' : np.random.randn(5)})
df

grouped = df['data1'].groupby(df['key1'])
grouped
grouped.mean()

means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means
means.unstack ()

states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()

df.groupby('key1').mean()   #  key2 消失了

df.groupby(['key1', 'key2']).mean()

df.groupby(['key1', 'key2']).size()

Iterating over groups

for name, group in df.groupby('key1'):
    #name =a; group = b
    print(name)
    print(group)

for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2))
    print(group)

for (k1 , k2) in df.groupby (['key1','key2' ]):
    print((k1 , k2))

for group in df.groupby ([ 'key1' , 'key2' ]):
    print(group)
pieces = dict(list(df.groupby('key1')))
pieces['b']

df.dtypes

grouped = df.groupby(df.dtypes, axis=1)
dict(list(grouped))

Selecting a column or subset of columns

df.groupby('key1')['data1']
df.groupby('key1')[['data2']]
df['data1'].groupby(df['key1'])
df[['data2']].groupby(df['key1'])

df.groupby(['key1', 'key2'])[['data2']].mean()

s_grouped = df.groupby(['key1', 'key2'])['data2']
s_grouped
s_grouped.mean()

Grouping with dicts and Series

people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.ix[2:3, ['b', 'c']] = np.nan # Add a few NA values
people

mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f' : 'orange'}

by_column = people.groupby(mapping)
by_column.sum()
by_column = people.groupby(mapping, axis=1)  # 多列合并
by_column.sum()

map_series = Series(mapping)
map_series

people.groupby(map_series, axis=1).count()

Grouping with functions

people.groupby(len).sum()  # index 的方向上的 字符串的 ‘长度’ 聚合

key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()
Grouping by index levels

columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
columns
hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
hier_df

hier_df.groupby(level='cty', axis=1).count()

Data aggregation
df
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)

def peak_to_peak(arr):
    return arr.max() - arr.min()

grouped.agg(peak_to_peak)
grouped.agg(peak_to_peak)
grouped.describe()

tips = pd.read_csv('ch08/tips.csv')
# Add tip percentage of total bill
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips[:6]

Column-wise and multiple function application

grouped = tips.groupby(['sex', 'smoker'])  #  标题

grouped_pct = grouped['tip_pct']           #  数值 即： 计算列
grouped_pct.agg('mean')
grouped_pct.agg ([  peak_to_peak ])
grouped_pct.agg(['mean', 'std', peak_to_peak])

grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])

functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
result

result['tip_pct']

ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
#分别重命名两个函数名
grouped['tip_pct', 'total_bill'].agg(ftuples)
#对不同的列用不同的函数
grouped.agg({'tip' : np.max, 'size' : 'sum'})

grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'],
             'size' : 'sum'})

Returning aggregated data in "unindexed" form

tips.groupby(['sex', 'smoker'], as_index=False).mean()
# 不需要传入层次化时

Group-wise operations and transformations
df
k1_means = df.groupby('key1').mean().add_prefix('mean_')
k1_means

pd.merge(df, k1_means, left_on='key1', right_index=True)

key = ['one', 'two', 'one', 'two', 'one']

people.groupby(key).mean() ＃　　按　ｋｅｙ　聚合并算出均值

people.groupby(key).transform(np.mean)　
　在原位置上填上相应的的均值

def demean(arr):
    return arr - arr.mean()  #  距平化函数然后传给　ｔｒａｎｓｆｏｒｍ
demeaned = people.groupby(key).transform(demean)
demeaned

demeaned.groupby(key).mean()
Apply: General split-apply-combine
apply()  的应用
def top(df, n=5, column='tip_pct'):
    return df.sort_index(by=column)[-n:]
df
def top(df,n=5,c_olm_n='tip_pct'):
    return df.sort_index(by =c_olm_n,ascending = False)[:n]





tips.head()
top(tips, n=6)
tips.groupby('smoker').apply(top)

tips.groupby(['smoker', 'day']).apply(top, n=1, c_olm_n='total_bill')

result = tips.groupby('smoker')['tip_pct'].describe()
result

result.unstack('smoker')
f = lambda x: x.describe()
grouped.apply(f)

Suppressing the group keys

tips.groupby('smoker').apply(top)
tips.groupby('smoker', group_keys=False).apply(top)  # groups -key = False
#  禁用层次化索引

Quantile and bucket analysis


frame = DataFrame ({
                       'data1': np.random.randn (1000) ,
                       'data2': np.random.randn (1000) })
frame.head()

factor = pd.cut(frame.data1, 4)

factor[:10]

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

grouped = frame.data2.groupby(factor)
grouped.apply(get_stats).unstack()
grouped.apply(get_stats).unstack().stack()
grouped.agg('mean')
df.groupby('key1').mean()
df.groupby(['key1', 'key2'])[['data2']].mean()
people.groupby(key).mean()
tips.groupby('smoker').apply(top)

#ADAPT the output is not sorted in the book while this is the case now (swap first two lines)

# Return quantile numbers

grouping = pd.qcut(frame.data1, 10, labels=False)
grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()
grouped
Example: Filling missing values with group-specific values

s = Series(np.random.randn(6))
s[:]
s[::]
s[::3] = np.nan
s[::2] = np.nan
s

s.fillna(s.mean())

states = ['Ohio', 'New York', 'Vermont', 'Florida',
          'Oregon', 'Nevada', 'California', 'Idaho']
states
group_key = ['East'] * 4 + ['West'] * 4
group_key
data = Series(np.random.randn(8), index=states)
data
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
data

data.groupby(group_key).mean()

fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name])#~~
data.groupby(group_key).apply(fill_func)

Example: Random sampling and permutation

# Hearts, Spades, Clubs, Diamonds

suits = ['H', 'S', 'C', 'D']
card_val = (range(1, 11) + [10] * 3) * 4
card_val[:4]
base_names = ['A'] + range(2, 11) + ['J', 'K', 'Q']
base_names
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)
cards[:3]
deck = Series(card_val, index=cards)
deck.take([51,6])
len(deck)
np.random.permutation(52)
def draw(deck, n=5):
    return deck.take(np.random.permutation(len(deck))[:n])

draw(deck)
apply()  的拆分合并  “应用”
get_suit = lambda card: card[-1] # last letter is suit
deck.groupby(get_suit).apply(draw, n=2)

# alternatively
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)

Example: Group weighted average and correlation
分组加权平均数和相关系数
 np.random.permutation(n)  #  n 为大数据集

df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'data': np.random.randn(8),
                'weights': np.random.rand(8)})
df

category()  # 计算加权平均数

grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
grouped.apply(get_wavg)   #  运用加权平均数

close_px = pd.read_csv('ch09/stock_px.csv', parse_dates=True, index_col=0)
close_px.info()
close_px.head()

close_px[-4:]

rets = close_px.pct_change().dropna()
spx_corr = lambda x: x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x: x.year)   #  年收益率
by_year                                    #  按年分组
by_year.apply(spx_corr)

# Annual correlation of Apple with Microsoft
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))   #  计算相关系数

Example: Group-wise linear regression
面向分组的线性回归

import statsmodels.api as sm
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()  #  最小二乘，计算线性回归
    return result.params

by_year.apply(regress, 'AAPL', ['SPX'])
Pivot tables and Cross-tabulation

tips.pivot_table(index=['sex', 'smoker'])
tips.pivot_table(rows=['sex', 'smoker'])

tips.pivot_table(['tip_pct', 'size'], rows=['sex', 'day'],
                 cols='smoker')

tips.pivot_table(['tip_pct', 'size'], rows=['sex', 'day'],
                 cols='smoker', margins=True)

tips.pivot_table('tip_pct', rows=['sex', 'smoker'], cols='day',aggfunc=len, margins=True)

tips.pivot_table('size', rows=['time', 'sex', 'smoker'],
                 cols='day', aggfunc='sum', fill_value=0)
Cross-tabulations: crosstab

from StringIO import StringIO
data = """\
Sample    Gender    Handedness
1    Female    Right-handed
2    Male    Left-handed
3    Female    Right-handed
4    Male    Right-handed
5    Male    Left-handed
6    Male    Right-handed
7    Female    Right-handed
8    Female    Left-handed
9    Male    Right-handed
10    Female    Right-handed"""
data = pd.read_table(StringIO(data), sep='\s+')

data

pd.crosstab(data.Gender, data.Handedness, margins=True)

pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)
Example: 2012 Federal Election Commission Database

fec = pd.read_csv('ch09/P00000001-ALL.csv')

fec.info()

fec.ix[123456]

unique_cands = fec.cand_nm.unique()
unique_cands

unique_cands[2]

parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}

fec.cand_nm[123456:123461]

fec.cand_nm[123456:123461].map(parties)

# Add it as a column
fec['party'] = fec.cand_nm.map(parties)

fec['party'].value_counts()

(fec.contb_receipt_amt > 0).value_counts()

fec = fec[fec.contb_receipt_amt > 0]

fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'][:10]

Donation statistics by occupation and employer

fec.contbr_occupation.value_counts()[:10]

occ_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
   'C.E.O.': 'CEO'
}

# If no mapping provided, return x
f = lambda x: occ_mapping.get(x, x)  #  巧妙地运用了 dict.get()
fec.contbr_occupation = fec.contbr_occupation.map(f)
dict.get()
emp_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'SELF' : 'SELF-EMPLOYED',
   'SELF EMPLOYED' : 'SELF-EMPLOYED',
}

# If no mapping provided, return x
f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)

by_occupation = fec.pivot_table('contb_receipt_amt',
                                rows='contbr_occupation',
                                cols='party', aggfunc='sum')

over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
over_2mm

over_2mm.plot(kind='barh')

def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()

    # Order totals by key in descending order
    return totals.order(ascending=False)[-n:]

grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n=7)

grouped.apply(get_top_amounts, 'contbr_employer', n=10)
Bucketing donation amounts

bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
labels

grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack(0)

bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
bucket_sums

normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
#  这一步主要作用是什么：进行科学计数法统一  ~~
normed_sums

normed_sums[:-2].plot(kind='barh', stacked=True)

Donation statistics by state

grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
totals[:10]

percent = totals.div(totals.sum(1), axis=0)
percent[:10]
##########################################################3
unit10
Time series

from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np

pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 4))

%matplotlib inline

Date and Time Data Types and Tools

from datetime import datetime
now = datetime.now()
now

now.year, now.month, now.day

delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta

delta.days

delta.seconds

from datetime import timedelta
start = datetime(2011, 1, 7)
start + timedelta(12)
start - 2 * timedelta(12)

Converting between string and datetime

stamp = datetime(2011, 1, 3)
type(stamp)
str(stamp)

stamp.strftime('%Y-%m-%d')
stamp.strftime('%Y%m%d')
value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')

datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]

from dateutil.parser import parse
parse('2011-01-03')

parse('Jan 31, 1997 10:45 PM')

parse('6/12/2011', dayfirst=True)

datestrs

pd.to_datetime(datestrs)
# note: output changed (no '00:00:00' anymore)

idx = pd.to_datetime(datestrs + [None])
idx

idx[2]

pd.isnull(idx)
Time Series Basics

from datetime import datetime
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
         datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
ts
type(ts)
# note: output changed to "pandas.core.series.Series"
ts.index
ts + ts[::2]  #  按时间对齐

ts.index.dtype  #  dtype('<M8[ns]')  纳秒时间对齐
# note: output changed from dtype('datetime64[ns]') to dtype('<M8[ns]')

stamp = ts.index[0]
stamp
# note: output changed from <Timestamp: 2011-01-02 00:00:00> to Timestamp('2011-01-02 00:00:00')
Indexing, selection, subsetting

stamp = ts.index[2]
ts[stamp]

ts['1/10/2011']
ts['20110110']

longer_ts = Series(np.random.randn(1000),
                   index=pd.date_range('1/1/2000', periods=1000))
longer_ts
longer_ts['2001']

longer_ts['2001-05']

ts[datetime(2011, 1, 7):]
ts
ts['1/6/2011':'1/11/2011']
ts.truncate(after='1/9/2011')
ts.truncate(before='1/9/2011')

dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = DataFrame(np.random.randn(100, 4),
                    index=pd.date_range('1/1/2000', periods=100, freq='W-WED'),
                    columns=['Colorado', 'Texas', 'New York', 'Ohio'])

long_df.ix['5-2001']  # 为什么要用ix[]
long_df['5-2001']
long_df['5-2001']

Time series with duplicate indices

dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
                          '1/3/2000'])
dup_ts = Series(np.arange(5), index=dates)
dup_ts
dup_ts.index.is_unique

dup_ts['1/3/2000']  # not duplicated
dup_ts['1/2/2000']  # duplicated

grouped = dup_ts.groupby(level=0)
grouped.mean()
grouped.count()

Date ranges, Frequencies, and Shifting

ts
ts.resample('D')   #  按天排列

Generating date ranges
index = pd.date_range('4/1/2012', '6/1/2012')
index

pd.date_range(start='4/1/2012', periods=20)

pd.date_range(end='6/1/2012', periods=20)

pd.date_range('1/1/2000', '12/1/2000', freq='BM')
  #  传入 'BM'  频率
pd.date_range('5/2/2012 12:56:31', periods=7)

pd.date_range('5/2/2012 12:56:31', periods=5, normalize=True)
# 被规范化到  午夜 的时间戳

Frequencies and Date Offsets

from pandas.tseries.offsets import Hour, Minute
hour = Hour()
hour
Hour(2)
four_hours = Hour(4)
four_hours

pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')

Hour(2) + Minute(30)
pd.date_range('1/1/2000', periods=10, freq='1h30min')

Week of month dates

rng = pd.date_range('1/1/2012', '9/1/2012', freq='WOM-3FRI')
'WOM-3FRI'  每个月第三个星期五
list(rng)
Shifting (leading and lagging) data

ts = Series(np.random.randn(4),
         index = pd.date_range ('1/1/2000', periods = 4 , freq = 'M'))
ts = Series(np.random.randn(4),
            index = pd.date_range('1/1/2000',periods = 4,freq = 'M'))
long_df = DataFrame (np.random.randn (100 , 4) ,
                     index = pd.date_range ('1/1/2000' , periods = 100 , freq = 'W-WED') ,
                     columns = [ 'Colorado' , 'Texas' , 'New York' , 'Ohio' ])
ts

ts.shift(2)
ts.shift(1)
ts.shift(-2)
ts / ts.shift(1) - 1 #  较前一日期的百分比变化

ts.shift(2, freq='M')   #  按月份衣移动 向前两个单元
ts.shift(3, freq='D')
ts.shift(1, freq='3D')
ts.shift(1, freq='90T')  # 往前推 90 分钟

Shifting dates with offsets

from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2000,11,7)
now + 3 * Day()
now + MonthEnd()
now + MonthEnd(2)

offset = MonthEnd()
MonthEnd().rollforward(now)

offset.rollforward(now)  # 当前一个月的月底。过去的一个月的月底
offset.rollback(now)

ts = Series(np.random.randn(20),
            index=pd.date_range('1/15/2000', periods=20, freq='4d'))
ts.groupby(offset.rollforward).mean()
ts.resample('M', how='mean')
ts.resample('D')
ts.resample('M',how = 'mean')
Time Zone Handling

import pytz
pytz.common_timezones[-5:]

tz = pytz.timezone('US/Eastern')
tz

Localization and Conversion

rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
list(rng)
ts = Series(np.random.randn(len(rng)), index=rng)
  ts
print(ts.index.tz)

pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')

ts_utc = ts.tz_localize('UTC')
ts_utc

ts_utc.index

ts_utc.tz_convert('US/Eastern')

ts_eastern = ts.tz_localize('US/Eastern')
ts_eastern.tz_convert('UTC')
ts_eastern.tz_convert('Europe/Berlin')
ts.index.tz_localize('Asia/Shanghai')

Operations with time zone-aware Timestamp objects

stamp = pd.Timestamp('2011-03-12 04:00')
stamp_utc = stamp.tz_localize('utc')
tz_localize()
stamp_utc.tz_convert('US/Eastern')

stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
stamp_moscow
stamp_utc.value
stamp_utc.tz_convert('US/Eastern').value

# 30 minutes before DST transition
from pandas.tseries.offsets import Hour
stamp = pd.Timestamp('2012-03-12 01:30', tz='US/Eastern')
stamp

stamp + Hour()

# 90 minutes before DST transition
# 夏令时转变前90 分钟
stamp = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern')
stamp

stamp + 2 * Hour()

Operations between different time zones

rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
ts = Series(np.random.randn(len(rng)), index=rng)
ts

ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
ts1   ;   ts2
result = ts1 + ts2
result
result.index

Periods and Period Arithmetic
#                                period  表示时间区间
p = pd.Period(2007, freq='A-DEC')
p
p + 5
p - 2
pd.Period('2014', freq='A-DEC') - p
rng = pd.period_range('1/1/2000', '6/30/2000', freq='M')
list(rng)

Series(np.random.randn(6), index=rng)

values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC')
index

Period Frequency Conversion
  时间的频率转换  asfreq()
p = pd.Period('2007', freq='A-DEC')
p.asfreq('M', how='start')
p.asfreq('M', how='end')
高频率转化为低频率
p = pd.Period('2007', freq='A-JUN')
p.asfreq('M', 'start')              # 往前推 12 个月
p.asfreq('M', 'end')

p = pd.Period('Aug-2007', 'M')
p.asfreq('A-JUN')

rng = pd.period_range('2006', '2009', freq='A-DEC')
ts = Series(np.random.randn(len(rng)), index=rng)
ts

ts.asfreq('M', how='start')
ts.asfreq('B', how='end')  #  具体到日期

Quarterly period frequencies
按季度计算的时期频率
p = pd.Period('2012Q4', freq='Q-JAN')
p

p.asfreq('D', 'start')
p.asfreq('D', 'end')
该季度倒数第二个工作日下午4点的时间戳
p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
p4pm
p4pm.to_timestamp()

rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
ts = Series(np.arange(len(rng)), index=rng)
ts

new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
ts.index = new_rng.to_timestamp()
ts

Converting Timestamps to Periods (and back)

rng = pd.date_range('1/1/2000', periods=3, freq='M')
list(rng)
时间戳索引转换为时期索引

ts = Series(randn(3), index=rng)
ts
pts = ts.to_period()
pts

rng = pd.date_range('1/29/2000', periods=6, freq='D')
list(rng)
ts2 = Series(randn(6), index=rng)
ts2
ts2.to_period('M')

pts = ts.to_period()
pts
 要转换为时间戳用  to_timestamp()
pts.to_timestamp(how='end')

Creating a PeriodIndex from arrays

data = pd.read_csv('ch08/macrodata.csv')
data.describe()
data.head()
data.year
data.quarter

index = pd.PeriodIndex(year=data.year,
                       quarter=data.quarter, freq='Q-DEC')
index

data.index = index
data.infl
Resampling and Frequency Conversion
重采样/降采样/升采样

rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(randn(len(rng)), index=rng)
ts.describe()
ts.resample('M', how='mean')
ts.resample('M', how='mean', kind='period')
Downsampling

rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = Series(np.arange(12), index=rng)
ts
ts.resample('5min', how='sum')
# note: output changed (as the default changed from closed='right', label='right' to closed='left', label='left'

ts.resample('5min', how='sum', closed='left')
ts.resample('5min', how='sum', closed='left', label='left')

ts.resample('5min', how='sum', loffset='-1s')
Open-High-Low-Close (OHLC) resampling
 OHLC 重采样
ts.resample('5min', how='ohlc')
# note: output changed because of changed defaults
Resampling with GroupBy

rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(np.arange(100), index=rng)
ts.groupby(lambda x: x.month).mean()
ts.groupby(lambda x: x.weekday).mean()

Upsampling and interpolation

frame = DataFrame(np.random.randn(2, 4),
                  index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
                  columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame

df_daily = frame.resample('D')
df_daily

frame.resample('D', fill_method='ffill')

frame.resample('D', fill_method='ffill', limit=2)

frame.resample('W-THU', fill_method='ffill')

Resampling with periods
  通过时期重采样
frame = DataFrame(np.random.randn(24, 4),
                  index=pd.period_range('1-2000', '12-2001', freq='M'),
                  columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame[:5]

annual_frame = frame.resample('A-DEC', how='mean')
annual_frame

# Q-DEC: Quarterly, year ending in December
annual_frame.resample('Q-DEC', fill_method='ffill')
# note: output changed, default value changed from convention='end' to convention='start' + 'start' changed to span-like
# also the following cells

annual_frame.resample('Q-DEC', fill_method='ffill', convention='start')

annual_frame.resample('Q-MAR', fill_method='ffill')

Time series plotting

close_px_all = pd.read_csv('ch09/stock_px.csv', parse_dates=True, index_col=0)
close_px_all.info()
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B', fill_method='ffill')
close_px.info()

close_px['AAPL'].plot()

close_px.ix['2009'].plot()

close_px['AAPL'].ix['01-2011':'03-2011'].plot()
季度频率
appl_q = close_px['AAPL'].resample('Q-DEC', fill_method='ffill')
appl_q.ix['2009':].plot()

Moving window functions

close_px = close_px.asfreq('B').fillna(method='ffill')

close_px.AAPL.plot()
pd.rolling_mean(close_px.AAPL, 250).plot()

plt.figure()
appl_std250 = pd.rolling_std(close_px.AAPL, 250, min_periods=10)
appl_std250[5:12]
appl_std250.plot()

# Define expanding mean in terms of rolling_mean
expanding_mean = lambda x: rolling_mean(x, len(x), min_periods=1)

pd.rolling_mean(close_px, 60).plot(logy=True)
plt.close('all')

Exponentially-weighted functions

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True,
                         figsize=(12, 7))

aapl_px = close_px.AAPL['2005':'2009']

ma60 = pd.rolling_mean(aapl_px, 60, min_periods=50)
ewma60 = pd.ewma(aapl_px, span=60)

aapl_px.plot(style='k-', ax=axes[0])
ma60.plot(style='k--', ax=axes[0])
aapl_px.plot(style='k-', ax=axes[1])
ewma60.plot(style='k--', ax=axes[1])
axes[0].set_title('Simple MA')
axes[1].set_title('Exponentially-weighted MA')

Binary moving window functions

close_px
spx_px = close_px_all['SPX']

spx_rets = spx_px / spx_px.shift(1) - 1
returns = close_px.pct_change()
corr = pd.rolling_corr(returns.AAPL, spx_rets, 125, min_periods=100)
corr.plot()
# 所有的相关性
corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100)
corr.plot()

User-defined moving window functions

from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
result = pd.rolling_apply(returns.AAPL, 250, score_at_2percent)
result.plot()
Performance and Memory Usage Notes

rng = pd.date_range('1/1/2000', periods=10000000, freq='10ms')
ts = Series(np.random.randn(len(rng)), index=rng)
ts

ts.resample('15min', how='ohlc').info()

%timeit ts.resample('15min', how='ohlc')

rng = pd.date_range('1/1/2000', periods=10000000, freq='1s')
ts = Series(np.random.randn(len(rng)), index=rng)
 tss = ts.resample('15s', how='ohlc')
########################################
unit11
Financial and Economic Data Applications

from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 6))

%matplotlib inline

%pwd
Data munging topics

Time series and cross-section alignment

close_px = pd.read_csv('ch11/stock_px.csv', parse_dates=True, index_col=0)
volume = pd.read_csv('ch11/volume.csv', parse_dates=True, index_col=0)
prices = close_px.ix['2011-09-05':'2011-09-14', ['AAPL', 'JNJ', 'SPX', 'XOM']]
volume = volume.ix['2011-09-05':'2011-09-12', ['AAPL', 'JNJ', 'XOM']]
prices
volume

prices * volume

vwap = (prices * volume).sum() / volume.sum()
vwap
vwap.dropna()
vwap
prices.align(volume, join='inner')  #  align()   手工对齐

s1 = Series(range(3), index=['a', 'b', 'c'])
s2 = Series(range(4), index=['d', 'b', 'c', 'e'])
s3 = Series(range(3), index=['f', 'a', 'c'])
DataFrame({'one': s1, 'two': s2, 'three': s3})

DataFrame({'one': s1, 'two': s2, 'three': s3}, index=list('face'))

Operations with time series of different frequencies

ts1 = Series(np.random.randn(3),
             index=pd.date_range('2012-6-13', periods=3, freq='W-WED'))
#  频率为周
ts1
ts1.resample('B')   # 重采样到工作日
ts1.resample('B', fill_method='ffill')

dates = pd.DatetimeIndex(['2012-6-12', '2012-6-17', '2012-6-18',
                          '2012-6-21', '2012-6-22', '2012-6-29'])
ts2 = Series(np.random.randn(6), index=dates)
ts2
ts1
ts1.reindex(ts2.index, method='ffill')

ts2 + ts1.reindex(ts2.index, method='ffill')

Using periods instead of timestamps
使用  periods()
gdp = Series([1.78, 1.94, 2.08, 2.01, 2.15, 2.31, 2.46],
             index=pd.period_range('1984Q2', periods=7, freq='Q-SEP'))
infl = Series([0.025, 0.045, 0.037, 0.04],
              index=pd.period_range('1982', periods=4, freq='A-DEC'))
gdp
infl

infl_q = infl.asfreq('Q-SEP', how='end')
infl_q
infl_q.reindex(gdp.index, method='ffill')

Time of day and "as of" data selection

# Make an intraday date range and time series               # 一天内的
rng = pd.date_range('2012-06-01 09:30', '2012-06-01 15:59', freq='T')
# Make a 5-day series of 9:30-15:59 values
rng = rng.append([rng + pd.offsets.BDay() for i in range(1, 4)])
ts = Series(np.arange(len(rng), dtype=float), index=rng)
ts
ts[:10]
from datetime import time
ts[time(10, 0)]

ts.at_time(time(10, 0))

ts.between_time(time(10, 0), time(10, 1))

np.random.seed(12346)
time.time()# 返回当前的时间戳
# Set most of the time series randomly to NA
# 时间序列的大部分数设置为   NA
indexer = np.sort(np.random.permutation(len(ts))[700:])
irr_ts = ts.copy()
irr_ts[indexer] = np.nan
irr_ts['2012-06-01 09:50':'2012-06-01 10:00']

selection = pd.date_range('2012-06-01 10:00', periods=4, freq='B')
list(selection)
irr_ts.asof(selection)  #  传入  selection

Splicing together data sources                          #  拼接

data1 = DataFrame(np.ones((6, 3), dtype=float),
                  columns=['a', 'b', 'c'],
                  index=pd.date_range('6/12/2012', periods=6))
data2 = DataFrame(np.ones((6, 3), dtype=float) * 2,
                  columns=['a', 'b', 'c'],
                  index=pd.date_range('6/13/2012', periods=6))
data1
data2
拼接两个数据源
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
spliced


data2 = DataFrame (np.ones ((6 , 4) , dtype = float) * 2 , columns = [ 'a' , 'b' , 'c' , 'd' ] ,
                   index = pd.date_range ('6/13/2012' , periods = 6))
data2
combiae_first()  引入合并点之前的数据
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
spliced                                                 # n, 叠接

spliced_filled = spliced.combine_first(data2)
spliced_filled

spliced.update(data2, overwrite=False)
spliced
直接对列进行设置
cp_spliced = spliced.copy()
cp_spliced[['a', 'c']] = data1[['a', 'c']]
cp_spliced

Return indexes and cumulative returns                           #累加的
收益指数和累计收益
import pandas.io.data as web
price = web.get_data_yahoo('AAPL', '2011-01-01')['Adj Close']
price[-5:]

price['2011-10-03'] / price['2011-3-01'] - 1

returns = price.pct_change()
type(returns)
returns.describe()
returns[:3]
利用cumprod()  计算简单的收益率指数
ret_index = (1 + returns).cumprod()
ret_index[0] = 1  # Set first value to 1
ret_index

m_returns = ret_index.resample('BM', how='last').pct_change()
m_returns['2012']
重采样聚合也可以得到同样的结果
m_rets = (1 + returns).resample('M', how='prod', kind='period') - 1
m_rets['2012']
returns[dividend_dates] += dividend_pcts
Group transforms and analysis

pd.options.display.max_rows = 100
pd.options.display.max_columns = 10
np.random.seed(12345)

import random; random.seed(0)
import string

N = 1000
def rands(n):
    choices = string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in xrange(n)]) #~~
tickers = np.array([rands(5) for _ in xrange(N)])
tickers[:5]
M = 500
df = DataFrame({'Momentum' : np.random.randn(M) / 200 + 0.03,
                'Value' : np.random.randn(M) / 200 + 0.08,
                'ShortInterest' : np.random.randn(M) / 200 - 0.02},
                index=tickers[:M])
df.head()
ind_names = np.array(['FINANCIAL', 'TECH'])
sampler = np.random.randint(0, len(ind_names), N)
sampler[:5]
industries = Series(ind_names[sampler], index=tickers,
                    name='industry')
industries.head()
by_industry = df.groupby(industries)

by_industry.mean()
by_industry.describe()

# Within-Industry Standardize 行业标准化处理
def zscore(group):                                  #  核心
    return (group - group.mean()) / group.std()

apply()
df_stand = by_industry.apply(zscore)
df_stand[:5]

df_stand.groupby(industries).agg(['mean', 'std'])

# Within-industry rank descending  行业内降序排名
ind_rank = by_industry.rank(ascending=False)
ind_rank.groupby(industries).agg(['min', 'max'])

# Industry rank and standardize  行业内排名和标准化
by_industry.apply(lambda x: zscore(x.rank()))
风险因子暴露
Group factor exposures
factor analysis

from numpy.random import rand
fac1, fac2, fac3 = np.random.rand(3, 1000)
tickers[:6]
ticker_subset = tickers.take(np.random.permutation(N)[:1000])
#   permutation()  排列 ；随机序列     | subset : 子集
# Weighted sum of factors plus noise  因子加权以及噪声
port = Series(0.7 * fac1 - 1.2 * fac2 + 0.3 * fac3 + rand(1000),  #随机绕动
              index=ticker_subset)
port[:6]
factors = DataFrame({'f1': fac1, 'f2': fac2, 'f3': fac3},
                    index=ticker_subset)
factors[:6]
type(factors)  # pandas.core.frame.DataFrame
type(port)     #  pandas.core.series.Series
factors.corrwith(port)
#  port.corrwith(factors   #'Series' object has no attribute 'corrwith'
pd.ols(y=port, x=factors).beta
# 某个股票的价格与某个基准（比如：标普500）的协整性被 称作 贝塔风险系数

def beta_exposure(chunk, factors=None):                              #暴露
    return pd.ols(y=chunk, x=factors).beta
industries[:4]
by_ind = port.groupby(industries)
exposures = by_ind.apply(beta_exposure, factors=factors)
exposures.unstack()                                  # stack ,unstack | 堆积  出栈

Decile and quartile analysis

import pandas.io.data as web
data = web.get_data_yahoo('SPY', '2010-01-01')  # 加上截止日期
data.info()

px = data['Adj Close']
type(px)  # pandas.core.series.Series # 'Series' object has no attribute 'info'

returns = px.pct_change()
returns[:6]; returns[-6:]
returns

def to_index(rets):
    index = (1 + rets).cumprod()
    first_loc = max(index.index.get_loc(index.idxmax()) - 1, 0)
    index.values[first_loc] = 1
    return index

def trend_signal (rets , lookback , lag):
    signal = pd.rolling_sum (rets , lookback , min_periods = lookback - 5)
    return signal.shift (lag)

signal = trend_signal(returns, 100, 3)
trade_friday = signal.resample('W-FRI').resample('B', fill_method='ffill')
#  每周五动量交易
trade_rets = trade_friday.shift(1) * returns
trade_rets = trade_rets[:len(returns)]

to_index(trade_rets).plot()

# 夏普比率
vol = pd.rolling_std(returns, 250, min_periods=200) * np.sqrt(250)

def sharpe(rets, ann=250):
    return rets.mean() / rets.std()  * np.sqrt(ann)
cats = pd.qcut(vol, 4)
print('cats: %d, trade_rets: %d, vol: %d' % (len(cats), len(trade_rets), len(vol)))

trade_rets.groupby(cats).agg(sharpe)
More example applications
信号前沿分析
Signal frontier analysis

names = ['AAPL', 'GOOG', 'MSFT', 'DELL', 'GS', 'MS', 'BAC', 'C']
def get_px(stock, start, end):
    return web.get_data_yahoo(stock, start, end)['Adj Close']
px = DataFrame({n: get_px(n, None, None) for n in names})

#px = pd.read_csv('ch11/stock_px.csv')

plt.close('all')

px = px.asfreq('B').fillna(method='pad')
rets = px.pct_change()
((1 + rets).cumprod() - 1).plot()  #  累计收益

计算特定回顾期的动量
def calc_mom(price, lookback, lag):
    mom_ret = price.shift(lag).pct_change(lookback)
    ranks = mom_ret.rank(axis=1, ascending=False)
    demeaned = ranks.subtract(ranks.mean(axis=1), axis=0)
    return demeaned.divide(demeaned.std(axis=1), axis=0)

compound = lambda x : (1 + x).prod() - 1
daily_sr = lambda x: x.mean() / x.std()

def strat_sr(prices, lb, hold):
    # Compute portfolio weights  计算投资组合权重
    freq = '%dB' % hold
    port = calc_mom(prices, lb, lag=1)

    daily_rets = prices.pct_change()

    # Compute portfolio returns  计算投资组合收益
    port = port.shift(1).resample(freq, how='first')
    returns = daily_rets.resample(freq, how=compound)
    port_rets = (port * returns).sum(axis=1)

    return daily_sr(port_rets) * np.sqrt(252 / hold)

strat_sr(px, 70, 30)

from collections import defaultdict

lookbacks = range(20, 90, 5)
holdings = range(20, 90, 5)
dd = defaultdict(dict)
for lb in lookbacks:
    for hold in holdings:
        dd[lb][hold] = strat_sr(px, lb, hold)

ddf = DataFrame(dd)
ddf.index.name = 'Holding Period'
ddf.columns.name = 'Lookback Period'

import matplotlib.pyplot as plt

def heatmap(df, cmap=plt.cm.gray_r):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axim = ax.imshow(df.values, cmap=cmap, interpolation='nearest')
    ax.set_xlabel(df.columns.name)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(list(df.columns))
    ax.set_ylabel(df.index.name)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(list(df.index))
    plt.colorbar(axim)

heatmap(ddf)

Future contract rolling

pd.options.display.max_rows = 10

import pandas.io.data as web
# Approximate price of S&P 500 index
px = web.get_data_yahoo('SPY')['Adj Close'] * 10
px
px.plot()
from datetime import datetime
expiry = {'ESU2': datetime(2012, 9, 21),
          'ESZ2': datetime(2012, 12, 21)}
expiry = Series(expiry).order()

expiry

np.random.seed(12347)
N = 200
walk = (np.random.randint(0, 200, size=N) - 100) * 0.25
perturb = (np.random.randint(0, 20, size=N) - 10) * 0.25
walk = walk.cumsum()

rng = pd.date_range(px.index[0], periods=len(px) + N, freq='B')
near = np.concatenate([px.values, px.values[-1] + walk])
far = np.concatenate([px.values, px.values[-1] + walk + perturb])
prices = DataFrame({'ESU2': near, 'ESZ2': far}, index=rng)

prices.tail()

def get_roll_weights(start, expiry, items, roll_periods=5):
    # start : first date to compute weighting DataFrame
    # expiry : Series of ticker -> expiration dates
    # items : sequence of contract names

    dates = pd.date_range(start, expiry[-1], freq='B')
    weights = DataFrame(np.zeros((len(dates), len(items))),
                        index=dates, columns=items)

    prev_date = weights.index[0]
    for i, (item, ex_date) in enumerate(expiry.iteritems()):
        if i < len(expiry) - 1:
            weights.ix[prev_date:ex_date - pd.offsets.BDay(), item] = 1
            roll_rng = pd.date_range(end=ex_date - pd.offsets.BDay(),
                                     periods=roll_periods + 1, freq='B')

            decay_weights = np.linspace(0, 1, roll_periods + 1)
            weights.ix[roll_rng, item] = 1 - decay_weights
            weights.ix[roll_rng, expiry.index[i + 1]] = decay_weights
        else:
            weights.ix[prev_date:, item] = 1

        prev_date = ex_date

    return weights

weights = get_roll_weights('6/1/2012', expiry, prices.columns)
weights.ix['2012-09-12':'2012-09-21']

rolled_returns = (prices.pct_change() * weights).sum(1)
Rolling correlation and linear regression

aapl = web.get_data_yahoo('AAPL', '2000-01-01')['Adj Close']
msft = web.get_data_yahoo('MSFT', '2000-01-01')['Adj Close']

aapl_rets = aapl.pct_change()
msft_rets = msft.pct_change()

plt.figure()

pd.rolling_corr(aapl_rets, msft_rets, 250).plot()

plt.figure()

model = pd.ols(y=aapl_rets, x={'MSFT': msft_rets}, window=250)
model.beta

model.beta['MSFT'].plot()
######################################################
unit12
Advanced NumPy

from __future__ import division
from numpy.random import randn
from pandas import Series
import numpy as np
np.set_printoptions(precision=4)
import sys
ndarray object internals
NumPy dtype hierarchy

ints = np.ones(10, dtype=np.uint16)
floats = np.ones(10, dtype=np.float32)
np.issubdtype(ints.dtype, np.integer)
np.issubdtype(floats.dtype, np.floating)

np.float64.mro()
Advanced array manipulation  
Reshaping arrays

arr = np.arange(8)
arr
arr.reshape((4, 2))

arr.reshape((4, 2)).reshape((2, 4))

arr = np.arange(15)
arr.reshape((5, -1))

other_arr = np.ones((3, 5))
other_arr.shape
arr.reshape(other_arr.shape)

arr = np.arange(15).reshape((5, 3))
arr
arr.ravel()

arr.flatten()
C vs. Fortran order

arr = np.arange(12).reshape((3, 4))
arr
arr.ravel()
arr.ravel('F')
Concatenating and splitting arrays

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
np.concatenate([arr1, arr2], axis=0)
np.concatenate([arr1, arr2], axis=1)

np.vstack((arr1, arr2))
np.hstack((arr1, arr2))

from numpy.random import randn
arr = randn(5, 2)
arr
first, second, third = np.split(arr, [1, 3])
first
second
third
Stacking helpers:
#堆叠
arr = np.arange(6)
arr
arr1 = arr.reshape((3, 2))
arr1
arr2 = randn(3, 2)
arr2
np.r_[arr1, arr2]
np.c_[np.r_[arr1, arr2], arr]

np.c_[1:6, -10:-5]
Repeating elements: tile and repeat
#  reshape  tile("铺瓷砖")

arr = np.arange(3)
arr
arr.repeat(3)

arr.repeat([2, 3, 4])

arr = randn(2, 2)
arr
arr.repeat(2)
arr.repeat(2, axis=0)#   =0 向下铺

arr.repeat([2, 3], axis=0)
arr.repeat([2, 3], axis=1)

arr
np.tile(arr, 2)             #  想右铺两遍

arr
np.tile(arr, (2, 1))    #  乡下铺两遍  ，向右铺一遍
np.tile(arr, (3, 2))
Fancy indexing equivalents: take and put

arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
arr[inds]

arr.take(inds)   #  通过下标 取值 take()
arr.put(inds, 42)  #
arr
arr.put(inds, [40, 41, 42, 43])
arr

inds = [2, 0, 2, 1]
arr = randn(2, 4)
arr
arr.take(inds, axis=1)  #  zai特定住上  取值
Broadcasting

arr = np.arange(5)
arr
arr * 4

arr = randn(4, 3)
arr.mean(0)
demeaned = arr - arr.mean(0)
demeaned
demeaned.mean(0)

arr
row_means = arr.mean(1)
row_means.reshape((4, 1))
demeaned = arr - row_means.reshape((4, 1))
demeaned.mean(1)
Broadcasting over other axes

arr - arr.mean(1)

arr - arr.mean(1).reshape((4, 1))

arr = np.zeros((4, 4))
arr_3d = arr[:, np.newaxis, :]
arr_3d.shape

arr_1d = np.random.normal(size=3)
arr_1d[:, np.newaxis]
arr_1d[np.newaxis, :]

arr = randn(3, 4, 5)
depth_means = arr.mean(2)
depth_means
demeaned = arr - depth_means[:, :, np.newaxis]
demeaned.mean(2)

def demean_axis(arr, axis=0):
    means = arr.mean(axis)

    # This generalized things like [:, :, np.newaxis] to N dimensions
    indexer = [slice(None)] * arr.ndim
    indexer[axis] = np.newaxis
    return arr - means[indexer]
Setting array values by broadcasting

arr = np.zeros((4, 3))
arr[:] = 5
arr

col = np.array([1.28, -0.42, 0.44, 1.6])
arr[:] = col[:, np.newaxis]
arr
arr[:2] = [[-1.37], [0.509]]
arr
Advanced ufunc usage
Ufunc instance methods

arr = np.arange(10)
np.add.reduce(arr)      #  对数组的各个元素求和
arr.sum()

np.random.seed(12346)

arr = randn(5, 5)
arr[::2].sort(1) # sort a few rows
arr[:, :-1] < arr[:, 1:]
np.logical_and.reduce(arr[:, :-1] < arr[:, 1:], axis=1)

arr = np.arange(15).reshape((3, 5))
np.add.accumulate(arr, axis=1)

arr = np.arange(3).repeat([1, 2, 2])
arr
np.multiply.outer(arr, np.arange(5))

result = np.subtract.outer(randn(3, 4), randn(5))
result.shape

arr = np.arange(10)
np.add.reduceat(arr, [0, 5, 8])

arr = np.multiply.outer(np.arange(4), np.arange(5))
arr
np.add.reduceat(arr, [0, 2, 4], axis=1)

########################
# Custom ufuncs

def add_elements(x, y):
    return x + y
#np.frompyfun  接受一个函数 包含输入输出参数的个数
add_them = np.frompyfunc(add_elements, 2, 1)
add_them(np.arange(8), np.arange(8))

add_them = np.vectorize(add_elements, otypes=[np.float64])
add_them(np.arange(8), np.arange(8))

arr = randn(10000)
%timeit add_them(arr, arr)
%timeit np.add(arr, arr)
Structured and record arrays

dtype = [('x', np.float64), ('y', np.int32)]
sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)
sarr

sarr[0]
sarr[0]['y']

sarr['x']
Nested dtypes and multidimensional fields

dtype = [('x', np.int64, 3), ('y', np.int32)]
arr = np.zeros(4, dtype=dtype)
arr

arr[0]['x']

arr['x']

dtype = [('x', [('a', 'f8'), ('b', 'f4')]), ('y', np.int32)]
data = np.array([((1, 2), 5), ((3, 4), 6)], dtype=dtype)
data['x']
data['y']
data['x']['a']
Why use structured arrays?
Structured array manipulations: numpy.lib.recfunctions
More about sorting

arr = randn(6)
arr.sort()
arr

arr = randn(3, 5)
arr
arr[:, 0].sort()  # Sort first column values in-place
# 支队第一行排序
arr

arr = randn(5)
arr
np.sort(arr)
arr

arr = randn(3, 5)
arr
arr.sort(axis=1)  #  axis = 1 对指定主 单独排序
arr

arr[:, ::-1]   #   取反 
arr[::-1]  # 默认对列取反
Indirect sorts: argsort and lexsort

values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()  #    返回下标
indexer
values[indexer]

from numpy import random
from numpy.random import randn
arr = randn(3, 5)
arr[0] = values
arr
arr[:, arr[0].argsort()]      #   可以对列标题排序

first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
sorter = np.lexsort((first_name, last_name))
zip(last_name[sorter], first_name[sorter])
Alternate sort algorithms

values = np.array(['2:first', '2:second', '1:first', '1:second', '1:third'])
key = np.array([2, 2, 1, 1, 1])
indexer = key.argsort(kind='mergesort')
indexer
values.take(indexer)
#-----------------
numpy.searchsorted: Finding elements in a sorted array # 在有序数组中查找元素

arr = np.array([0, 1, 7, 12, 15])
arr
arr.searchsorted(9)

arr.searchsorted([0, 8, 11, 16])

arr = np.array([0, 0, 0, 1, 1, 1, 1])
arr.searchsorted([0, 1])
arr.searchsorted([0, 1], side='right')

data = np.floor(np.random.uniform(0, 10000, size=50))# 面元数据
bins = np.array([0, 100, 1000, 5000, 10000])
data

labels = bins.searchsorted(data)
labels

Series(data).groupby(labels).mean()

np.digitize(data, bins)

###########################################################
NumPy matrix class

X =  np.array([[ 8.82768214,  3.82222409, -1.14276475,  2.04411587],
               [ 3.82222409,  6.75272284,  0.83909108,  2.08293758],
               [-1.14276475,  0.83909108,  5.01690521,  0.79573241],
               [ 2.04411587,  2.08293758,  0.79573241,  6.24095859]])
X[:, 0]  # one-dimensional#    单个数字索引行  从 “ 1 ”开始
y = X[:, :1]  # two-dimensional by slicing
type(X)
X
y

np.dot(y.T, np.dot(X, y))

Xm = np.matrix(X)  #----

type(Xm)
ym = Xm[:, 0]
Xm
ym
ym.T * Xm * ym

Xm.I * X  # 你举证

Advanced array input and output
Memory-mapped files

mmap = np.memmap('mymmap', dtype='float64', mode='w+', shape=(10000, 10000))
mmap

section = mmap[:5]

section[:] = np.random.randn(5, 10000)
mmap.flush()
mmap
del mmap

mmap = np.memmap('mymmap', dtype='float64', shape=(10000, 10000))
mmap

%xdel mmap
!rm mymmap
HDF5 and other array storage options
Performance tips
The importance of contiguous memory

arr_c = np.ones((1000, 1000), order='C')
arr_f = np.ones((1000, 1000), order='F')
arr_c.flags
arr_f.flags
arr_f.flags.f_contiguous

%timeit arr_c.sum(1)
%timeit arr_f.sum(1)

arr_f.copy('C').flags

arr_c[:50].flags.contiguous
arr_c[:, :50].flags

%xdel arr_c
%xdel arr_f
%cd ..
Other speed options: Cython, f2py, C
from numpy cimport ndarray, float64_t

def sum_elements(ndarray[float64_t] arr):
    cdef Py_ssize_t i, n = len(arr)
    cdef float64_t result = 0

    for i in range(n):
        result += arr[i]

    return result




reduce
filter
map
import decimal
from decimal import Decimal
a=10
a.bit_length()
type(a)
import math
import matplotlib.pyplot as plt
#import bsm_functions import bsm_call_value
mkdir python
%bookmark object
%time %prun
%magic

sigma 
