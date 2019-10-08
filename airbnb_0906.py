import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,preprocessing,cross_validation,feature_extraction
import random
import seaborn as sns

train_users = pd.read_csv('train_users_2.csv')
test_users = pd.read_csv('test_users.csv')
print(train_users.shape)
print(test_users.shape)

train_users.head()
test_users.head()

all_users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
all_users.head()
all_users.shape
all_users.tail()

#看一下样本标签什么情况
print(pd.value_counts(train_users['country_destination'].values))


sns.set_style()
des_countries = train_users.country_destination.value_counts(dropna=False)/train_users.shape[0]*100
des_countries.plot(kind='bar',rot=0)
plt.xlabel('Destination country')
plt.ylabel('Percetage of booking')

#下面看样本的整体情况
all_users.info()
test_users.info()
#发现 在测试集中date_first_booking完全缺失，所以下一阶段特征工程，可以把它删掉

#先看age整体情况
all_users['age'].describe() #age情况有点奇怪，最大年龄2014是不可能的


fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,5))
axes[0].set_title('Age<200')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
all_users[all_users.age<200].age.hist(bins=10,ax=axes[0])

axes[1].set_title('Age>200')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Count')
all_users[all_users.age>200].age.hist(bins=10,ax=axes[1])

# 特征gender
all_users.gender.value_counts(dropna=False)

#以下针对train set 具体看一下有什么表现
women = sum(train_users['gender']=='FEMALE')
men = sum(train_users['gender']=='MALE')
other = sum(train_users['gender']=='OTHER')

female_destinations = train_users.loc[train_users['gender']=='FEMALE','country_destination'].value_counts()/women*100
male_destinations = train_users.loc[train_users['gender']=='MALE','country_destination'].value_counts()/men*100
other_destinations = train_users.loc[train_users['gender']=='OTHER','country_destination'].value_counts()/other*100

female_destinations.plot(kind='bar',width=0.2,color='red',position=0,label='Female',rot=0)
male_destinations.plot(kind='bar',width=0.2,color='blue',position=1,label='Male',rot=0)
other_destinations.plot(kind='bar',width=0.2,color='green',position=2,label='Other',rot=0)

plt.legend()
plt.xlabel('Destination country')
plt.ylabel('Percentage of booking')
plt.show

#关于男女的users，旅行地的比例基本是一致的，other在US比其他性别低

#-----特征date_account_created和特征timestamp_forst_active----
#以下查看创建账号的时间趋势
fig=plt.figure(figsize=(12,6))
all_users['date_account_created']=pd.to_datetime(all_users['date_account_created']) #转换为python能认识的时间格式
all_users['date_account_created'].value_counts().plot(kind='line')
plt.xlabel('Year')
plt.ylabel('Count created')

#看一下第一次激活时间的时间趋势
fig=plt.figure(figsize=(12,6))
all_users['date_first_active']=pd.to_datetime((all_users.timestamp_first_active//1000000),format='%Y%m%d') #
all_users['date_first_active'].value_counts().plot(kind='line')
plt.xlabel('Year')
plt.ylabel('First active count')

#以上两个时间特征比较类似

#=----特征signup_method,signup_app-----

pd.value_counts(all_users['signup_method'].values)

pd.value_counts(all_users['signup_app'].values)

pd.value_counts(all_users['first_device_type'].values)

pd.value_counts(all_users['first_browser'].values)

pd.value_counts(all_users['language'].values)

lang = all_users.language.value_counts()/all_users.shape[0]*100
plt.figure(figsize=(12,10))
plt.xlabel('User_language')
plt.ylabel('percentage')
lang.plot(kind='bar',fontsize=17,rot=0)
#  96%都是英语
pd.value_counts(all_users['affiliate_channel'].values)
pd.value_counts(all_users['affiliate_provider'].values)

#session data
sessions=pd.read_csv('sessions.csv')
sessions.head()
sessions.shape
sessions.info()

len(sessions.user_id.unique())

#换个角度看一下
df_sess=sessions.groupby(['user_id'])['user_id'].count().reset_index(name='session_count')
df_sess.head()
df_sess.session_count.describe() #平均77

secs=sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
secs.columns=['user_id','secs_elapsed']
secs.describe()

sns.boxplot(x=secs['secs_elapsed'])

sessions.action_type.value_counts()
at=sessions.action_type.value_counts(dropna=False)/sessions.shape[0]*100
plt.figure(figsize=(12,8))
plt.xlabel('Action type')
plt.ylabel('Percentage')
at.plot(kind='bar',fontsize=17)

sessions.action.value_counts()
#对目标特征进行编码
label_df=train_users_labels.to_frame()
for data in [label_df]:
    data['country_deatination']=le.fit_transform(fata['country_destination'])
label_df.head()