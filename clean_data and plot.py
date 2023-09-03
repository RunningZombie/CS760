#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('data.csv', encoding='utf-8')###读取文件


# In[3]:


data.head()


# In[4]:


###预处理
data.dropna(inplace=True)
###转换格式
###转换日期
month_dict = {
    '1月': 'January', '2月': 'February', '3月': 'March',
    '4月': 'April', '5月': 'May', '6月': 'June',
    '7月': 'July', '8月': 'August', '9月': 'September',
    '10月': 'October', '11月': 'November', '12月': 'December'
}
def convert_to_english_date(chinese_date_str):
    month_chinese = chinese_date_str.split('月')[0] + '月'
    year = chinese_date_str.split('月')[1]
    month_english = month_dict.get(month_chinese, 'Unknown')
    return f"{month_english} {year}"

data['Year Month 月'] = data['Year Month 月'].apply(convert_to_english_date)
data['Year Month 月'] = pd.to_datetime(data['Year Month 月'], errors='coerce')
data.rename(columns={'Year Month 月': 'Date'}, inplace=True)
data.sort_values(by='Date', inplace=True)


# In[5]:


###转换年龄
def classify_age_group(age_group):
    if age_group in ['0-4', '5-9', '10-14', '15-19']:
        return 'Teenagers'
    elif age_group in ['20-24', '25-29', '30-34', '35-39']:
        return 'Middle-aged-young'
    elif age_group in ['40-44', '45-49', '50-54', '55-59']:
        return 'Middle-aged-old'
    elif age_group in ['60-64', '65-69', '70-74', '75-79', '80yearsorover']:
        return 'Elderly'
    else:
        return 'Unknown'
data['Age Group'] = data['Age Group'].apply(classify_age_group)
data = data[data['Age Group'] != 'Unknown']


# In[6]:


data = data[data['Ethnic Group'] != 'Not Stated']
data["Ethnic Group"].unique()


# In[7]:


###检查Victimisations
data['Victimisations'] = pd.to_numeric(data['Victimisations'], errors='coerce')
if data['Victimisations'].isna().sum() == 0:
    print("The column is all numbers.")
    min_value = data['Victimisations'].min()
    max_value = data['Victimisations'].max()
    print(f"the range is from {min_value} to {max_value}")

else:
    print("invalid")


# In[18]:


data.to_csv('clean_data.csv', index=False)


# In[9]:


sns.countplot(data=data, x='Age Group')
plt.show()


# In[10]:


ethnic_group_counts = data['Ethnic Group'].value_counts()
plt.figure(figsize=(10, 7))
plt.pie(ethnic_group_counts, labels=ethnic_group_counts.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Ethnic Groups')
plt.show()


# In[16]:


sns.countplot(data=data, y='ANZSOC Division')
plt.show()


# In[17]:


data.groupby('Date')['Victimisations'].sum().plot()
plt.xlabel('Date')
plt.ylabel('Total Victimisations')
plt.show()


# In[ ]:




