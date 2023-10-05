import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime



column_names = ['Year.Month', 'Victimisations', 'SEX','Age.Group.5Yr.Band','OOI.Exclusion','Ethnic.Group','ANZSOC.Group','class']

# Read the CSV file and add column names
df = pd.read_csv('clean_class.csv', encoding='utf-8')




df_sex = df.groupby('SEX')['Victimisations'].sum()
df_sex.plot(kind='bar', title='Number of Victimizations by Gender')
plt.show()


df_ethnic = df.groupby('Ethnic.Group')['Victimisations'].sum()
df_ethnic.plot(kind='pie', title='Number of Victimizations by Ethnic Group', autopct='%1.1f%%')
plt.show()

# Visualization 3: Stacked bar plot - Victimizations by Age Group based on ANZSOC Group.
df_age_anzsoc = df.groupby(['Age.Group.5Yr.Band', 'ANZSOC.Group'])['Victimisations'].sum().unstack()
ax = df_age_anzsoc.plot(kind='bar', stacked=True, title='Victimizations by Age Group & ANZSOC Group', figsize=(12, 8))
plt.legend(title='ANZSOC Group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
ax.set_ylabel("Victimisations")
plt.tight_layout()
plt.show()

# Visualization 4: Bar plot - Victimizations with 'Court action' vs 'No crime'.
df_ooi = df.groupby('OOI.Exclusion')['Victimisations'].sum()
df_ooi.plot(kind='bar', title='Victimizations with Court action vs No crime')
plt.show()



# Visualization 5: Line plot - Trends in total victimizations over time.
df_date = df.groupby('Date')['Victimisations'].sum()
df_date.plot(kind='line', title='Trends in Total Victimizations Over Time')
plt.ylabel('Total Victimisations')
plt.show()

# Visualization 6: Bar plot - Number of victimizations by ANZSOC Group.
df_anzsoc = df.groupby('ANZSOC.Group')['Victimisations'].sum().sort_values(ascending=False)
df_anzsoc.plot(kind='bar', title='Number of Victimizations by ANZSOC Group', figsize=(10, 5))
plt.ylabel('Total Victimisations')
plt.show()

# Visualization 7: Heatmap - Relationship between Ethnic Group, Age Group, and total victimizations.
df_heatmap = df.groupby(['Ethnic.Group', 'Age.Group.5Yr.Band'])['Victimisations'].sum().unstack()
plt.figure(figsize=(10, 7))
sns.heatmap(df_heatmap, annot=True, cmap="YlGnBu", fmt='g')
plt.title('Relationship between Ethnic Group, Age Group, and Total Victimizations')
plt.show()

# Visualization 8: Bar plot - Distribution of victimizations for each Age Group.
df_age = df.groupby('Age.Group.5Yr.Band')['Victimisations'].sum().sort_values(ascending=False)
df_age.plot(kind='bar', title='Distribution of Victimizations for Each Age Group')
plt.ylabel('Total Victimisations')
plt.show()













