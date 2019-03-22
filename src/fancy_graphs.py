import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from tabulate import tabulate


def to_markdown(df, round_places=3):
    """Returns a markdown, rounded representation of a dataframe"""
    print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe', showindex=False))

def to_markdown_with_index(df, round_places=3):
    """Returns a markdown, rounded representation of a dataframe"""
    print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe', showindex=True))


mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
})
data_dir = '../data/'
img_dir = '../fancy_images/'
cities_500_csv = data_dir +'500_Cities.csv'
acs_merged_csv = data_dir + 'Clean_data_w_state.csv'

if not os.path.isdir(img_dir):
    os.makedirs(img_dir)

df1 = pd.read_csv(cities_500_csv)
df2 = pd.read_csv(acs_merged_csv)
outcome_vars = ['SLEEP', 'CSMOKING', 'BINGE']
df1_sub = df1[df1['MeasureId'].isin(outcome_vars)]
ax = sns.violinplot(x="MeasureId", y="Data_Value", data=df1_sub)
plt.title('Outcome Variables')
plt.tight_layout()
# plt.show()
plt.savefig(img_dir+'Outcome_Violin.png')
plt.close()

ax = sns.violinplot(x=df2["Med_age"], orient='v')
plt.title('Median Age')
ax.set(xlabel='', ylabel='')
plt.tight_layout()
# plt.show()
plt.savefig(img_dir+'Med_Age_Violin.png')
plt.close()


state_vars = [ x for x in df2.columns if x.startswith('State')]
states = df2[state_vars]
states['State_Abbr_CO']=states[state_vars].apply(lambda row: 1 if row.sum()==0 else 0, axis=1)
states_sum = states.sum()

ax = sns.violinplot(x=states_sum, orient='v')
plt.title('Count of State Tracts')
ax.set(xlabel='', ylabel='')
plt.tight_layout()
# plt.show()
plt.savefig(img_dir+'State_Count_Violin.png')
plt.close()

vois = ['Percent_female', 'Edu_less_than_hs_or_GED',
       'Income_to_pov_rat_lt_1_5', 'Commute_time_lt_30',
       'Work_depart_before_8am', 'Percent_insured']
mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
})

ax = sns.violinplot(data=df2[vois])
plt.title('Demographic Percentages')
ax.set(ylabel='%')
xticks=['Female', 'Edu < HS','IPR < 1.5','Commute < 30', 'Work before 8', 'Insured']
ax.set_xticklabels(xticks, rotation=30)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.tight_layout()
# plt.show()
plt.savefig(img_dir+'Demographics_Violin.png')
plt.close()

#### TABLE FOR SAMPLE SIZE #######

'''
# BD
|   Initial |   Final |   Dropped |   Dropped % |
|----------:|--------:|----------:|------------:|
|     28004 |   27141 |       863 |       3.082 |
Directory exists: /home/danny/Desktop/galvanize/500_cities/src/../images_smoking/
Table for missing data
|   Initial |   Final |   Dropped |   Dropped % |
|----------:|--------:|----------:|------------:|
|     28004 |   27080 |       924 |         3.3 |
|   Initial |   Final |   Dropped |   Dropped % |
|----------:|--------:|----------:|------------:|
|     28004 |   27137 |       867 |       3.096 |
'''
data = {'Initial': [28004, 28004, 28004],
                'Final' : [27141, 27080, 27137]}
df = pd.DataFrame.from_dict(data)
df['Dropped'] = df.Initial-df.Final
df['Drop %'] = df.Dropped/df.Initial*100
df = df.T
df.columns = ['Binge Drinking', 'Smoking', 'Sleep < 7hrs']
to_markdown_with_index(df)


data = {'Binge Drinking': [43.0240, -0.2442, -17.0208, -6.5088, -10.0818, 0.9218, -0.8901, -4.1470],
                'Smoking' : [4.6686, -0.0455, -3.1133, 12.0400, 16.5517,0.3347, 3.8534, 6.2355 ],
                'Sleep < 7hrs': [22.4630, -0.0675, 7.5038, 10.9263, 12.2285, -7.2640, 2.2277, 3.0339 ]}
labels = ['Intecept',
            'Median Age',
            '% Female',
            '% Edu < HS',
            '% IPR < 1.5',
            '% Commute < 30',
            '% Depart before 8',
            '% Insured']
df = pd.DataFrame.from_dict(data)
# df['Dropped'] = df.Initial-df.Final
# df['Drop %'] = df.Dropped/df.Initial*100
df.index = labels
# df.columns = ['Binge Drinking', 'Smoking', 'Sleep < 7hrs']
print('\n')
to_markdown_with_index(df)




vois = ['Med_age', 'Percent_female', 'Edu_less_than_hs_or_GED',
       'Income_to_pov_rat_lt_1_5', 'Commute_time_lt_30',
       'Work_depart_before_8am', 'Percent_insured']

corr = df2[vois].corr()
to_markdown_with_index(corr)

# ax = sns.pairplot(data=df2[vois])
# plt.title('Demographic Pairplot')
# ax.set(ylabel='%')
# # xticks=['Female', 'Edu < HS','IPR < 1.5','Commute < 30', 'Work before 8', 'Insured']
# # ax.set_xticklabels(xticks, rotation=30)
# # ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
# plt.tight_layout()
# # plt.show()
# plt.savefig(img_dir+'Demographics_Pairplot.png')
# plt.close()


data = {'Binge Drinking': [0.680, 2.159, 2.189],
                'Smoking' : [0.825,  2.662, 2.742],
                'Sleep < 7hrs': [ 0.769,  3.056, 3.078]}
df = pd.DataFrame.from_dict(data)
df.index = ['Adj R^2', 'Train RMSE', 'Test RMSE']
to_markdown_with_index(df)



mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'xx-small',
})

data = {'Binge Drinking': [-17.0208, -6.5088, -10.0818, 0.9218, -0.8901, -4.1470],
                'Smoking' : [-3.1133, 12.0400, 16.5517,0.3347, 3.8534, 6.2355 ],
                'Sleep < 7hrs': [7.5038, 10.9263, 12.2285, -7.2640, 2.2277, 3.0339 ]}
labels = ['% Female',
            '% Edu < HS',
            '% IPR < 1.5',
            '% Commute < 30',
            '% Depart before 8',
            '% Insured']
df = pd.DataFrame.from_dict(data)
# df['Dropped'] = df.Initial-df.Final
# df['Drop %'] = df.Dropped/df.Initial*100
df.index = labels
# df.columns = ['Binge Drinking', 'Smoking', 'Sleep < 7hrs']
print('\n')
to_markdown_with_index(df)
ax = df.T.plot(kind='bar')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
ax.set(ylabel='Coefficient')
ax.legend(loc='lower right', ncol=2, fancybox=False, shadow=False)
plt.tight_layout()
# plt.show()
plt.savefig(img_dir+'Coeffs_Bar.png')
plt.close()
