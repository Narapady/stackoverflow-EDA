# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from collections import Counter, OrderedDict
import re
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import opendatasets as od
from IPython import get_ipython

# %% [markdown]
# # EDA on Stackoverflow Developer Survey

# %%
get_ipython().system('pip install opendatasets --upgrade --quiet')

# %% [markdown]
# # Download the dataset

# %%
od.download('stackoverflow-developer-survey-2020')


# %%
# Import necessary libraries
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
df_raw = pd.read_csv(
    './stackoverflow-developer-survey-2020/survey_results_public.csv')
df_raw.head()


# %%
schema_df = pd.read_csv(
    './stackoverflow-developer-survey-2020/survey_results_schema.csv', index_col='Column')
# Using schema_raw to retrieve questions
schema_raw = schema_df.QuestionText

# %% [markdown]
# ## Data Preparation & Cleaning
# ### Limiting analysis to the following areas:
# - Demorgraphics of the survey respondents & the global programming community
# - Distribution of programming skills, experience and preferences
# - Employment-related inforamtion, preferences & opinions

# %%
selected_columns = [
    # Demographics
    'Country',
    'Age',
    'Gender',
    'EdLevel',
    'UndergradMajor',
    # Programming experience
    'Hobbyist',
    'Age1stCode',
    'YearsCode',
    'YearsCodePro',
    'LanguageWorkedWith',
    'LanguageDesireNextYear',
    'NEWLearn',
    'NEWStuck',
    # Employment
    'Employment',
    'DevType',
    'WorkWeekHrs',
    'JobSat',
    'JobFactors',
    'NEWOvertime',
    'NEWEdImpt'
]


# %%
# Filter only selected columns
df = df_raw[selected_columns]
df.head()


# %%
df.describe()


# %%
df.info()


# %%
df.isnull().sum()

# %% [markdown]
# ### Convert YearCode, Age1stCode, YearCodePro, and WorkWeekHrs columns to number

# %%
# Make a copy of original df
survey_df = df.copy()
# Replace non-numeric value with NaN
survey_df['YearsCode'] = pd.to_numeric(survey_df['YearsCode'], errors='coerce')
survey_df['Age1stCode'] = pd.to_numeric(
    survey_df['Age1stCode'], errors='coerce')
survey_df['YearsCodePro'] = pd.to_numeric(
    survey_df['YearsCodePro'], errors='coerce')
survey_df['WorkWeekHrs'] = pd.to_numeric(
    survey_df['WorkWeekHrs'], errors='coerce')
survey_df.describe()

# %% [markdown]
# ### Drop rows with invalid information

# %%
survey_df.drop(survey_df[survey_df['Age'] > 100].index, inplace=True)
survey_df.drop(survey_df[survey_df['Age'] < 10].index, inplace=True)
survey_df.drop(survey_df[survey_df['Age'] <
                         survey_df['Age1stCode']].index, inplace=True)
survey_df.drop(survey_df[survey_df['WorkWeekHrs'] > 100].index, inplace=True)
survey_df.drop(survey_df[survey_df['Age'] <
                         survey_df['YearsCode']].index, inplace=True)


# %%
survey_df.describe()


# %%


def replace_with_nan(gender):
    """ Replace ambigious gender with nan"""
    if gender is np.nan:
        return 'NaN'

    gender_str = re.split(';|, |\*|\n', gender)
    if len(gender_str) > 1 and gender_str[0] == 'Non-binary':
        return ' '.join(gender_str)
    elif len(gender_str) == 1:
        return gender_str[0]
    else:
        return 'NaN'


# %%
survey_df['Gender'] = survey_df['Gender'].apply(lambda x: replace_with_nan(x))

# %% [markdown]
# ## Exploratory Analysis and Visualization

# %%
# Configuration
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# %%
schema_raw.Country


# %%
survey_df['Country'].nunique()


# %%
# Top 15 countries with highest repsonses
top_15 = survey_df['Country'].value_counts().head(15)
lowest_15 = survey_df['Country'].value_counts().tail(15)
top_15


# %%
# Plot the distributions
sns.barplot(x=top_15.index, y=top_15)
plt.title(schema_raw['Country'])
plt.xlabel('Country')
plt.ylabel('Number of respondants')
plt.xticks(rotation=90)


# %%
sns.displot(survey_df['Age'], kde=True)
plt.title("Age Distribution")


# %%
plt.hist(survey_df['Age'], bins=np.arange(10, 100, 5), color='purple')
plt.title(schema_raw['Age'])
plt.ylabel('Counts')
plt.xlabel('Age')


# %%
gender = survey_df['Gender'].value_counts()
gender


# %%
label = gender.index
value = gender.to_list()
plt.title('Respndants Gender Wise')
plt.pie(value, labels=label, shadow=True, autopct='%1.1f%%')


# %%
plt.figure(figsize=(15, 13))
sns.countplot(y=survey_df['EdLevel'])
plt.title(schema_raw['EdLevel'])
plt.ylabel(None)


# %%
undergrad_majors = survey_df['UndergradMajor'].value_counts()
plt.figure(figsize=(15, 13))
sns.countplot(y=survey_df['UndergradMajor'])
plt.title(schema_raw['UndergradMajor'])
plt.ylabel(None)


# %%
percentage_by_major = (survey_df['UndergradMajor'].value_counts(
) / survey_df['UndergradMajor'].count()) * 100
percentage_by_major


# %%
plt.figure(figsize=(15, 13))
plt.title('Percentage of Undergrad Major')
plt.xlabel('Percentage')
plt.ylabel(None)
sns.barplot(y=percentage_by_major.index, x=percentage_by_major.to_list())


# %%
edu_importance = survey_df['NEWEdImpt'].value_counts()

plt.figure(figsize=(12, 10))
plt.title('Importance of Formal Education')
plt.xlabel('Level of Importance')
plt.xticks(rotation=65)
plt.ylabel('Numbers of respondants')
sns.barplot(x=edu_importance.index, y=edu_importance.to_list())


# %%
# CS students who think CS degree is not important at all
not_important = survey_df[(survey_df['UndergradMajor'] == 'Computer science, computer engineering, or software engineering') & (
    survey_df['NEWEdImpt'] == 'Not at all important/not necessary')].shape[0]


# %%
total_cs_students = (survey_df['UndergradMajor'] ==
                     'Computer science, computer engineering, or software engineering').sum()


# %%
pct_employment = survey_df['Employment'].value_counts(
    normalize=True, ascending=True) * 100
plt.title('Percentage of Developers Regarding Employment')
plt.xlabel('Percentage')
pct_employment.plot(kind='barh', color='g')


# %%


def count_items(series):
    """Take a series and count different kind of position"""
    result = []
    # loop through each row, split, and add to result list
    for row in series:
        if row is not np.nan:
            item_list = re.split(';|,', row)
            for item in item_list:
                result.append(item.strip())
    # Count distint item in result list
    return Counter(result)


# %%
job_title_dict = count_items(survey_df['DevType'])

# Get keys and values and plit
keys = list(job_title_dict.keys())
values = list(job_title_dict.values())
job_title_pct = [(value / sum(values)) * 100 for value in values]

# plot bar chart
plt.figure(figsize=(15, 12))
plt.title('Percentage of Job titles of Employed Respondants')
plt.xlabel('Percentage')
y_pos = len(keys)
sns.barplot(x=job_title_pct, y=keys)


# %%
def is_data(series):
    """Return a list of boolean if if items in series contains key word"""
    word = re.compile('Data|Database|data|machine learning')
    bool_list = []
    for item in series:
        if item is not np.nan:
            if word.search(item):
                bool_list.append(True)
            else:
                bool_list.append(False)
        else:
            bool_list.append(False)
    return bool_list


# %%
data = is_data(survey_df['DevType'])
# selected data related fileds
data_df = survey_df[data]
# only data scientists
data_science = data_df[data_df['DevType'] ==
                       'Data scientist or machine learning specialist']
count = data_science['Employment'].value_counts()

# Employment type of data scientists
plt.title("Employment type of data scientists")
sns.barplot(x=count, y=count.index)
plt.xlabel("Count")

# %% [markdown]
# ### What are the most popular programming language among developers?

# %%
language_dict = count_items(survey_df['LanguageWorkedWith'])
language_dict = {k: v for k, v in sorted(
    language_dict.items(), key=lambda item: item[1], reverse=True)}

plt.figure(figsize=(12, 12))
plt.title('Languages Used in 2020')
sns.barplot(x=list(language_dict.values()), y=list(language_dict.keys()))

# %% [markdown]
# ### What top languages used by data scientist?

# %%
language_dict = count_items(data_science['LanguageWorkedWith'])
language_dict = {k: v for k, v in sorted(
    language_dict.items(), key=lambda item: item[1], reverse=True)}

plt.figure(figsize=(12, 12))
plt.title('Top Languages Used By Data Scientists')
sns.barplot(x=list(language_dict.values()), y=list(language_dict.keys()))

# %% [markdown]
# ### What languages are popular in Cambodia?

# %%
cambo_df = survey_df[survey_df['Country'] == 'Cambodia']
cambo_lang = count_items(cambo_df['LanguageWorkedWith'])

plt.figure(figsize=(12, 12))
plt.title('Top Languages Used in Cambodia')
sns.barplot(x=list(cambo_lang.values()), y=list(cambo_lang.keys()))

# %% [markdown]
# ### Top languages interested next year

# %%
data = is_data(cambo_df['DevType'])
data_cambo = cambo_df[data]

language_dict = count_items(survey_df['LanguageDesireNextYear'])
language_dict = {k: v for k, v in sorted(
    language_dict.items(), key=lambda item: item[1], reverse=True)}

plt.figure(figsize=(12, 12))
plt.title('Languages Interested Next Year')
sns.barplot(x=list(language_dict.values()), y=list(language_dict.keys()))

# %% [markdown]
# ### Which language gain the most popularity ?

# %%


def pct_change(num1, num2):
    """ Calculate percentage change of two numbers"""
    dif = num2 - num1
    result = 0

    if dif > 0:
        result = (dif / num1) * 100
        return round(result, 2)
    elif dif < 0:
        result = -abs((dif / num1) * 100)
        return round(result, 2)
    return 0


# %%

language_worked = count_items(survey_df['LanguageWorkedWith'])
language_interested = count_items(survey_df['LanguageDesireNextYear'])

language_worked = OrderedDict(sorted(language_worked.items()))
language_interested = OrderedDict(sorted(language_interested.items()))

most_loved_lang = {}

for (k, v), (k1, v1) in zip(language_worked.items(), language_interested.items()):
    most_loved_lang[k] = pct_change(v, v1)

plt.figure(figsize=(13, 10))
plt.title(
    'Percentage of Change in Popularity of the Languague Over One Year (2020-2021')
plt.xlabel('Percentage')
sns.barplot(x=list(most_loved_lang.values()), y=list(most_loved_lang.keys()))

# %% [markdown]
# ### Which countries do developers work the highgest number of hours per week? Consider countries with more than 250 response only.

# %%


def less_than_250(dicts):
    """Return countries with less than 250 respondants"""
    result = []
    for k, v in dicts.items():
        if v < 250:
            result.append(k)
    return result


# %%
# Get a list of excluded countries
excluded_countries = less_than_250(Counter(survey_df['Country']))

# Fileter out
df = survey_df.copy()
for country in excluded_countries:
    df = df[df['Country'] != country]


# %%
# Average hour worked in those countries that have more than 60 hours work week
average_hrs = df.groupby('Country')['WorkWeekHrs'].mean()
average_hrs.sort_values(ascending=False, inplace=True)
top_15 = average_hrs.head(15)
# plot
plt.figure(figsize=(14, 9))
plt.title('Top 15 Countries With Long Average Work Week')
plt.xlabel('Hours')
sns.barplot(x=top_15.to_list(), y=top_15.index)

# %% [markdown]
# ### How important is it to start young to build a career in programming?

# %%
sns.scatterplot(x='Age', y='YearsCodePro', hue='Hobbyist', data=survey_df)
plt.ylabel('Years of professional coding experience')


# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
not_hobby = survey_df[survey_df['Hobbyist'] == 'No']
hobby = survey_df[survey_df['Hobbyist'] == 'Yes']

axes[0].scatter(not_hobby['Age'], not_hobby['YearsCodePro'])
axes[0].set_title('Not Hobbyist')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Years of Professional Experience')

axes[1].scatter(hobby['Age'], hobby['YearsCodePro'], color='g')
axes[1].set_title('Hobbyist')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Years of Professional Experience')


# %%
pro_ex_gender = survey_df.groupby('Gender')['YearsCodePro'].mean()
pro_ex_gender.drop(
    ['NaN', 'Non-binary genderqueer or gender non-conforming'], inplace=True)
plt.title('Average Professional Experience in Programming for Men and Women')
plt.xlabel('Years')
sns.barplot(y=pro_ex_gender.index, x=pro_ex_gender)


# %%
