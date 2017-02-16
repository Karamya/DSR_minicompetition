# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

df = pd.read_csv("./data/unimelb_training.csv", low_memory = False)

###############################################################################

#### Fill nan values with mode in case of categorical variables
#### and perform one hot encoding on the categorical data

###############################################################################


df["Sponsor.Code"] = df["Sponsor.Code"].fillna(df["Sponsor.Code"].value_counts().index[0])
df = df.join(pd.get_dummies(df["Sponsor.Code"], prefix = "S"))



df["Grant.Category.Code"] = df["Grant.Category.Code"].fillna(df[
        "Grant.Category.Code"].value_counts().index[0])
df = df.join(pd.get_dummies(df["Grant.Category.Code"], prefix = "G"))



#### This is an ordered category, so no one-hot encoding was performed
df["Contract.Value.Band...see.note.A"] = df["Contract.Value.Band...see.note.A"].fillna(
    df["Contract.Value.Band...see.note.A"].value_counts().index[0])


###############################################################################

#### Drop the columns which were one-hot encoded

###############################################################################

df.drop(["Sponsor.Code"], axis = 1, inplace = True)
df.drop(["Grant.Category.Code"], axis = 1, inplace = True)



###############################################################################

#### Convert Start. Date format to date time and convert it into an integer
#### good for training (date column could not be used as such for training the data)

###############################################################################

#df["Start.date"] = pd.to_datetime(df["Start.date"])


###############################################################################

#### Split Start.Date to year, month and day, day of week, week of year

###############################################################################

df["Start.Year"] = pd.DatetimeIndex(df["Start.date"]).year
df["Start.Month"] = pd.DatetimeIndex(df["Start.date"]).month
df["Start.Year"] = df["Start.Year"]- df["Start.Year"].min() + 1 ### Making year column from 2005-2008 to 1-4
df["Start.Day"] = pd.DatetimeIndex(df["Start.date"]).day
df["Start.DayofWeek"] = pd.DatetimeIndex(df["Start.date"]).dayofweek
df["Start.WeekofYear"] = pd.DatetimeIndex(df["Start.date"]).weekofyear


def conv_day(a): return a.split("/")[0]
def conv_month(a): return a.split("/")[1]
def conv_season(a):
    month = int(conv_month(a))
    day = int(conv_day(a))
    if month in [1,2]:
        return 0 # "Winter"
    elif month in [4, 5]:
        return 1 # "Spring"
    elif month in [7, 8]:
        return 2 # "Summer"
    elif month in [10, 11]:
        return 3 # "Autumn"
    elif month in [3]:
        if day <= 21:
            return 0 # "Winter"
        else:
            return 1 # "Spring"
    elif month in [6]:
        if day <= 21:
            return 1 # "Spring"
        else:
            return 2 # "Summer"
    elif month in [9]:
        if day <= 21:
            return 2 # "Summer"
        else:
            return 3 # "Autumn"
    elif month in [12]:
        if day <= 21:
            return 3 # "Autumn"
        else:
            return 0 # "Winter"
    return -1
df["Start.Season"] = df["Start.date"].apply(conv_season)


df.drop(["Start.date"], axis = 1, inplace = True)


###############################################################################

#### Ordered category are assigned and given categorical codes

###############################################################################

contract_value_ordered = ['A ', 'B ', 'C ', 'D ', 'E ', 'F ', 'G ', 'H ', 'I ', 'J ', 'K ', 'L ', 'M ',
                         'N ', 'O ', 'P ', 'Q ']
df['Contract.Value.Band...see.note.A'] = df["Contract.Value.Band...see.note.A"].astype(
    'category', ordered = True, categories = contract_value_ordered).cat.codes

###############################################################################

#### Here the codes and percentages containing nan values were replaced with 0

###############################################################################

for feature in ["RFCD.Code.", "RFCD.Percentage.", "SEO.Code.", "SEO.Percentage."]:
    for i in range(1, 6):
        df[feature+str(i)] = df[feature+str(i)].fillna(0)

###############################################################################

#### perform one hot encoding on RFCD/SEO code and do scalar multiplication with
#### their respective percentages and then add all 5 dataframes

###############################################################################


RFCD_dummy = pd.DataFrame()
for i in range(1, 6):
    RFCD_dummy = RFCD_dummy.add(pd.get_dummies(df["RFCD.Code." + str(i)],
                                       columns = ["RFCD.Code." + str(i)], prefix = "RFCD").multiply(
                                       df["RFCD.Percentage." + str(i)],axis = 0),
                        fill_value = 0.0)

SEO_dummy = pd.DataFrame()
for i in range(1, 6):
    SEO_dummy = SEO_dummy.add(pd.get_dummies(df["SEO.Code." + str(i)],
                                       columns = ["SEO.Code." + str(i)], prefix = "SEO").multiply(
                                       df["SEO.Percentage." + str(i)],axis = 0),
                        fill_value = 0.0)


###############################################################################

#### combine all dataframes (df, RFCD_dummy and SEO_dummy)

###############################################################################

df2 = df.join(RFCD_dummy.join(SEO_dummy))



###############################################################################

#### drop all sets of RFCD/SEO Code and Percentage from the dataframe

###############################################################################


for i in range(1,6):
    df2.drop(["RFCD.Code." + str(i), "RFCD.Percentage." + str(i),
                 "SEO.Code." + str(i), "SEO.Percentage." + str(i)],
                   axis = 1, inplace = True)

df2.dropna(axis = 1, how = 'any', inplace = True)
###############################################################################

#### Dump the first park of data to a pickle file

###############################################################################


#df2.to_pickle("first_half_data.pickle")
