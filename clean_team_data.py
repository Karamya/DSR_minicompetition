import pandas as pd
import numpy as np
import pickle

######################################################################
### Define various column names and categories
######################################################################

use_cols = np.arange(26, 251)

p1_features_cat_noorder = [
    "Person.ID",
    "Role",
    "Country.of.Birth",
    "Home.Language",
    "Dept.No.",
    "Faculty.No.",
    "With.PHD"
        ]

p1_features_cat_order = [
    "No..of.Years.in.Uni.at.Time.of.Grant"
        ]

p1_features_cont = [
    "Year.of.Birth",
    "Number.of.Successful.Grant",
    "Number.of.Unsuccessful.Grant",
    "A.",
    "A",
    "B",
    "C"
        ]

no_years_cats = [
    'Less than 0', 
    '>=0 to 5', 
    '>5 to 10', 
    '>10 to 15',
    'more than 15'
    ]


def get_all_poss_values(df_in, col_name):
    # return list of all possible values for given col_name
    return pd.Series(df_in[col_name].values.ravel()).dropna().unique()

def get_all_poss_values_prefixed(df_in, col_name):
    # return list of all possible values for given col_name
    x = pd.Series(df_in[col_name].values.ravel()).dropna().unique()
    return col_name + "_" +  x

######################################################################
### Read data into dataframe
######################################################################

# define a dic for data type for read_csv
type_dic = {}
for i in np.arange(1,16):
    for feat in p1_features_cat_noorder + p1_features_cat_order:
        type_dic[feat + '.' + str(i)] = "str" #convert to cat later after manipulation
    for feat in p1_features_cont:
        type_dic[feat + '.' + str(i)] = "float"

df = pd.read_csv("data/unimelb_training.csv", usecols=use_cols, dtype=type_dic)
df.columns = pd.MultiIndex.from_tuples([tuple(c.rsplit('.', 1)) for c in df.columns])
n_rows = df.shape[0]

#syntax to get second level index
#df.iloc[:, df.columns.get_level_values(1)=='1'].head(20)

######################################################################
### Fill missing data
######################################################################

# fill in language based on country
eng_countries = ['Australia', 'North America', 'Great Britain', 'New Zealand', 'South Africa']
for i in np.arange(1, 16): 
    is_eng_country = df["Country.of.Birth", str(i)].isin(eng_countries)
    is_nan_country = df["Country.of.Birth", str(i)].isnull()
    is_lang_nan = df["Home.Language", str(i)].isnull()
    # 2 following lines give a warning that I'm trying to set a value on a 
    # copy of a slice, but I have checked and it is working
    df["Home.Language", str(i)].loc[is_eng_country & is_lang_nan] = 'English'
    df["Home.Language", str(i)].loc[~is_eng_country & ~is_nan_country & is_lang_nan] = 'Other'

# now convert string series to categorical
for i in np.arange(1, 16): 
    for feat in p1_features_cat_noorder:
        df[feat, str(i)] = df[feat, str(i)].astype("category")
    # for "No..of.Years.in.Uni.at.Time.of.Grant" make it ordered
    for feat in p1_features_cat_order:
        df[feat, str(i)] = pd.Series(pd.Categorical(
                df[feat, str(i)].values,
                categories=no_years_cats,
                ordered=True
            ))

# Now fill in missing data, only for the first team member. This ensures that
# when we aggregate teams, we will not have any missing data

# fill missing continuous data (median) for first team member
for feat in p1_features_cont:
    df[feat, "1"] = df[feat, "1"].fillna(df[feat, "1"].median())

# fill missing continuous data (mode) for first team member
for feat in p1_features_cat_noorder + p1_features_cat_order:
    df[feat, "1"] = df[feat, "1"].fillna(df[feat, "1"].mode()[0])

# drop Person.ID column for now
df.drop("Person.ID", axis=1, inplace=True)
del p1_features_cat_noorder[0]

######################################################################
### Create dataframe for learning algorithm
######################################################################

X_team = pd.DataFrame()

######################################################################
### Categorical feature engineering
######################################################################

def create_team_fracs_df(df_in, col_name):
    #make a df_out representing the fractions of team members beloning to a given
    #category for a given col_name

    #create empty one hot for this column
    all_values = get_all_poss_values_prefixed(df_in, col_name)
    A_dummy = np.zeros((n_rows, len(all_values)))
    df_out = pd.DataFrame(A_dummy, columns=all_values)

    #loop through team members and get one hot for each. 
    #Reindex like df_out and add to it
    for i in np.arange(1, 16):
        x = pd.get_dummies(df[col_name, str(i)], prefix=col_name)\
              .reindex_like(df_out)\
              .fillna(0)
        df_out = df_out.add(x)

    #normalise by rows
    df_out = df_out.div(df_out.sum(axis=1), axis=0)

    return df_out

# add one-hot encoded fraction of categorical data
for feat in p1_features_cat_noorder:                                                   
    x = create_team_fracs_df(df, feat)                                         
    X_team = pd.concat([X_team, x], axis=1) 

######################################################################
### Continous feature engineering
######################################################################

# add number of people in team
x = df["Role"].count(axis=1)
x.name = "No. team members"
X_team = pd.concat([X_team, x], axis=1) 

######################################################################
### Dump pickle file of X_team
######################################################################
#pickle.dump(X_team, open("X_team_cat_fracs.p", "wb"))
