import pandas as pd
import numpy as np
from data_cleaning import df2
from clean_team_data import X_team

######################################################################
### combine grant data with team data
######################################################################

print("Number of columns in grant data is ", len(df2.columns))
print("Number of columns in team data is ", len(X_team.columns))
clean_data = df2.join(X_team, on="Grant.Application.ID") 
clean_data.set_index("Grant.Application.ID", inplace=True)
print("Number of columns in the complete & cleaned data sets  is ", len(clean_data.columns))

######################################################################
### split into training, validation, and test sets
######################################################################

test_ids = pd.read_csv("data/testing_ids.csv", dtype=int)
test_ids.shape[0]

validation_ids = pd.read_csv("data/training2_ids.csv", dtype=int)
validation_ids.shape[0]

training_ids = clean_data.index.values
training_ids = np.setdiff1d(training_ids, test_ids)
training_ids = np.setdiff1d(training_ids, validation_ids)

y = clean_data["Grant.Status"]
X = clean_data.drop(["Grant.Status"], axis=1)

y_test = y.loc[test_ids["ids"].values]
y_validation = y.loc[validation_ids["ids"].values]
y_training = y.loc[training_ids]
print("\n")
print("Number of rows in y_training ", y_training.shape[0])
print("Number of rows in y_validation", y_validation.shape[0])
print("Number of rows in y_test", y_test.shape[0])

X_test = X.loc[test_ids["ids"].values]
X_validation = X.loc[validation_ids["ids"].values]
X_training = X.loc[training_ids]
print("\n")
print("Number of rows in X_training ", X_training.shape[0])
print("Number of rows in X_validation", X_validation.shape[0])
print("Number of rows in X_test", X_test.shape[0])

