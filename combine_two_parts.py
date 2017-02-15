import pandas
from data_cleaning import df2
from clean_team_data import X_team

print("Number of columns in the first part is ", len(df2.columns))
print("Number of columns in the second part is ", len(X_team.columns))
clean_data = df2.join(X_team)
y = clean_data["Grant.Status"]
X = clean_data.drop(["Grant.Status"], axis=1)
print("Number of columns in the complete & cleaned data sets  is ", len(clean_data.columns))
