# Zack Abdirashid
# NBA Predictor that uses scikit-learn to predit from the nba_games.csv sheet of games from 2015-2022

import pandas as pd


df = pd.read_csv("nba_games.csv", index_col=0)
df = df.sort_values("date")
df = df.reset_index(drop=True)

# removing extra columns
del df["mp.1"]
del df["mp_opp.1"]
del df ["index_opp"]

# figuring out how a team did in their next game
def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team

df = df.groupby("team", group_keys=False).apply(add_target)

# replacing NaN values in a teams row (some teams won't have a next game)
df["target"][pd.isnull(df["target"])] = 2 

# converting from a bool to an int
df["target"] = df["target"].astype(int, errors="ignore")

nulls = pd.isnull(df)
nulls = nulls[nulls > 0]

valid_columns = df.columns[~df.columns.isin(nulls.index)] # geting rid of all the columns that are null

df = df[valid_columns].copy()

from sklearn.model_selection import TimeSeriesSplit # will help split the data nicely when feature selecting
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier # helps classify if a team will win/lose their next game

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)

removed_columns = ["season", "date", "won", "target", "team", "team_opp"]

selected_columns = df.columns[~df.columns.isin(removed_columns)]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns]) # rescaleing values between 0 and 1
df[selected_columns] = df[selected_columns].fillna(0) # converting any leftover NaN values to 0

sfs.fit(df[selected_columns], df["target"])
predictors = list(selected_columns[sfs.get_support()]) 

def backtest(data, model, predictors, start=2, step=1): # splits data up by seasons and will initially predict a season based off of 2 past seasons
    all_predictions = [] # dataframes that are predictions of a single szn
    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i] 
        train = data[data["season"] < season] # all data that comes before a season
        test = data[data["season"] == season] # used to generate predictions

        model.fit(train[predictors], train["target"])
        predictions = model.predict(test[predictors]) 
        predictions = pd.Series(predictions, index=test.index) # converting from an np array to a series
        combined = pd.concat([test["target"], predictions], axis=1) # combining the two datasets in separate columns
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)

predictions = backtest(df, rr, predictors) 

from sklearn.metrics import accuracy_score

accuracy_score(predictions["actual"], predictions["prediction"]) # testing accuracy

df.groupby("home").apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])

df_rolling = df[list(selected_columns) + ["won", "team", "season"]]


def find_team_averages(team):
    rolling = team[selected_columns].rolling(10).mean() # how well a team has done in the last 10 games
    return rolling

df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

rolling_cols = [f"{col}_10" for col in df_rolling.columns] # making new cols with rolling average values
df_rolling.columns = rolling_cols

df = pd.concat([df, df_rolling], axis=1)
df = df.dropna() # dropping any rows with missing values

def shift_column(team, col_name): #
    next_col = team[col_name].shift(-1)
    return next_col

def add_column(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_column(x, col_name))

# help the algorithm factor in whether the next game is home/away
df["home_next"] = add_column(df, "home") 
df["team_opp_next"] = add_column(df, "team_opp")
df["date_next"] = add_column(df, "date")


full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], 
                left_on=["team", "date_next"], 
                right_on=["team_opp_next", "date_next"])

# making sure we only include numeric types in the machine learning algorithm
removed_columns += list(full.columns[full.dtypes == "object"])

selected_columns = full.columns[~full.columns.isin(removed_columns)] # selecting cols that aren't in removed_columns
sfs.fit(full[selected_columns], full["target"])

predictors = list(selected_columns[sfs.get_support()])

predictions = backtest(full, rr, predictors)
accuracy_score(predictions["actual"], predictions["prediction"]) # Improved the starting accuracy_score of 57% to 63%





# Inspired by dataquest tutorial



