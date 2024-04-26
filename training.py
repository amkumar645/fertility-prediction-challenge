import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import submission

"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    random.seed(1) # not useful here because logistic regression deterministic
    train_df = pd.read_csv(cleaned_df)
    cleaned_df = submission.clean_df(train_df, None)
    outcome = pd.read_csv(outcome_df)
    outcome_supervised = outcome.dropna()['new_child']

    X = cleaned_df
    y = outcome_supervised
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, "model.joblib")

if __name__ == "__main__":
    train_save_model("PreFerData/training_data/PreFer_train_data.csv", "PreFerData/training_data/PreFer_train_outcome.csv")