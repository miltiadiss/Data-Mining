import os
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score

def create_lags(df, lag):
    cols_to_shift = [col for col in df.columns if col not in ['timestamp', 'label']]
    for col in cols_to_shift:
        for i in range(1, lag + 1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    df.fillna(0, inplace=True)  # Replace NaN values with 0
    return df

def discretize_columns(df, bins):
    cols_to_discretize = [col for col in df.columns if col not in ['timestamp', 'label']]
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    df[cols_to_discretize] = discretizer.fit_transform(df[cols_to_discretize])
    return df

# Set the folder containing the CSV files
path = '/content/harth'
accuracies = []

# Loop through the files in the folder
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Load the CSV file
        df = pd.read_csv(os.path.join(path, filename))

        # Create the lags
        df = create_lags(df, 10)

        # Discretize the features
        df = discretize_columns(df, bins=3)

        # Split into train and test sets
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df.drop(['timestamp', 'label'], axis=1)
        y_train = train_df['label']
        X_test = test_df.drop(['timestamp', 'label'], axis=1)
        y_test = test_df['label']

        # Combine X_train and y_train to create a DataFrame for training the Bayesian Network
        train_df = pd.concat([X_train, y_train], axis=1)

        # Create Bayesian Network structure
        hc = HillClimbSearch(train_df)
        best_model = hc.estimate(scoring_method=BicScore(train_df))

        model = BayesianNetwork(best_model.edges())
        model.fit(train_df, estimator=MaximumLikelihoodEstimator)

        # Inference
        infer = VariableElimination(model)
        y_pred = []

        for _, row in X_test.iterrows():
            evidence = row.to_dict()
            query_result = infer.map_query(variables=['label'], evidence=evidence)
            y_pred.append(query_result['label'])

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Print the accuracy for each file
        print(f"Accuracy for file {filename}: {accuracy}")

# Calculate and print the mean accuracy
mean_accuracy = sum(accuracies) / len(accuracies)
print(f"Mean accuracy: {mean_accuracy}")
