import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def create_lags(df, lag):
    cols_to_shift = [col for col in df.columns if (col != 'timestamp' and col != 'label')]
    for col in cols_to_shift:
        for i in range(1, lag + 1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    df.fillna(0, inplace=True)  # Replace NaN values with 0
    return df


# Define the folder containing the CSV files
path = '/content/harth'
output_csv_path = '/content/output_labels.csv'

accuracies = []  # List to store accuracies from each file

# Loop through the files in the folder
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Load the CSV file
        df = pd.read_csv(os.path.join(path, filename))

        # Create lag features
        df = create_lags(df, 50)

        # Split into train and test sets
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df.drop(['timestamp', 'label'], axis=1)
        y_train = train_df['label']
        X_test = test_df.drop(['timestamp', 'label'], axis=1)
        y_test = test_df['label']

        # Encode labels
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        # Initialize ANN
        ann = tf.keras.models.Sequential()
        # Adding first hidden layer
        ann.add(tf.keras.layers.Dense(units=50, activation="relu", input_shape=(X_train.shape[1],)))
        # Adding second hidden layer
        ann.add(tf.keras.layers.Dense(units=50, activation="relu"))
        # Adding output layer
        ann.add(tf.keras.layers.Dense(units=y_train.shape[1], activation="softmax"))

        # Compiling ANN
        ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

        # Fitting ANN
        ann.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

        # Making predictions
        y_pred = ann.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Calculate accuracy and other metrics
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

        # Append accuracy to the list
        accuracies.append(accuracy)

        # Print metrics for each file
        print(f"File: {filename}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

# Calculate and print mean accuracy
mean_accuracy = sum(accuracies) / len(accuracies)
print(f"Mean Accuracy: {mean_accuracy}")
