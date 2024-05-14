import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf

def create_lags(df, lag):
    # Δημιουργία των lags για τα χαρακτηριστικά
    cols_to_shift = [col for col in df.columns if (col != 'timestamp' and col != 'label')]
    for col in cols_to_shift:
        for i in range(1, lag + 1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    df.fillna(0, inplace=True)  # Αντικατάσταση των τιμών NaN με 0
    return df

# Ορισμός του φακέλου που περιέχει τα αρχεία CSV
path = r"C:\Users\me\PycharmProjects\pythonProject2\harth"

accuracies = []  # Λίστα για την αποθήκευση των ακριβειών από κάθε αρχείο

# Διάβασμα των αρχείων στο φάκελο
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Διάβασμα του CSV αρχείου
        df = pd.read_csv(os.path.join(path, filename))

        # Δημιουργία των lags
        df = create_lags(df, 50)

        # Διαχωρισμός σε train και test sets
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        # Προετοιμασία των δεδομένων εκπαίδευσης
        X_train = train_df.drop(['timestamp', 'label'], axis=1).values
        y_train = train_df['label'].values

        # Προετοιμασία των δεδομένων ελέγχου
        X_test = test_df.drop(['timestamp', 'label'], axis=1).values
        y_test = test_df['label'].values

        # Ορισμός του μοντέλου Bayesian Neural Network
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(141, activation='softmax')
        ])

        # Σύνταξη του μοντέλου
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Εκπαίδευση του μοντέλου
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Αξιολόγηση του μοντέλου
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)

        # Εκτύπωση της ακρίβειας για κάθε αρχείο
        print(f"Ακρίβεια για το αρχείο {filename}: {accuracy}")

# Υπολογισμός της μέσης ακρίβειας
mean_accuracy = np.mean(accuracies)
print(f"Μέση ακρίβεια για όλα τα αρχεία: {mean_accuracy}")
