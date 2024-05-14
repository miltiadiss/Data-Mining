import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_lags(df, lag):
    cols_to_shift = [col for col in df.columns if (col != 'timestamp' and col != 'label')]
    for col in cols_to_shift:
        for i in range(1, lag + 1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    df.fillna(0, inplace=True)  # Αντικατάσταση των τιμών NaN με 0
    return df


# Ορίστε τον φάκελο που περιέχει τα αρχεία CSV
path = r"C:\Users\me\PycharmProjects\pythonProject2\harth"
output_csv_path = r"C:\Users\me\PycharmProjects\pythonProject2\output_labels.csv"

accuracies = []  # Λίστα για την αποθήκευση των ακριβειών από κάθε αρχείο
# Δημιουργούμε και εκπαιδεύουμε τον ταξινομητή Random Forest
rf_classifier = RandomForestClassifier(n_estimators=30, random_state=42)

# Διατρέχουμε τα αρχεία στον φάκελο
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Φορτώνουμε το CSV αρχείο
        df = pd.read_csv(os.path.join(path, filename))

        # Δημιουργούμε τα lags
        df = create_lags(df, 50)

        # Διαχωρίζουμε σε train και test sets
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df.drop(['timestamp', 'label'], axis=1)
        y_train = train_df['label']
        X_test = test_df.drop(['timestamp', 'label'], axis=1)
        y_test = test_df['label']

        rf_classifier.fit(X_train, y_train)

        # Προβλέπουμε τις ετικέτες του test set
        y_pred = rf_classifier.predict(X_test)

        # Υπολογίζουμε την ακρίβεια και την προσθέτουμε στη λίστα των ακριβειών
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Εκτυπώνουμε την ακρίβεια για κάθε αρχείο
        print(f"Ακρίβεια πρόβλεψης δραστηριότητας για συμμετέχοντα {filename}: {accuracy}")

# Υπολογισμός της μέσης ακρίβειας
mean_accuracy = sum(accuracies) / len(accuracies)
print(f"Μέση ακρίβεια πρόβλεψης δραστηριότητας: {mean_accuracy}")




