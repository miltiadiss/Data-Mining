import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB


def create_lags(df, lag):
    # Επιλογή των στηλών που θα καθυστερήσουν (αφαιρούμε τις άχρηστες στήλες 'timestamp' και 'label')
    cols_to_shift = [col for col in df.columns if (col != 'timestamp' and col != 'label')]
    # Δημιουργία ενός λεξικού που περιέχει τις καθυστερημένες στήλες
    lagged_data = {f'{col}_lag_{i}': df[col].shift(i) for col in cols_to_shift for i in range(1, lag + 1)}
    # Δημιουργία ενός νέου DataFrame από το λεξικό με τα καθυστερημένα δεδομένα
    lagged_df = pd.DataFrame(lagged_data)
    # Συνένωση του αρχικού DataFrame με το DataFrame που περιέχει τα καθυστερημένα δεδομένα
    df = pd.concat([df, lagged_df], axis=1)
    # Αντικατάσταση των τιμών NaN που δημιουργήθηκαν από τις καθυστερήσεις με 0
    df.fillna(0, inplace=True)
    return df


# Ορισμός του φακέλου που περιέχει τα αρχεία CSV
path = r"C:\Users\chryssa_pat\PycharmProjects\data_mining\harth"

accuracies = []  # Λίστα για την αποθήκευση των ακρίβειων από κάθε αρχείο

# Διάβασμα των αρχείων στο φάκελο
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Διάβασμα του CSV αρχείου
        df = pd.read_csv(os.path.join(path, filename))

        # Δημιουργία καθυστερημένων χαρακτηριστικών
        df = create_lags(df, 50)

        # Διαχωρισμός σε σύνολα εκπαίδευσης και δοκιμών
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df.drop(['timestamp', 'label'], axis=1)
        y_train = train_df['label']
        X_test = test_df.drop(['timestamp', 'label'], axis=1)
        y_test = test_df['label']

        # Κανονικοποίηση των δεδομένων
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Εκπαίδευση του μοντέλου Gaussian Naive Bayes
        gnb = GaussianNB()
        gnb.fit(X_train_scaled, y_train)

        # Πρόβλεψη των ετικετών για το σύνολο δοκιμών
        y_pred = gnb.predict(X_test_scaled)


        # Υπολογισμός ακρίβειας και άλλων μετρικών
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Προσθήκη της ακρίβειας στη λίστα
        accuracies.append(accuracy)

        # Εκτύπωση μετρικών για κάθε αρχείο
        print(f"File: {filename}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

# Υπολογισμός και εκτύπωση της μέσης ακρίβειας
mean_accuracy = sum(accuracies) / len(accuracies)
print(f"Mean Accuracy: {mean_accuracy}")
