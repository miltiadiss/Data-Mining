import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer

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

# Μονοπάτι για τον φάκελο που περιέχει τα αρχεία CSV
path = '/content/drive/MyDrive/harth'
# Δημιουργία νέου φακέλου για τις εικόνες των confusion matrices
output_folder = '/content/drive/MyDrive/harth_confusion_matrices'
os.makedirs(output_folder, exist_ok=True)

def discretize_columns(df, bins):
    cols_to_discretize = [col for col in df.columns if col not in ['timestamp', 'label']]
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    df[cols_to_discretize] = discretizer.fit_transform(df[cols_to_discretize])
    return df

# Λίστες για την αποθήκευση των μετρικών από κάθε αρχείο
metrics_data = []

# Επανάληψη μέσω των αρχείων στο φάκελο
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Φόρτωση του CSV αρχείου
        df = pd.read_csv(os.path.join(path, filename))

        # Δημιουργία των καθυστερήσεων
        df = create_lags(df, 10)

        # Διακριτοποίηση των χαρακτηριστικών
        df = discretize_columns(df, bins=3)

        # Διαίρεση σε train και test σύνολα
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df.drop(['timestamp', 'label'], axis=1)
        y_train = train_df['label']
        X_test = test_df.drop(['timestamp', 'label'], axis=1)
        y_test = test_df['label']

        # Συνδυασμός X_train και y_train για δημιουργία DataFrame για εκπαίδευση του Bayesian Network
        train_df = pd.concat([X_train, y_train], axis=1)

        # Δημιουργία δομής του Bayesian Network με χρήση του αλγορίθμου HillClimbSearch και της μετρικής BIC
        hc = HillClimbSearch(train_df)
        best_model = hc.estimate(scoring_method=BicScore(train_df))

        # Δημιουργία του μοντέλου Bayesian Network με βάση τις βέλτιστες συνδέσεις που προέκυψαν από τον αλγόριθμο HillClimbSearch
        model = BayesianNetwork(best_model.edges())
        model.fit(train_df, estimator=MaximumLikelihoodEstimator)

        # Απόδοση του μοντέλου
        infer = VariableElimination(model)
        y_pred = []

        for _, row in X_test.iterrows():
            evidence = row.to_dict()
            query_result = infer.map_query(variables=['label'], evidence=evidence)
            y_pred.append(query_result['label'])

        # Υπολογίζουμε τις μετρικες
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Υπολογισμός και αποθήκευση του confusion matrix
        unique_labels = sorted(df['label'].unique())

        # Προσθήκη των μετρικών και του confusion matrix στη λίστα
        base_name = os.path.splitext(filename)[0]  # Χωρίς κατάληξη
        participant_metrics = {
            'Participant': base_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        }
        metrics_data.append(participant_metrics)

        # Αποθήκευση του confusion matrix ως εικόνα στο νέο φάκελο
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        cm_filename = os.path.splitext(filename)[0] + '_confusion_matrix.png'
        cm_filepath = os.path.join(output_folder, cm_filename)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(cm_filepath)
        plt.close()

# Δημιουργία DataFrame από τη λίστα με τα δεδομένα μετρικών
metrics_df = pd.DataFrame(metrics_data)

# Αποθήκευση του DataFrame σε αρχείο CSV
metrics_df.to_csv('participant_metrics.csv', index=False)

# Υπολογισμός της μέσης τιμής για κάθε μετρική από όλους τους συμμετέχοντες
bn_mean_accuracy = metrics_df['Accuracy'].mean()
bn_mean_precision = metrics_df['Precision'].mean()
bn_mean_recall = metrics_df['Recall'].mean()
bn_mean_f1 = metrics_df['F1-Score'].mean()

print("Mean Metrics of All Participants:")
print(f"  Mean Accuracy: {bn_mean_accuracy}")
print(f"  Mean Precision: {bn_mean_precision}")
print(f"  Mean Recall: {bn_mean_recall}")
print(f"  Mean F1-Score: {bn_mean_f1}")
