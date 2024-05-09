import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ορίστε τον φάκελο που περιέχει τα αρχεία CSV
path = r"C:\Users\me\PycharmProjects\pythonProject2\harth"
output_csv_path = r"C:\Users\me\PycharmProjects\pythonProject2\output_predictions.csv"

# Φορτώνουμε το CSV αρχείο
df = pd.read_csv(os.path.join(path, "S024.csv"), parse_dates=['timestamp'])

# Εξασφαλίζουμε ότι οι στήλες που χρησιμοποιούνται ως χαρακτηριστικά είναι αριθμητικές
numeric_columns = df.select_dtypes(include=['float', 'int']).columns
non_numeric_columns = df.select_dtypes(exclude=['float', 'int']).columns

# Χρησιμοποιούμε LabelEncoder για τις μη αριθμητικές στήλες
for col in non_numeric_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Διαχωρίζουμε το σύνολο δεδομένων σε σύνολα εκπαίδευσης και δοκιμής
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Δημιουργία και εκπαίδευση του μοντέλου Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Πρόβλεψη στο σύνολο δοκιμής
y_pred = rf_classifier.predict(X_test)

# Αξιολόγηση του μοντέλου
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Δημιουργία DataFrame για τις προβλεπόμενες και πραγματικές τιμές
comparison_df = pd.DataFrame({
    'Actual': y_test.values,  # Πραγματικές τιμές
    'Predicted': y_pred  # Προβλεπόμενες τιμές
})

# Αποθήκευση του DataFrame σε αρχείο CSV
comparison_df.to_csv(output_csv_path, index=False)

# Εκτύπωση των στατιστικών του μοντέλου
print("Προβλέψεις και Πραγματικές τιμές αποθηκεύτηκαν στο CSV.")
print(f"Απόδοση του μοντέλου Random Forest:")
print("Ακρίβεια:", accuracy)
print("Προσέγγιση:", precision)
print("Ανάκληση:", recall)
print("F1-βαθμολογία:", f1)
