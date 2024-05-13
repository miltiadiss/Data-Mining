import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def create_window_data(X, y, current_timestamp, window_size):
    current_index = X[X['timestamp'] == current_timestamp].index[0]
    start_index = max(0, current_index - window_size + 1)
    end_index = current_index + 1
    return X.iloc[start_index:end_index], y.iloc[start_index:end_index]

# Ορίστε τον φάκελο που περιέχει τα αρχεία CSV
path = r"C:\Users\me\PycharmProjects\pythonProject2\harth"
output_csv_path = r"C:\Users\me\PycharmProjects\pythonProject2\output_predictions.csv"

# Φορτώστε το CSV αρχείο
df = pd.read_csv(os.path.join(path, "S024.csv"))
timestamps = []
for timestamp in df['timestamp']:
    timestamps.append(timestamp)

# Εξασφαλίστε ότι οι στήλες που χρησιμοποιούνται ως χαρακτηριστικά είναι αριθμητικές
numeric_columns = df.select_dtypes(include=['float', 'int']).columns
non_numeric_columns = df.select_dtypes(exclude=['float', 'int']).columns

# Χρησιμοποιήστε LabelEncoder για τις μη αριθμητικές στήλες
for col in non_numeric_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Δημιουργήστε τα χαρακτηριστικά και τις ετικέτες
X = df.drop(['label', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z'], axis=1)
y = df['label']

# Δημιουργία και εκπαίδευση του μοντέλου Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Πρόβλεψη για κάθε χρονική στιγμή
window_size = len(X)  # Ορίστε το μέγεθος του παραθύρου ίσο με τον αριθμό των δειγμάτων
predictions = []

for timestamp in df['timestamp']:
    X_window, _ = create_window_data(X, y, timestamp, window_size)
    prediction = rf_classifier.predict(X_window)[-1]  # Λαμβάνουμε την τελευταία πρόβλεψη
    predictions.append(prediction)

# Αποθηκεύστε τις προβλέψεις σε ένα αρχείο CSV
df['predicted_label'] = predictions
df['timestamp'] = timestamps

# Διαβάστε το αρχείο με τα αρχικά δεδομένα
original_df = pd.read_csv(os.path.join(path, "S024.csv"))
# Αντικαταστήστε τη στήλη label με τις αρχικές τιμές από το αρχικό αρχείο
df['label'] = original_df['label']

# Σχεδίαση των χρονοσειρών πραγματικού και προβλεπόμενου label στο ίδιο παράθυρο
plt.figure(figsize=(10, 6))
plt.plot(df['label'], label='Πραγματική Δραστηριότητα', color='blue')
plt.plot( df['predicted_label'], label='Προβλεπόμενη Δραστηριότητα', color='red')

plt.xlabel('Χρόνος')
plt.ylabel('Δραστηριότητα')
plt.title('Πραγματική vs Προβλεπόμενη Δραστηριότητα')
plt.legend()
plt.xticks([])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
