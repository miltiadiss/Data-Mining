import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Ορισμός του φακέλου που περιέχει τα αρχεία CSV
path = r"C:\Users\chryssa_pat\PycharmProjects\data_mining\harth"

# Λίστες για την αποθήκευση των δεδομένων από όλα τα αρχεία
all_data = []
# Λίστα για τα ονόματα των αρχείων
file_names = []
# Λίστες για τις μέσες τιμές από κάθε αρχείο
mean_values_list = []

# Διάβασμα των αρχείων στο φάκελο
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Διάβασμα του CSV αρχείου
        df = pd.read_csv(os.path.join(path, filename))
        file_names.append(filename)  # Κρατάμε το όνομα του αρχείου για κάθε δείγμα

        # Επιλογή των συγκεκριμένων χαρακτηριστικών
        columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
        X = df[columns]

        # Αποθήκευση των δεδομένων στη λίστα
        all_data.append(X)

        # Υπολογισμός μέσης τιμής κάθε στήλης
        mean_values = X.mean().values

        # Προσθήκη των μέσων τιμών στη λίστα
        mean_values_list.append(mean_values)

# Δημιουργία DataFrame με τα ονόματα των αρχείων ως δείκτες και τις μέσες τιμές ως δεδομένα
mean_values_df = pd.DataFrame(mean_values_list, columns=columns, index=file_names)

# Εμφάνιση του DataFrame
print(mean_values_df)

# Κανονικοποίηση των μέσων τιμών
scaler = StandardScaler()
mean_values_scaled = scaler.fit_transform(mean_values_df)

# Εφαρμογή του αλγορίθμου k-means στα κανονικοποιημένα δεδομένα
kmeans = KMeans(n_clusters=4, random_state=42)  # Ορισμός αριθμού συστάδων και τυχαίας κατάστασης για επαναληψιμότητα
kmeans.fit(mean_values_scaled)

# Προβλέψεις για τις συστάδες των μέσων τιμών
cluster_assignments = kmeans.predict(mean_values_scaled)

# Δημιουργία DataFrame με τα αρχεία και τις αντίστοιχες συστάδες
clusters_df = pd.DataFrame({'filename': file_names, 'cluster': cluster_assignments})

# Ομαδοποίηση και εκτύπωση των αρχείων ανά συστάδα
grouped = clusters_df.groupby('cluster')['filename'].apply(list)

for cluster, files in grouped.items():
    print(f"Cluster {cluster}:")
    for file in files:
        print(f"  {file}")
