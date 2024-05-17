import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Ορισμός του φακέλου που περιέχει τα αρχεία CSV
path = r"C:\Users\chryssa_pat\PycharmProjects\data_mining\harth"

# Λίστες για την αποθήκευση των δεδομένων από όλα τα αρχεία
all_data = []

# Διάβασμα των αρχείων στο φάκελο
file_names = []  # Λίστα με τα ονόματα των αρχείων
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Διάβασμα του CSV αρχείου
        df = pd.read_csv(os.path.join(path, filename))
        file_names.append(filename)  # Κρατάμε το όνομα του αρχείου για κάθε δείγμα

        # Επιλογή των συγκεκριμένων χαρακτηριστικών
        X = df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].values

        # Αποθήκευση των δεδομένων στη λίστα
        all_data.append(X)

# Συνδυασμός όλων των δεδομένων σε ένα συνολικό πίνακα
all_data_combined = np.concatenate(all_data, axis=0)

# Κανονικοποίηση των δεδομένων
scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data_combined)

# Εφαρμογή του αλγορίθμου Agglomerative Clustering
n_clusters = 3  # Ορίζουμε τον αριθμό των συστάδων
clustering = AgglomerativeClustering(n_clusters=n_clusters)
clustering.fit(all_data_scaled)

# Εκτύπωση των αποτελεσμάτων
print("Cluster Assignments:")
for filename, label in zip(file_names, clustering.labels_):
    print(f"{filename}: Cluster {label + 1}")
