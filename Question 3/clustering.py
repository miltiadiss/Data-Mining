import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ορισμός του φακέλου που περιέχει τα αρχεία CSV
path = r"C:\Users\me\PycharmProjects\pythonProject2\harth"

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

# Εφαρμογή του αλγορίθμου k-means
kmeans = KMeans(n_clusters=3)  # Ορισμός αριθμού συστάδων
kmeans.fit(all_data_scaled)

# Αποθήκευση των κέντρων των συστάδων
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:")
print(cluster_centers)

# Διαβάζουμε τη στήλη "Mean" από κάθε αρχείο CSV
mean_values = []
path = r"C:\Users\me\PycharmProjects\pythonProject2\participant_statistics"
for filename in file_names:
    df = pd.read_csv(os.path.join(path, filename))
    mean_values.append(df['Mean'].values)

# Βρίσκουμε σε ποια συστάδα ανήκει κάθε μέση τιμή
for i, mean_value in enumerate(mean_values):
    cluster = kmeans.predict([mean_value])
    print(f"File {file_names[i]} belongs to cluster {cluster}")


