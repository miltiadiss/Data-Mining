import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from collections import Counter

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

# Λίστα για την αποθήκευση των inertia για κάθε αριθμό συστάδων
inertia = []

# Δοκιμάζουμε διαφορετικούς αριθμούς συστάδων
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(all_data_scaled)
    inertia.append(gmm.bic(all_data_scaled))  # Χρησιμοποιούμε το Bayesian Information Criterion (BIC) ως μέτρο αξιολόγησης

# Οπτικοποίηση του Elbow Method
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('BIC')
plt.title('Elbow Method using BIC')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
"""
# Εφαρμογή του GMM
n_components = 5  # Πλήθος συστάδων
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(all_data_scaled)
labels = gmm.predict(all_data_scaled)

# Λίστα για την αποθήκευση των αποτελεσμάτων ανά συμμετέχοντα
cluster_assignments = []

# Προβλέψεις για κάθε συμμετέχοντα
for i, X in enumerate(all_data):
    # Κανονικοποίηση των δεδομένων του συμμετέχοντα
    X_scaled = scaler.transform(X)

    # Προβλέψεις των συστάδων
    labels = gmm.predict(X_scaled)

    # Βρίσκουμε την επικρατέστερη συστάδα για τον συμμετέχοντα
    most_common_cluster = Counter(labels).most_common(1)[0][0]

    # Αποθήκευση των αποτελεσμάτων
    cluster_assignments.append({
        'filename': file_names[i],
        'most_common_cluster': most_common_cluster
    })

# Οπτικοποίηση των συστάδων και των σημείων των συμμετεχόντων
plt.figure(figsize=(12, 8))

# Δημιουργία χρωματικής παλέτας για τις συστάδες
unique_labels = set(labels)
colors = plt.cm.get_cmap('tab10', len(unique_labels)).colors

# Δημιουργία κενών λιστών για τις συντεταγμένες x και y
x_coords = []
y_coords = []

# Προσθήκη των σημείων στις συντεταγμένες ανά συστάδα
for i, assignment in enumerate(cluster_assignments):
    cluster = assignment['most_common_cluster']
    # Δημιουργία τυχαίων συντεταγμένων για κάθε συστάδα
    x_coords.append(
        cluster + np.random.rand() * 0.5 - 0.25)  # Προσθέτει μικρή τυχαία μετατόπιση για καλύτερη οπτικοποίηση
    y_coords.append(np.random.rand() * 0.5 - 0.25)  # Προσθέτει μικρή τυχαία μετατόπιση για καλύτερη οπτικοποίηση
    plt.scatter(x_coords[-1], y_coords[-1], s=100, color=colors[cluster % len(colors)])

# Προσθήκη ετικετών στα σημεία για να δείχνουν το όνομα του αρχείου
for i, assignment in enumerate(cluster_assignments):
    plt.text(x_coords[i], y_coords[i], assignment['filename'], fontsize=9, ha='right')

# Προσθήκη υπομνήματος
for cluster in unique_labels:
    plt.scatter([], [], color=colors[cluster % len(colors)], label=f'Cluster {cluster + 1}')

plt.title('Clustering of Participants Based on Activities')
plt.axis('off')  # Απενεργοποίηση των αξόνων
plt.legend()
plt.show()
"""
