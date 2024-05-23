import os
import pandas as pd

# Ορίστε τον φάκελο που περιέχει τα αρχεία CSV
path = r"/harth"
output_folder = r"C:\Users\me\PycharmProjects\pythonProject2\activity_statistics"
os.chdir(path)

# Δημιουργία φακέλου εξόδου αν δεν υπάρχει
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Δημιουργία λεξικού με κενά λεξικά ως τιμές για κάθε διακριτή τιμή του label
mean_value_dictionary = {}
variance_dictionary = {}
std_deviation_dictionary = {}
covariance_dictionary = {}
max_values = {}
min_values = {}
total_counts = {column: 0 for column in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']}

# Πραγματοποίηση πέρασματος στο φάκελο για τον εντοπισμό όλων των διακριτών τιμών του label
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, filename))

        for column in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']:
            total_counts[column] += df[column].count()

        unique_labels = df['label'].unique()
        for label in unique_labels:
            if label not in mean_value_dictionary:
                mean_value_dictionary[label] = {}
                variance_dictionary[label] = {}
                std_deviation_dictionary[label] = {}
                covariance_dictionary[label] = {}
                max_values[label] = {}
                min_values[label] = {}
            label_df = df[df['label'] == label]
            for column in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']:
                mean_value = label_df[column].mean()
                variance_value = label_df[column].var()
                std_deviation_value = label_df[column].std()
                mean_value_dictionary[label][column] = round(mean_value, 6)  # Κρατάμε 6 δεκαδικά ψηφία
                variance_dictionary[label][column] = round(variance_value, 6)
                std_deviation_dictionary[label][column] = round(std_deviation_value, 6)
                max_value = label_df[column].max()
                min_value = label_df[column].min()
                max_values[label][column] = round(max_value, 6)
                min_values[label][column] = round(min_value, 6)

        # Δημιουργία αρχείων CSV για κάθε ετικέτα
        for label, values in mean_value_dictionary.items():
            # Δημιουργία ενός DataFrame από το λεξικό με τις μέσες τιμές, διακυμάνσεις, τυπικές αποκλίσεις, μεγίστες,
            # ελάχιστες τιμές, συνολικό αριθμό τιμών και συνδιασπορές
            df_label = pd.DataFrame({
                'Feature': ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z'],
                'Count': [total_counts[col] for col in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
                'Min': [min_values[label][col] for col in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
                'Max': [max_values[label][col] for col in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
                'Mean': [mean_value_dictionary[label][col] for col in
                         ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
                'Variance': [variance_dictionary[label][col] for col in
                             ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
                'Standard Deviation': [std_deviation_dictionary[label][col] for col in
                                       ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
            })
            # Εξαγωγή σε αρχείο CSV με το όνομα label.csv
            df_label.to_csv(f"{output_folder}/{label}.csv", index=False)
