import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ορίζει τον φάκελο που περιέχει τα αρχεία CSV
path = r"C:\Users\me\PycharmProjects\pythonProject2\harth"
os.chdir(path)

# Αρχικοποίηση λεξικών για τα στατιστικά στοιχεία
mean_value_dictionary = {}
variance_dictionary = {}
std_deviation_dictionary = {}
max_values = {}
min_values = {}

# Πραγματοποίηση πέρασματος σε κάθε αρχείο CSV
dfs = []
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, filename))
        dfs.append(df)

        # Υπολογισμός των στατιστικών στοιχείων για κάθε γνώρισμα
        for column in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']:
            mean_value = df[column].mean()
            variance_value = df[column].var()
            std_deviation_value = df[column].std()
            max_value = df[column].max()
            min_value = df[column].min()

            # Αποθήκευση των στατιστικών στοιχείων στα αντίστοιχα λεξικά
            if column not in mean_value_dictionary:
                mean_value_dictionary[column] = []
                variance_dictionary[column] = []
                std_deviation_dictionary[column] = []
                max_values[column] = []
                min_values[column] = []
            mean_value_dictionary[column].append(round(mean_value, 6))
            variance_dictionary[column].append(round(variance_value, 6))
            std_deviation_dictionary[column].append(round(std_deviation_value, 6))
            max_values[column].append(round(max_value, 6))
            min_values[column].append(round(min_value, 6))

# Δημιουργία DataFrame με τα στατιστικά στοιχεία για κάθε γνώρισμα
df_stats = pd.DataFrame({
    'Feature': ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z'],
    'Mean': [sum(mean_value_dictionary[col]) / len(mean_value_dictionary[col]) for col in
             ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
    'Variance': [sum(variance_dictionary[col]) / len(variance_dictionary[col]) for col in
                 ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
    'Standard Deviation': [sum(std_deviation_dictionary[col]) / len(std_deviation_dictionary[col]) for col in
                           ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
    'Max': [max(max_values[col]) for col in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
    'Min': [min(min_values[col]) for col in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']],
})

# Εξαγωγή του DataFrame σε αρχείο CSV
output_file = "statistics_summary.csv"
df_stats.to_csv(output_file, index=False)

# Συγχώνευση όλων των DataFrame σε ένα
merged_df = pd.concat(dfs, ignore_index=True)

# Αναπαράσταση γραφικών της συχνότητας των τιμών για τις 6 στήλες
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Appearance Frequency Distributions of Features', fontsize=10)

for ax, column in zip(axes.flatten(), ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']):
    ax.hist(merged_df[column], bins=35, color='skyblue', edgecolor='black', density=True)
    ax.set_title(f'{column}', fontsize=10)
    ax.set_ylabel('Density', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True)

    mu, sigma = merged_df[column].mean(), merged_df[column].std()
    xmin, xmax = merged_df[column].min(), merged_df[column].max()
    x = np.linspace(xmin, xmax, 50000)
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    ax.plot(x, p, 'k', linewidth=1)

plt.tight_layout()
plt.show()

# Δημιουργία του heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(merged_df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Feature Correlations')
plt.show()

# Σχεδίαση των γραφικών για τα 4 ζευγάρια γνωρισμάτων
plt.figure(figsize=(12, 8))

# Ζευγάρι 1: thigh_x και back_x
plt.scatter(merged_df['thigh_x'], merged_df['back_x'], alpha=0.5)
plt.xlabel('thigh_x')
plt.ylabel('back_x')
plt.title('Correlation between thigh_x and back_x')
plt.show()

# Ζευγάρι 2: thigh_x και thigh_z
plt.figure(figsize=(12, 8))
plt.scatter(merged_df['thigh_x'], merged_df['thigh_z'], alpha=0.5)
plt.xlabel('thigh_x')
plt.ylabel('thigh_z')
plt.title('Correlation between thigh_x and thigh_z')
plt.show()

# Ζευγάρι 3: back_y και thigh_y
plt.figure(figsize=(12, 8))
plt.scatter(merged_df['back_y'], merged_df['thigh_y'], alpha=0.5)
plt.xlabel('back_y')
plt.ylabel('thigh_y')
plt.title('Correlation between thigh_y and back_y')
plt.show()

# Ζευγάρι 4: back_z και thigh_z
plt.figure(figsize=(12, 8))
plt.scatter(merged_df['back_z'], merged_df['thigh_z'], alpha=0.5)
plt.xlabel('back_z')
plt.ylabel('thigh_z')
plt.title('Correlation between thigh_z and back_z')
plt.show()
