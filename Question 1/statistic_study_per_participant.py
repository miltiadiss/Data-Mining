import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Χρησιμοποιήστε μια παλέτα χρωμάτων για συνέπεια στα χρώματα
colors = sns.color_palette("tab10")  # 10 διαφορετικά χρώματα
# Ορίστε τον φάκελο που περιέχει τα αρχεία CSV
path = r"/harth"
output_folder = r"C:\Users\me\PycharmProjects\pythonProject2\participant_statistics"
# Ορίστε το μονοπάτι του αρχείου στατιστικών
statistics_summary_file = r"/statistics_summary.csv"
# Διάβασε το αρχείο στατιστικών
statistics_df = pd.read_csv(statistics_summary_file)
os.chdir(path)

# Δημιουργία φακέλου εξόδου αν δεν υπάρχει
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Δημιουργία λεξικού με κενά λεξικά ως τιμές για κάθε διακριτή τιμή του label
mean_value_dictionary = {}
variance_dictionary = {}
std_deviation_dictionary = {}
max_values = {}
min_values = {}
total_counts = {column: 0 for column in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']}

# Πραγματοποίηση πέρασματος στο φάκελο για τον εντοπισμό όλων των διακριτών τιμών του label
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, filename))

        for column in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']:
            total_counts[column] += df[column].count()

        for column in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']:
            mean_value = df[column].mean()
            variance_value = df[column].var()
            std_deviation_value = df[column].std()
            mean_value_dictionary[column] = round(mean_value, 6)  # Κρατάμε 6 δεκαδικά ψηφία
            variance_dictionary[column] = round(variance_value, 6)
            std_deviation_dictionary[column] = round(std_deviation_value, 6)
            max_value = df[column].max()
            min_value = df[column].min()
            max_values[column] = round(max_value, 6)
            min_values[column] = round(min_value, 6)

            # Δημιουργία αρχείου CSV με το ίδιο όνομα με το αρχικό αρχείο
            file_name = os.path.splitext(filename)[0]  # Αφαίρεση της κατάληξης .csv
            output_file = os.path.join(output_folder, f"{file_name}.csv")
            df_stats = pd.DataFrame({
                'Feature': [column],
                'Count': [total_counts[column]],
                'Min': [min_values[column]],
                'Max': [max_values[column]],
                'Mean': [mean_value_dictionary[column]],
                'Variance': [variance_dictionary[column]],
                'Standard Deviation': [std_deviation_dictionary[column]],
            })
            if os.path.exists(output_file):
                df_stats.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df_stats.to_csv(output_file, index=False)

# Λίστα για τα ονόματα των συμμετεχόντων χωρίς την κατάληξη .csv
file_names = []

# Λεξικό για τις μέσες τιμές των γνωρισμάτων
mean_values = {
    'back_x': [],
    'back_y': [],
    'back_z': [],
    'thigh_x': [],
    'thigh_y': [],
    'thigh_z': []
}

# Πραγματοποίηση πέρασματος σε όλα τα αρχεία CSV
for filename in os.listdir(output_folder):
    if filename.endswith('.csv'):
        # Χωρίς κατάληξη
        base_name = os.path.splitext(filename)[0]
        file_names.append(base_name)
        df = pd.read_csv(os.path.join(output_folder, filename))
        for column in mean_values.keys():
            mean_value = df[df['Feature'] == column]['Mean'].values[0]
            mean_values[column].append(mean_value)


# Δημιουργία γραφήματος
plt.figure(figsize=(10, 6))
for i, (column, color) in enumerate(zip(mean_values.keys(), colors)):
    plt.plot(file_names, mean_values[column], label=column, color=color)

# Προσθέστε τα σημεία δεδομένων από το "statistics_summary.csv" και χρησιμοποιήστε αντίστοιχα χρώματα
statistics_mean = statistics_df['Mean'].tolist()

# Προσθέστε οριζόντια γραμμή με αντίστοιχο χρώμα για κάθε γνώρισμα
for i, (mean, color) in enumerate(zip(statistics_mean, colors)):
    plt.axhline(y=mean, color=color, linestyle='--', linewidth=1, label=f'Horizontal Line {i+1}')

plt.title('Mean Values of Features Across Participants')
plt.xlabel('Participants')
plt.ylabel('Mean Value')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Λίστα για τα ονόματα των συμμετεχόντων χωρίς την κατάληξη .csv
file_names = []

# Σχεδιάζουμε τη χρονοσειρά για κάθε αρχείο CSV
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    df = pd.read_csv(file_path)
    # Χωρίς κατάληξη
    base_name = os.path.splitext(filename)[0]

    # Μετατροπή του timestamp σε datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Δημιουργία γραφήματος για τα γνωρίσματα
    plt.figure(figsize=(12, 6))

    # Σχεδιάζουμε τη χρονοσειρά για κάθε γνώρισμα
    for column in ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']:
        plt.plot(df['timestamp'], df[column], label=column)

    # Διαμόρφωση γραφήματος
    plt.title(f"Time Series for {base_name}")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Εμφάνιση γραφήματος
    plt.show()
