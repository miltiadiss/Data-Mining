import os
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np

def create_lags(df, lag):
    cols_to_shift = [col for col in df.columns if (col != 'timestamp' and col != 'label')]
    lagged_columns = {}
    for col in cols_to_shift:
        lagged_columns[col] = [df[col].shift(i) for i in range(1, lag + 1)]
    lagged_df = pd.DataFrame(lagged_columns)
    df = pd.concat([df, lagged_df], axis=1)
    df.fillna(0, inplace=True)  # Replace NaN values with 0
    return df

def compute_cpt_values(variable, parents, dataframes):
    num_states = len(dataframes[0][variable].unique())
    num_parent_states = [len(dataframes[0][parent].unique()) for parent in parents]
    cpt_values = {}

    for df in dataframes:
        for i, row in df.iterrows():
            current_value = row[variable]
            parent_values = [row[parent] for parent in parents]
            parent_index = tuple([int(val) for val in parent_values])  # Convert to integers

            if current_value not in cpt_values:
                cpt_values[current_value] = {}
            parent_key = tuple(parent_index)
            if parent_key not in cpt_values[current_value]:
                cpt_values[current_value][parent_key] = 0
            cpt_values[current_value][parent_key] += 1

    return cpt_values

# Define the folder containing the CSV files
path = r"C:\Users\chryssa_pat\PycharmProjects\data_mining\harth"

# Load all CSV files into a list of dataframes
dataframes = []
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, filename))
        dataframes.append(df)

# Define lag
lag = 50

# Define variables and create the Bayesian Network
variables = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
model = BayesianModel()
model.add_nodes_from(variables)
for i in range(len(variables) - 1):
    model.add_edge(variables[i], variables[i+1])

# Create lag features and compute conditional probability tables
for variable in variables:
    parents = model.get_parents(variable)
    if not parents:
        cpd = TabularCPD(variable=variable, variable_card=2, values=[[0.5], [0.5]])
    else:
        cpd_values = compute_cpt_values(variable, parents, dataframes)
        cpd = TabularCPD(variable=variable, variable_card=2,
                         values=cpd_values, evidence=parents, evidence_card=[2]*len(parents))
    model.add_cpds(cpd)

# Inference can be performed here if needed

# Check the model for consistency
assert model.check_model()
