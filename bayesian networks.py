import pgmpy

# pip install pgmpy (This line is a comment, indicating the installation command)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = DiscreteBayesianNetwork([
    ('Cold', 'Cough'),  # 'Cold' influences 'Cough'
    ('Cold', 'Fever')   # 'Cold' influences 'Fever'
])

# Define the Conditional Probability Distribution (CPD) for 'Cold'
cpd_cold = TabularCPD(
    variable='Cold',
    variable_card=2,  # Binary variable: 0 for No Cold, 1 for Cold
    values=[[0.9], [0.1]]  # P(Cold=0) = 0.9, P(Cold=1) = 0.1 (prior probability)
)

# Define the CPD for 'Cough' given 'Cold'
cpd_cough_given_cold = TabularCPD(
    variable='Cough',
    variable_card=2,  # Binary variable: 0 for No Cough, 1 for Cough
    values=[[0.8, 0.1],  # P(Cough=0 | Cold=0) = 0.8, P(Cough=0 | Cold=1) = 0.1
            [0.2, 0.9]], # P(Cough=1 | Cold=0) = 0.2, P(Cough=1 | Cold=1) = 0.9
    evidence=['Cold'],
    evidence_card=[2]
)

# Define the CPD for 'Fever' given 'Cold'
cpd_fever_given_cold = TabularCPD(
    variable='Fever',
    variable_card=2,  # Binary variable: 0 for No Fever, 1 for Fever
    values=[[0.7, 0.2],  # P(Fever=0 | Cold=0) = 0.7, P(Fever=0 | Cold=1) = 0.2
            [0.3, 0.8]], # P(Fever=1 | Cold=0) = 0.3, P(Fever=1 | Cold=1) = 0.8
    evidence=['Cold'],
    evidence_card=[2]
)

# Add the CPDs to the model
model.add_cpds(cpd_cold, cpd_cough_given_cold, cpd_fever_given_cold)

# Validate the model
assert model.check_model()
print("Bayesian Network structure and CPDs have been defined and validated.")

# Create an inference object
infer = VariableElimination(model)

# Query 1: Calculate the probability of 'Cold' given 'Cough' (evidence = 1 for Cough)
query_result_1 = infer.query(variables=['Cold'], evidence={'Cough': 1})
print("P(Cold | Cough) =\n", query_result_1)

# Query 2: Calculate the probability of 'Cold' given 'Cough' and 'Fever'
query_result_2 = infer.query(variables=['Cold'], evidence={'Cough': 1, 'Fever': 1})
print("P(Cold | Cough, Fever) =\n", query_result_2)
