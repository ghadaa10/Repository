from use_case import BethEntry

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import f1_score
import tqdm


def data_analysis(training,validation):
    print("Data training Overview:")
    print(training.head())  # Show the first few rows of the data
    print("\nData training Description:")
    print(training.describe())  # Statistical summary of numerical columns
    print("\nData training Information:")
    print(training.info())  # Information about the dataset, including column types and non-null counts

    # Check for missing values
    print("\nMissing Values in Each Column:")
    print(training.isnull().sum())
    y= training['sus']
    print("Distribution of sus Variable:")
    print(y.value_counts())
    plt.figure(figsize=(6, 6))
    y.value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], labels=['Class 0', 'Class 1'], startangle=90)
    plt.title('Distribution of sus Variable')
    plt.ylabel('')  
    plt.show()
    y= training['evil']
    print("Distribution of evil Variable:")
    print(y.value_counts())
    sns.countplot(x=y)
    plt.title('Distribution of evil Variable')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

    print("Data validation Overview:")
    print(validation.head())  # Show the first few rows of the data
    print("\nData validation Description:")
    print(validation.describe())  # Statistical summary of numerical columns
    print("\nData validation Information:")
    print(validation.info())  # Information about the dataset, including column types and non-null counts

    # Check for missing values
    print("\nMissing Values in Each Column:")
    print(validation.isnull().sum())
    y= validation['sus']
    print("Distribution of sus Variable:")
    print(y.value_counts())
    plt.figure(figsize=(6, 6))
    y.value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], labels=['Class 0', 'Class 1'], startangle=90)
    plt.title('Distribution of sus Variable')
    plt.ylabel('')  
    plt.show()
    y= validation['evil']
    print("Distribution of evil Variable:")
    print(y.value_counts())
    sns.countplot(x=y)
    plt.title('Distribution of evil Variable')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
    
    
    

df = pd.read_csv('data/training.csv')  
df_test = pd.read_csv('data/validation.csv')  

import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.circuit.library import RawFeatureVector
from sklearn.model_selection import train_test_split


algorithm_globals.random_seed = 42

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import Sampler
from sklearn.feature_selection import SelectKBest, f_classif

#preprocessing
label_encoder = LabelEncoder()

df_train = df.apply(label_encoder.fit_transform)
df_test = df_test.apply(label_encoder.fit_transform)

fraction_of_data = 0.02  # 20% of the data
df_train= df_train.sample(frac=fraction_of_data, random_state=42)


X = df_train.drop(columns=['sus','evil'])  # Features
y = df_train['sus']  # Target column

X_test = df_test.drop(columns=['sus','evil'])  # Features
y_test = df_test['sus']  # Target column

#Select best features
selector = SelectKBest(score_func=f_classif, k=5)
X= selector.fit_transform(X, y)

num_features = 5

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", style="clifford", fold=20)

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", style="clifford", fold=20)

optimizer = COBYLA(maxiter=20)

sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
    

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []
start = time.time()
vqc.fit(np.array(X), np.array(y))
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")
X_test= selector.fit_transform(X_test, y_test)
y_pred=vqc.predict(X_test)

recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro') 
print(recall,precision,f1)

#data_analysis(df,df_test)
