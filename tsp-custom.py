import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from qat.opt import CombinatorialProblem
from qat.vsolve.ansatz import AnsatzFactory
from qat.pylinalg import PyLinalg
from qat.lang.AQASM import Program
from qat.qpus import get_default_qpu
from qat.clinalg import CLinalg
from qat.plugins import ScipyMinimizePlugin
from qat.plugins import SPSAMinimizePlugin
from qat.core import Result
from qat.core.qpu import QPUHandler
from qat.core import Result
from qat.core.wrappers.result import aggregate_data
from qat.lang.AQASM import RX, X, H, Y, Z
import time

nb_villes = 3
nbCouches = 5
C = [[0, 1000, 1000], [1, 0, 1000], [1000, 1, 0]]
constraint = max(C) * nb_villes
problem = CombinatorialProblem("TSP", False)
var_array = problem.new_vars(nb_villes**2)
for i in range(nb_villes):
    for j in range(nb_villes):
        for t in range(nb_villes - 1):
            problem.add_clause(
                var_array[i + nb_villes * t]
                & var_array[j + nb_villes * ((t + 1) % nb_villes)],
                C[i][j],
            )
for t in range(nb_villes):
    for i in range(nb_villes):
        problem.add_clause(var_array[i + nb_villes * t], (-2) * 10)
        for i2 in range(nb_villes):
            if i2 != i:
                problem.add_clause(
                    var_array[i + nb_villes * t] & var_array[i2 + nb_villes * t],
                    10,
                )
        
        for t2 in range(nb_villes):
            if t2 != t:
                problem.add_clause(
                    var_array[i + nb_villes * t] & var_array[i + nb_villes * t2],
                    10,
                )
observable = problem.get_observable()
prog = Program()
reg = prog.qalloc(nb_villes**2)
var =  [ prog.new_var(float,'theta'+str(i)) for i in range(nb_villes**2)]
for i in range(nb_villes**2):
    prog.apply(X, reg[i])
for i in range(nb_villes**2):
    prog.apply(RX(var[i]),reg[i])
circ = prog.to_circ()
job = circ.to_job(observable=observable)
optimize =ScipyMinimizePlugin(method="BFGS", tol=1e-5, options={"maxiter": 150}, x0 =np.array([0.01]*nb_villes**2)) #L'optimiseur 
qpu = PyLinalg()
stack =  optimize | qpu #on crée un stack : on va lancer le optimize sur le job qui sera soumis
result = stack.submit(job) #résultat de l'optimisation
circuit = job.circuit.bind_variables(result.parameter_map) #On récupère le circuit précédent en 
#spécifiant les variables (qui sont celles de l'optimiseur)
job = circuit.to_job(nbshots = 10000) #Le second job, qui va nous permettre de connaître l'état du vecteur avec 10000 mesures
qpu = PyLinalg()
result = qpu.submit(job) #Résultat des mesures
probasRes = np.zeros(2**(nb_villes**2))
for sample in result:
    print("State %s amplitude %s" % (sample.state, sample.probability))
    bitstring = sample.state.bitstring
    i = int(bitstring, base=2)
    probasRes[i] = sample.probability
    #s = format(i, 'b').zfill(nb_villes**2)
