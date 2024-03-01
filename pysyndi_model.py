import numpy as np
import warnings
from copy import copy
import pysindy as ps
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error
import math

import matplotlib.pyplot as plt
from scipy.io import loadmat

from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning

def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    warnings.filters = filters


#### TERCERODATOS

# Cargar el archivo .mat
u_ref= loadmat('T_ref_11.mat')
data = loadmat('states_11.mat')
time = loadmat('t_11.mat')


ext = 600
init = 50


# Acceder a la variable 'states'
states = data['states']


t = time['t']
t_vector = t[0,init:ext]
dt = 1/30

u = u_ref['T_ref']

# Crear matriz de referencia T_ref con las mismas dimensiones que u
u_zeros = np.zeros((2, u.shape[1]))

# Concatenar dos filas de ceros en las primeras posiciones de la matriz 'u'
u = np.vstack((u_zeros, u))

print(u.shape)

# Extraer las submatrices de 'states' según las filas especificadas
pose = states[0:3, init:ext]
euler = states[3:6, init:ext]
vel = states[6:9, init:ext]
euler_p = states[9:12, init:ext]
omega = states[12:15, init:ext]
quat = states[15:19, init:ext]
v_body = states[19:22, init:ext]



# Apilar verticalmente v y omega
v1 = np.vstack((pose, euler))
v1_p = np.vstack((vel, euler_p))

v1 = v1.T

v1_p = v1_p.T







u_train = u[:,init:ext]

u_train = u_train.T


# Fit the model





x_train_multi = []
u_train_multi = []

# Agrega los vectores a la lista x_train_multi
x_train_multi.append(v1)
x_train_multi.append(v1)
x_train_multi.append(v1)
x_train_multi.append(v1)

# Agrega los vectores a la lista x_train_multi
u_train_multi.append(u_train)
u_train_multi.append(u_train)
u_train_multi.append(u_train)
u_train_multi.append(u_train)


library1 = ps.PolynomialLibrary(degree=1)
library2 = ps.FourierLibrary(n_frequencies=2)
library3 = ps.IdentityLibrary(  )
lib_generalized = ps.GeneralizedLibrary([library1, library2])
lib_generalized.fit(v1)

optimizador = ps.SR3(trimming_fraction=0.1)


variables = ["nx", "ny","nz","phi", "theta","psi","u0","u0","u1","u2","u3","u4" ]

# Define las funciones de la biblioteca
def identity(x):
    return x

def seno(x):
    if 'u' in x.tolist():
        return str(x)
    else:
        return np.sin(x)
    
def coseno(x):
    if 'u' in x.tolist():
        return str(x)
    else:
        return np.cos(x)



# Define los nombres de las funciones
def identity_name(x):
    return str(x)

def seno_name(x):
    if 'u' not in x:
        return "sin("+str(x)+")"
    else:
        return str(x)
    
def coseno_name(x):
    if 'u' not in x:
        return "cos("+str(x)+")"
    else:
        return str(x)

library_functions = [
    identity,
    seno,
    coseno,
]

library_function_names = [
    identity_name,
    seno_name,
    coseno_name
]

custom_library = ps.CustomLibrary(
    library_functions=library_functions, 
    function_names=library_function_names
)

model = ps.SINDy(
    
    optimizer=ps.STLSQ(threshold=0.1, alpha=.05, verbose=True),
    feature_library=custom_library,
    differentiation_method = ps.SINDyDerivative(kind="kalman", alpha=0.05),
    feature_names=["nx", "ny","nz","phi", "theta","psi","u0","u0","u1","u2","u3","u4" ], 
)

x_dot_precomputed = ps.FiniteDifference()._differentiate(v1, t_vector)

model.fit(v1,  t=t_vector, u = u_train, multiple_trajectories=False)
model.print()

feature_names = model.get_feature_names()
for name in feature_names:
    print(name)



x0_test = v1[0,:]



x_test_sim = model.simulate(x0=x0_test, t=t_vector, u=u_train )

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(v1, u=u_train)  

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(v1, t=dt)


# Plot original data and model prediction
plt.figure(figsize=(10, 6))


# Plot original data
plt.subplot(3, 2, 1)
plt.plot(range(len(x_dot_test_computed[:, 0])), x_dot_test_computed[:, 0], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 0])), x_dot_test_predicted [:, 0], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 0]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot second variable
plt.subplot(3, 2, 2)
plt.plot(range(len(x_dot_test_computed[:, 1])), x_dot_test_computed[:, 1], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 1])), x_dot_test_predicted [:, 1], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 1]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot third variable
plt.subplot(3, 2, 3)
plt.plot(range(len(x_dot_test_computed[:, 2])), x_dot_test_computed[:, 2], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 2])), x_dot_test_predicted [:, 2], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 2]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot fourth variable
plt.subplot(3, 2, 4)
plt.plot(range(len(x_dot_test_computed[:, 3])), x_dot_test_computed[:, 3], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 3])), x_dot_test_predicted [:, 3], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 3]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot fifth variable
plt.subplot(3, 2, 5)
plt.plot(range(len(x_dot_test_computed[:, 4])), x_dot_test_computed[:, 4], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 4])), x_dot_test_predicted [:, 4], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 4]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot sixth variable
plt.subplot(3, 2, 6)
plt.plot(range(len(x_dot_test_computed[:, 5])), x_dot_test_computed[:, 5], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 5])), x_dot_test_predicted [:, 5], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 5]) and Model Prediction')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





# Plot original data and model prediction
plt.figure(figsize=(10, 6))


# Plot original data
plt.subplot(3, 2, 1)
plt.plot(range(len(x_test_sim[:, 0])), x_test_sim[:, 0], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 0])), v1[:, 0], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 0]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot second variable
plt.subplot(3, 2, 2)
plt.plot(range(len(x_test_sim[:, 1])), x_test_sim[:, 1], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 1])), v1[:, 1], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 1]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot third variable
plt.subplot(3, 2, 3)
plt.plot(range(len(x_test_sim[:, 2])), x_test_sim[:, 2], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 2])), v1[:, 2], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 2]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot fourth variable
plt.subplot(3, 2, 4)
plt.plot(range(len(x_test_sim[:, 3])), x_test_sim[:, 3], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 3])), v1[:, 3], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 3]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot fourth variable
plt.subplot(3, 2, 5)
plt.plot(range(len(x_test_sim[:, 4])), x_test_sim[:, 4], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 4])), v1[:, 4], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 3]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot fourth variable
plt.subplot(3, 2, 6)
plt.plot(range(len(x_test_sim[:, 5])), x_test_sim[:, 5], label='Prediction Data', color='blue')
plt.plot(range(len(v1[:, 5])), v1[:, 5], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 3]) and Model Prediction')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()