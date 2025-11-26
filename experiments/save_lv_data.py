from scipy import integrate
import numpy as np
import os

# Lotka-Volterra model
def derivative(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx =  alpha*x - beta * x * y
    doty =  -delta*y + gamma * x*y
    return np.array([dotx, doty])

# specify the LV parameters
alpha = 2/3 #mortality rate due to predators
beta = 4/3
delta = 1.
gamma = 1.

# specify the initial conditions and solve the ODE
x0 = 1
y0 = 1
Nt = 21
tmax = 16.
tspan = np.linspace(0.,tmax, Nt)
X0 = [x0, y0]

os.makedirs("data/lotka_volterra", exist_ok = True)
np.save("data/lotka_volterra/tspan.npy", tspan)

# create 100 new datasets contaminated with noise
for i in range(101):
    # add noise to simulate measurement error
    res = integrate.odeint(derivative, X0, tspan, args = (alpha, beta, delta, gamma))
    x, y = res.T

    # combine the data into a form for SINDy
    X = np.stack((x,y), axis = -1)
    

    X_new = X
    X_new[1:,:] = X_new[1:,:] + np.random.normal(0,0.05, size = X_new[1:,:].shape)
    X_new[X_new<0] = 0 # zero out any observations that are negative (not physically possible)
    

    if i < 100:
        np.save(f"data/lotka_volterra/lv_{i}.npy", X_new)
    else:
        np.save("data/lotka_volterra/lv_CV.npy", X_new)