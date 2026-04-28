import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


alphas = [ 0.5 , 1, 3] 

#Sistema de Ecuaciones diferenciales (delta = 0)

def f_delta_0(u, v):
    return v, -u + 1/alpha


def derivadas_delta_0(uv, theta):
    u, v = uv
    du_dtheta, dv_dtheta = f_delta_0(u, v)
    return du_dtheta, dv_dtheta


#Sistema de Ecuaciones diferenciales (delta = 0.05)

def f_delta_05(u, v):
    return v, -u + 1/alpha + 0.05 * u**2

def derivadas_delta_05(uv, theta):
    u, v = uv
    du_dtheta, dv_dtheta = f_delta_05(u, v)
    return du_dtheta, dv_dtheta


# Intervalo de valores de Theta a analizar. 
ti = 0
tf = 8*np.pi  
cant_puntos = 2000
lt = np.linspace(ti, tf, cant_puntos)

# Distintos valores de epsilon
epsilons = [0.2, 0.5, 0.8, 1, 1.2]

# Separamos en e< 1 y e >= 1, para que los resultados de delta = 0.05 sean mas legibles
epsilons1 = [0.2, 0.5, 0.8]
epsilons2 = [1, 1.2]


####################
# Grafico delta = 0
####################

fig, ax = plt.subplots()
ax.set_aspect(1.0)

for alpha in alphas:
    for eps in epsilons:
        
        # Condiciones iniciales
        u0 = (1 + eps)/alpha
        v0 = 0
        estado_inicial = [u0, v0]
        
        # Resolver sistema
        sol = odeint(derivadas_delta_0, estado_inicial, lt)
        
        u, v = sol.T
        r = 1/u
        
        # Pasaje a coordenadas cartesianas
        x = r * np.cos(lt)
        y = r * np.sin(lt)
        
        plt.plot(x, y, label=f"ε = {eps}")

    # Sol en el origen
    plt.scatter(0, 0, color='orange', label='Sol')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.grid()
    plt.legend()
    plt.title("Órbitas de Kepler (δ = 0 ; α=" + str(alpha) + ")")
    plt.show()

#############################
#Grafico delta = 0.05 ; e < 1
#############################

fig, ax = plt.subplots()
ax.set_aspect(1.0)

for alpha in alphas:
    for eps in epsilons1:
        u0 = (1 + eps)/alpha
        v0 = 0
        estado_inicial = [u0, v0]

        sol = odeint(derivadas_delta_05, estado_inicial, lt)
        u, v = sol.T
        r = 1/u

        x = r * np.cos(lt)
        y = r * np.sin(lt)

        plt.plot(x, y, label=f"ε = {eps}")

    plt.scatter(0, 0, color='orange', label='Sol')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.legend()
    plt.title("Órbitas (δ = 0.05, ε < 1, α=" + str(alpha) + ")")
    plt.show()

##############################
#Grafico delta = 0.05 ; e >= 1
##############################

fig, ax = plt.subplots()
ax.set_aspect(1.0)

for alpha in alphas:
    for eps in epsilons2:
        u0 = (1 + eps)/alpha
        v0 = 0
        estado_inicial = [u0, v0]

        sol = odeint(derivadas_delta_05, estado_inicial, lt)
        u, v = sol.T
        r = 1/u

        x = r * np.cos(lt)
        y = r * np.sin(lt)

        plt.plot(x, y, label=f"ε = {eps}")

    plt.scatter(0, 0, color='orange', label='Sol')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.legend()
    plt.title("Órbitas (δ = 0.05, ε ≥ 1, α=" + str(alpha) + ")")
    plt.show()