import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

alpha = 1

# Campo vectorial
def f(u, v):
    return v, -u + 1/alpha

# Sistema para odeint
def derivadas_para_odeint(uv, theta):
    u, v = uv
    du_dtheta, dv_dtheta = f(u, v)
    return du_dtheta, dv_dtheta

fig, ax = plt.subplots()
ax.set_aspect(1.0)

# Intervalo angular (varias vueltas)
ti = 0.0
tf = 8*np.pi   # 4 vueltas (es u (theta) = u (theta + 2pi))
nt = 100
lt = np.linspace(ti, tf, nt)

# Distintos valores de excentricidad
epsilons = [0.2, 0.5, 0.8, 1, 1.2]

for eps in epsilons:
    
    # Condiciones iniciales
    u0 = (1 + eps)/alpha
    v0 = 0
    estado_inicial = [u0, v0]
    
    # Resolver sistema
    sol = odeint(derivadas_para_odeint, estado_inicial, lt)
    
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
plt.title("Órbitas de Kepler (δ = 0)")
plt.show()