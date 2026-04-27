import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

alphas = [ 0.5 , 1, 3] 

# Campo vectorial
def f(u, v):
    return v, -u + 1/alpha + 0.05 * u**2

# Sistema para odeint
def derivadas_para_odeint(uv, theta):
    u, v = uv
    du_dtheta, dv_dtheta = f(u, v)
    return du_dtheta, dv_dtheta


# Intervalo angular (varias vueltas)
ti = 0.0
tf = 8*np.pi   # 4 vueltas
nt = 2000
lt = np.linspace(ti, tf, nt)

# 0<= eps < 1
epsilons1 = [0.2, 0.5, 0.8]

fig, ax = plt.subplots()
ax.set_aspect(1.0)

for alpha in alphas:
    for eps in epsilons1:
        u0 = (1 + eps)/alpha
        v0 = 0
        estado_inicial = [u0, v0]

        sol = odeint(derivadas_para_odeint, estado_inicial, lt)
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
    plt.title("Órbitas (ε < 1, δ = 0.05, α=" + str(alpha) + ")")
    plt.show()


# eps >= 1
epsilons2 = [1, 1.2]

fig, ax = plt.subplots()
ax.set_aspect(1.0)

for alpha in alphas:
    for eps in epsilons2:
        u0 = (1 + eps)/alpha
        v0 = 0
        estado_inicial = [u0, v0]

        sol = odeint(derivadas_para_odeint, estado_inicial, lt)
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
    plt.title("Órbitas (ε ≥ 1, δ = 0.05 α=" + str(alpha) + ")")
    plt.show()