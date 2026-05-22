import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

alphas = [0.5, 1, 3]
delta_rel = 0.05

def derivadas(uv, theta, alpha, delta):
    u, v = uv
    du = v
    dv = -u + 1/alpha + delta*u**2
    return [du, dv]

ti = 0.0
tf = 8*np.pi
nt = 4000
theta = np.linspace(ti, tf, nt)

epsilons_cerradas = [0.2, 0.5, 0.8]
epsilons_abiertas = [1, 1.2]

for alpha in alphas:
    for eps in epsilons_cerradas:
        u0 = (1 + eps) / alpha
        v0 = 0
        estado_inicial = [u0, v0]

        sol_kep = odeint(derivadas, estado_inicial, theta, args=(alpha, 0))
        u_kep = sol_kep[:, 0]
        r_kep = 1/u_kep
        x_kep = r_kep * np.cos(theta)
        y_kep = r_kep * np.sin(theta)

        sol_rel = odeint(derivadas, estado_inicial, theta, args=(alpha, delta_rel))
        u_rel = sol_rel[:, 0]
        r_rel = 1/u_rel
        x_rel = r_rel * np.cos(theta)
        y_rel = r_rel * np.sin(theta)

        plt.figure(figsize=(7, 7))
        plt.plot(x_kep, y_kep, label="Kepler δ = 0")
        plt.plot(x_rel, y_rel, label="Relativista δ = 0.05")

        plt.scatter(0, 0, color="orange", label="Sol")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        plt.title(f"Órbita cerrada: α = {alpha}, ε = {eps}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
