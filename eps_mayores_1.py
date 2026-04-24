from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

alpha = 1

def system(theta, Y):
    u, y = Y
    return [y, -u + 1/alpha]

epsilons = [1.1, 1.5, 2]

for eps in epsilons:
    
    theta_s = np.arccos(-1/eps)
    
    theta1 = np.linspace(0, theta_s - 0.01, 1000)
    theta2 = np.linspace(theta_s + 0.01, 2*np.pi - 0.01, 1000)
    
    plt.figure(figsize=(5,5))
    
    for theta in [theta1, theta2]:
        
        u0 = (1 + eps)/alpha
        y0 = 0
        
        sol = solve_ivp(system,
                        [theta[0], theta[-1]],
                        [u0, y0],
                        t_eval=theta)
        
        u = sol.y[0]
        r = 1/u
        
        # FILTRO: nos quedamos con la zona física relevante
        mask = (r > 0) & (r < 10)
        
        x = r[mask] * np.cos(theta[mask])
        y_cart = r[mask] * np.sin(theta[mask])
        
        plt.plot(x, y_cart)
    
    plt.scatter(0, 0, color='orange', label='Sol')
    
    # 🔍 ZOOM
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    plt.axis('equal')
    plt.title(f"Órbita abierta (zoom) ε = {eps}")
    plt.legend()
    plt.grid()
    
    plt.show()