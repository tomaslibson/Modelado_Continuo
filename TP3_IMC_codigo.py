import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from scipy.special import erf


# Datos del problema
alfa = 2.0
beta = 1.0


# -----------------------------------------------------------------------------
# Malla uniforme
# -----------------------------------------------------------------------------
def crear_malla(k):
    N = 2**k                  # cantidad de puntos interiores
    h = 1.0 / (N + 1)         # distancia entre nodos
    x = np.linspace(0, 1, N + 2)
    return N, h, x


# -----------------------------------------------------------------------------
# Funciones y soluciones exactas
# -----------------------------------------------------------------------------
def f_constante(x):
    return 3.0 * np.ones_like(x)


def exacta_constante(x):
    # Solución de u''=3, u(0)=2 y u'(1)=1
    return 1.5*x**2 - 2.0*x + 2.0


def f_exponencial(x):
    return np.exp(-x**2)


def exacta_exponencial(x):
    # Solución de u''=exp(-x²), u(0)=2 y u'(1)=1
    c = beta - np.sqrt(np.pi)/2 * erf(1.0)
    return (
        alfa
        + c*x
        + np.sqrt(np.pi)/2 * x * erf(x)
        + 0.5*(np.exp(-x**2) - 1.0)
    )


# -----------------------------------------------------------------------------
# Sistema con matriz densa
# Incógnitas: U = [u_1, ..., u_N, u_(N+1)]
# -----------------------------------------------------------------------------
def sistema_denso(k, f):
    N, h, x = crear_malla(k)
    m = N + 1

    A = (1.0/h**2) * (
        np.diag(-2.0*np.ones(m))
        + np.diag(np.ones(m - 1), 1)
        + np.diag(np.ones(m - 1), -1)
    )

    b = np.zeros(m)
    b[:N] = f(x[1:N + 1])

    # u_0 = alfa es conocido y pasa al vector b
    b[0] = b[0] - alfa/h**2

    # Condición de Neumann en x=1:
    # (u_(N-1) - 4u_N + 3u_(N+1))/(2h) = beta
    A[-1, :] = 0.0
    A[-1, -3:] = np.array([1.0, -4.0, 3.0])/(2.0*h)
    b[-1] = beta

    return x, h, A, b


# -----------------------------------------------------------------------------
# El mismo sistema, pero con matriz esparsa
# -----------------------------------------------------------------------------
def sistema_esparso(k, f):
    N, h, x = crear_malla(k)
    m = N + 1

    A = (1.0/h**2) * diags_array(
        [np.ones(m - 1), -2.0*np.ones(m), np.ones(m - 1)],
        offsets=[-1, 0, 1],
        shape=(m, m),
        format="lil"
    )

    b = np.zeros(m)
    b[:N] = f(x[1:N + 1])
    b[0] = b[0] - alfa/h**2

    A[-1, :] = 0.0
    A[-1, -3:] = np.array([1.0, -4.0, 3.0])/(2.0*h)
    b[-1] = beta

    return x, h, A.tocsc(), b


# =============================================================================
# ÍTEM 1: f(x)=3 y matriz densa
# =============================================================================
k = 3
x, h, A, b = sistema_denso(k, f_constante)
u = np.concatenate(([alfa], np.linalg.solve(A, b)))

print("ÍTEM 1")
print("h =", h)
print("\nMatriz A:\n", A)
print("\nVector b:\n", b)

plt.figure()
plt.plot(x, exacta_constante(x), label="Exacta")
plt.plot(x, u, "o", label="Numérica")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


# =============================================================================
# ÍTEM 2: error para k=3,...,14
# =============================================================================
valores_k = range(3, 15)
pasos = []
errores = []

print("\nÍTEM 2")
for k in valores_k:
    x, h, A, b = sistema_esparso(k, f_constante)
    u = np.concatenate(([alfa], spsolve(A, b)))

    error = np.max(np.abs(u - exacta_constante(x)))
    pasos.append(h)
    errores.append(error)

    print(f"k={k:2d}   h={h:.6e}   error={error:.6e}")

pasos = np.array(pasos)
errores = np.array(errores)

plt.figure()
plt.loglog(pasos, errores, "o-")
plt.xlabel("h")
plt.ylabel("Error infinito")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()
plt.close()

print("\nEl orden teórico es 2.")
print("Con f(x)=3 la solución es cuadrática, por eso el método resulta exacto")
print("salvo errores de redondeo y la pendiente del gráfico no es confiable.")


# =============================================================================
# ÍTEM 3: f(x)=exp(-x²) y matriz esparsa
# =============================================================================
k = 14
x, h, A, b = sistema_esparso(k, f_exponencial)
u = np.concatenate(([alfa], spsolve(A, b)))
error = np.max(np.abs(u - exacta_exponencial(x)))

print("\nÍTEM 3")
print("k =", k)
print("Cantidad de incógnitas =", len(u) - 1)
print("Elementos no nulos de A =", A.nnz)
print("Error infinito =", error)

plt.figure()
plt.plot(x, exacta_exponencial(x), label="Exacta")
plt.plot(x, u, ".", markersize=2, label="Numérica")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


# Prueba simple del mayor k. El resultado depende de la memoria de la computadora.
ultimo_k = None
for k in range(14, 24):
    try:
        x, h, A, b = sistema_esparso(k, f_exponencial)
        spsolve(A, b)
        ultimo_k = k
        print(f"k={k} resuelto correctamente")
    except MemoryError:
        break

print("\nMayor k probado sin error =", ultimo_k)
print("Para buscar el límite real, aumentar de a uno el 24 del range.")
print("Cada vez que k aumenta en 1, la cantidad de incógnitas se duplica.")
