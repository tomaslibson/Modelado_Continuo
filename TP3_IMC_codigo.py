import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from scipy.special import erf


alfa = 2.0
beta = 1.0


# Malla uniforme
def malla(k):
    N = 2**k
    h = 1 / (N + 1)
    x = np.linspace(0, 1, N + 2)

    return N, h, x


# Funciones del ítem 1
def f_seno(x):
    return np.sin(128 * np.pi * x)


def u_seno(x):
    a = 128 * np.pi

    return (
        -np.sin(a*x) / a**2
        + (beta + 1/a)*x
        + alfa
    )


# Funciones del ítem 3
def f_exp(x):
    return np.exp(-x**2)


def u_exp(x):
    c = beta - np.sqrt(np.pi)/2 * erf(1)

    return (
        alfa
        + c*x
        + np.sqrt(np.pi)/2*x*erf(x)
        + 0.5*(np.exp(-x**2) - 1)
    )


# Matriz densa
def sistema_denso(k, f):
    N, h, x = malla(k)
    m = N + 1

    A = (
        np.diag(-2*np.ones(m))
        + np.diag(np.ones(m - 1), 1)
        + np.diag(np.ones(m - 1), -1)
    ) / h**2

    b = np.zeros(m)
    b[:N] = f(x[1:N + 1])
    b[0] -= alfa/h**2

    # Condición de Neumann en x=1
    A[-1, :] = 0
    A[-1, -3:] = [1/(2*h), -4/(2*h), 3/(2*h)]
    b[-1] = beta

    return x, h, A, b


# Matriz esparsa
def sistema_esparso(k, f):
    N, h, x = malla(k)
    m = N + 1

    A = diags_array(
        [
            np.ones(m - 1),
            -2*np.ones(m),
            np.ones(m - 1)
        ],
        offsets=[-1, 0, 1],
        shape=(m, m),
        format="lil"
    ) / h**2

    b = np.zeros(m)
    b[:N] = f(x[1:N + 1])
    b[0] -= alfa/h**2

    A[-1, :] = 0
    A[-1, -3:] = [1/(2*h), -4/(2*h), 3/(2*h)]
    b[-1] = beta

    return x, h, A.tocsc(), b


# ============================================================
# ÍTEM 1
# ============================================================

# Matriz pequeña para mostrar en papel
k = 2

x, h, A, b = sistema_denso(k, f_seno)

print("\nÍTEM 1")
print("k =", k)
print("h =", h)
print("\nA =")
print(A)
print("\nb =")
print(b)


# Soluciones para algunos valores de k
for k in [8, 10, 12]:

    x, h, A, b = sistema_esparso(k, f_seno)

    u = spsolve(A, b)
    u = np.concatenate(([alfa], u))

    plt.figure(figsize=(8, 5))

    plt.plot(x, u_seno(x), label="Solución exacta")
    plt.plot(x, u, ".", markersize=2, label="Solución numérica")

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(rf"Solución para $k={k}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# ÍTEM 2
# ============================================================

pasos = []
errores = []

print("\nÍTEM 2")

for k in range(3, 15):

    x, h, A, b = sistema_esparso(k, f_seno)

    u = spsolve(A, b)
    u = np.concatenate(([alfa], u))

    error = np.max(np.abs(u - u_seno(x)))

    pasos.append(h)
    errores.append(error)

    print(f"k={k:2d}   h={h:.6e}   error={error:.6e}")


pasos = np.array(pasos)
errores = np.array(errores)


# Pendiente usando los últimos cinco puntos
pendiente, ordenada = np.polyfit(
    np.log(pasos[-5:]),
    np.log(errores[-5:]),
    1
)

ajuste = np.exp(ordenada) * pasos[-5:]**pendiente

print("\nOrden estimado =", pendiente)


plt.figure(figsize=(8, 5))

plt.loglog(
    pasos,
    errores,
    "o-",
    label="Error numérico"
)

plt.loglog(
    pasos[-5:],
    ajuste,
    "--",
    label=f"Pendiente = {pendiente:.2f}"
)

plt.xlabel(r"Paso de la malla $h$")
plt.ylabel(r"Error $\|U-u\|_{\infty}$")
plt.title("Convergencia del método")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# ÍTEM 3
# ============================================================

k = 14

x, h, A, b = sistema_esparso(k, f_exp)

u_aux = spsolve(A, b)
u = np.concatenate(([alfa], u_aux))

error = np.max(np.abs(u - u_exp(x)))

print("\nÍTEM 3")
print("k =", k)
print("h =", h)
print("Cantidad de incógnitas =", len(u_aux))
print("Elementos no nulos de A =", A.nnz)
print("Error infinito =", error)


plt.figure(figsize=(8, 5))

# Se muestran menos puntos para no tapar la curva exacta
plt.plot(
    x[::100],
    u[::100],
    "o",
    markersize=3,
    label="Solución numérica"
)

plt.plot(
    x,
    u_exp(x),
    label="Solución exacta"
)

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(rf"Solución para $f(x)=e^{{-x^2}}$, $k={k}$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# PRUEBA DEL MAYOR k
# Ejecutar localmente
# ============================================================

probar_limite = True

if probar_limite:

    ultimo_k = None

    for k in range(14, 24):

        try:
            x, h, A, b = sistema_esparso(k, f_exp)
            u = spsolve(A, b)

            ultimo_k = k

            print(
                f"k={k:2d}   "
                f"incógnitas={len(u):10d}   "
                f"no nulos={A.nnz:10d}"
            )

            del x, A, b, u

        except (MemoryError, RuntimeError, ValueError):
            print("No se pudo resolver para k =", k)
            break

    print("\nMayor k resuelto =", ultimo_k)