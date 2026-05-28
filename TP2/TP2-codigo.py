import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq


#
# Para facilitar el run del codigo juntamos ambas funciones (analizar_señal y obtener_numero). 
# Sin embargo, para realizar el TP corrimos primero analizar_señal para distinguir y armar a mano
# los segmentos a y b.
#


def analizar_señal(archivo):

    fs, data = wavfile.read(archivo)

    x = data[:,0]

    t = np.arange(len(x)) / fs

    # Graficamos señal entera 
    
    plt.figure(figsize=(12,4))

    plt.plot(t, x)

    plt.xlabel("Tiempo [s]")
    plt.xticks(np.arange(0, t[-1], 0.2))
    plt.ylabel("Amplitud")

    plt.title("Señal telefónica")

    plt.grid(True)

    plt.show()

    return(x, fs)

tabla = {
    (697,1209): "1",
    (697,1336): "2",
    (697,1477): "3",
    (770,1209): "4",
    (770,1336): "5",
    (770,1477): "6",
    (852,1209): "7",
    (852,1336): "8",
    (852,1477): "9",
    (941,1336): "0"
}

bajas = np.array([697,770,852,941])
altas = np.array([1209,1336,1477])


# Segmentos obtenidos visualmente por analizar_señal

segmentos_a = [
    (0.10, 0.16),
    (0.24, 0.31),
    (0.37, 0.44),
    (0.51, 0.58),
    (0.65, 0.72),
    (0.79, 0.85),
    (0.92, 0.98)
]


segmentos_b = [
    (0.03, 0.32),   
    (0.34, 0.63),   
    (0.95, 1.24),   
    (1.30, 1.55),   
    (2.20, 2.50),    
    (2.87, 3.16),    
    (3.55, 3.83),    
    (3.87, 4)        
]

# Obtenemos numero comparando señales con mayor magnitud

def obtener_numero(archivo, segmentos):

    x, fs = analizar_señal(archivo)

    numero = ""

    for i, (a,b) in enumerate(segmentos):

        # ---------------------------------
        # Recortar segmento.
        # ---------------------------------

        inicio = int(a * fs)
        fin = int(b * fs)

        xs = x[inicio:fin]

        # ---------------------------------
        # FFT.
        # ---------------------------------

        N = len(xs)

        X = np.abs(fft(xs))

        freqs = fftfreq(N, 1/fs)

        # ---------------------------------
        # Nos quedamos solo con las frecuencias 
        # positivas.
        # ---------------------------------

        mask = freqs > 0

        freqs_pos = freqs[mask]
        X_pos = X[mask]

        # ---------------------------------
        # Buscamos los dos picos principales
        # ---------------------------------

        indices = np.argsort(X_pos)[-2:]

        frecuencias_detectadas = np.sort(freqs_pos[indices])

        f1 = frecuencias_detectadas[0]
        f2 = frecuencias_detectadas[1]

        print(f"\nTono {i+1}")

        print("Frecuencias detectadas:")
        print(f"{f1:.1f} Hz")
        print(f"{f2:.1f} Hz")

        # ---------------------------------
        # Aproximamos frecuencias a las de
        # la tabla. 
        # ---------------------------------

        fbaja = bajas[np.argmin(np.abs(bajas - f1))]
        falta = altas[np.argmin(np.abs(altas - f2))]

        print("\nFrecuencias aproximadas:")
        print(fbaja, "Hz")
        print(falta, "Hz")

        # ---------------------------------
        # Reconocer dígito
        # ---------------------------------

        digito = tabla[(fbaja,falta)]

        print("\nDígito:", digito)

        numero += digito

        # ---------------------------------
        # Graficamos Frecuencia/Magnitud por 
        # segmento.
        # ---------------------------------

        plt.figure(figsize=(8,4))

        plt.plot(freqs_pos, X_pos)

        plt.xlim(0,2000)

        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Magnitud")

        plt.title(f"FFT del tono {i+1}")

        plt.grid(True)

        plt.show()

    # =========================
    # Resultado final
    # =========================

    print("\n=========================")
    print("Número detectado:")
    print(numero)
    print("=========================")



obtener_numero("tlfn-a.wav", segmentos_a)
obtener_numero("tlfn-b.wav", segmentos_b)
