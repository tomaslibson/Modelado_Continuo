import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq



def analizar(archivo):

    fs, data = wavfile.read(archivo)

    x = data[:,0]

    t = np.arange(len(x)) / fs

    #
    # Grafico señal entera 
    #

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

# 
# Intervalos aproximados donde aparece cada tono
# 
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
    (0.03, 0.32),   # tono 1
    (0.34, 0.63),   # tono 2
    (0.95, 1.24),   # tono 3
    (1.30, 1.55),   # tono 4
    (2.20, 2.50),    # tono 5
    (2.87, 3.16),    # tono 6
    (3.55, 3.83),    # tono 7
    (3.87, 4)        #tono 8
]

    # =========================
    # Analizar cada segmento
    # =========================

def numero(archivo, segmentos):

    x, fs = analizar(archivo)

    numero = ""

    for i, (a,b) in enumerate(segmentos):

        # ---------------------------------
        # Recortar segmento
        # ---------------------------------

        inicio = int(a * fs)
        fin = int(b * fs)

        xs = x[inicio:fin]

        # ---------------------------------
        # FFT
        # ---------------------------------

        N = len(xs)

        X = np.abs(fft(xs))

        freqs = fftfreq(N, 1/fs)

        # ---------------------------------
        # Quedarse solo con frecuencias positivas
        # ---------------------------------

        mask = freqs > 0

        freqs_pos = freqs[mask]
        X_pos = X[mask]

        # ---------------------------------
        # Buscar los dos picos principales
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
        # Aproximar a frecuencias DTMF
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
        # Graficar espectro
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


numero("tlfn-a.wav", segmentos_a)
numero("tlfn-b.wav", segmentos_b)
