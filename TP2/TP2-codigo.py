import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq


bajas = np.array([697, 770, 852, 941])
altas = np.array([1209, 1336, 1477])

tabla = {
    (697, 1209): "1",
    (697, 1336): "2",
    (697, 1477): "3",
    (770, 1209): "4",
    (770, 1336): "5",
    (770, 1477): "6",
    (852, 1209): "7",
    (852, 1336): "8",
    (852, 1477): "9",
    (941, 1209): "*",
    (941, 1336): "0",
    (941, 1477): "#"
}


def analizar_tono(x, fs, a, b):
    inicio = int(a * fs)
    fin = int(b * fs)

    xs = x[inicio:fin]
    xs = xs - np.mean(xs)
    xs = xs * np.hanning(len(xs))

    N = len(xs)

    X = np.abs(fft(xs))
    freqs = fftfreq(N, 1 / fs)

    mask = freqs > 0
    freqs_pos = freqs[mask]
    X_pos = X[mask]

    magnitudes_bajas = []
    for f in bajas:
        idx = np.argmin(np.abs(freqs_pos - f))
        magnitudes_bajas.append(X_pos[idx])

    magnitudes_altas = []
    for f in altas:
        idx = np.argmin(np.abs(freqs_pos - f))
        magnitudes_altas.append(X_pos[idx])

    fbaja = bajas[np.argmax(magnitudes_bajas)]
    falta = altas[np.argmax(magnitudes_altas)]

    digito = tabla[(fbaja, falta)]

    return fbaja, falta, digito, freqs_pos, X_pos


def analizar_archivo(archivo, segmentos):
    fs, data = wavfile.read(archivo)

    if data.ndim == 2:
        x = data[:, 0]
    else:
        x = data

    t = np.arange(len(x)) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t, x)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title(f"Señal completa - {archivo}")
    plt.grid(True)

    for a, b in segmentos:
        plt.axvspan(a, b, alpha=0.25)

    plt.show()

    numero = ""

    print("\n====================================")
    print(f"Archivo: {archivo}")
    print("====================================")
    print("Tono | Intervalo [s] | f baja | f alta | Dígito")
    print("------------------------------------")

    for i, (a, b) in enumerate(segmentos):
        fbaja, falta, digito, freqs_pos, X_pos = analizar_tono(x, fs, a, b)

        numero += digito

        print(f"{i+1:4d} | {a:.2f} - {b:.2f}   | {fbaja:6d} | {falta:6d} | {digito}")

        plt.figure(figsize=(8, 4))
        plt.plot(freqs_pos, X_pos)
        plt.xlim(500, 1700)
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Magnitud")
        plt.title(f"FFT tono {i+1} - {archivo} - dígito {digito}")
        plt.grid(True)
        plt.axvline(fbaja, linestyle="--", label=f"{fbaja} Hz")
        plt.axvline(falta, linestyle="--", label=f"{falta} Hz")
        plt.legend()
        plt.show()

    print("------------------------------------")
    print(f"Número detectado en {archivo}: {numero}")
    print("====================================\n")

    return numero


# Archivo A: tonos cortos
segmentos_a = [
    (0.10, 0.16),
    (0.24, 0.31),
    (0.37, 0.44),
    (0.51, 0.58),
    (0.65, 0.72),
    (0.79, 0.85),
    (0.92, 0.98)
]

# Archivo B: tonos largos con ruido.
# Usamos intervalos centrados más largos.
segmentos_b = [
    (0.03, 0.32),   # tono 1
    (0.34, 0.63),   # tono 2

    (0.98, 1.27),   # tono 3
    (1.30, 1.60),   # tono 4

    (2.20, 2.50),   # tono 5
    (2.85, 3.15),   # tono 6
    (3.70, 3.98)    # tono 7
]

numero_a = analizar_archivo("tlfn-a.wav", segmentos_a)
numero_b = analizar_archivo("tlfn-b.wav", segmentos_b)

print("RESULTADOS FINALES")
print("------------------")
print("Número tlfn-a:", numero_a)
print("Número tlfn-b:", numero_b)
