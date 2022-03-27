import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, pi
from funkcje import rysuj

A1 = 0.3
A2 = 0.4
A3 = 0.5
F1 = 5000
F2 = 6000
F3 = 11000
Fp = 48000
N = 2048

def wartosc_w_czasie(t):
    wartosc = A1 * sin(2 * pi * F1 * t)
    wartosc += A2 * sin(2 * pi * F2 * t)
    wartosc += A3 * sin(2 * pi * F3 * t)
    return wartosc

def stworz_sygnal(n):
    signal = [wartosc_w_czasie(x / Fp) for x in range(n)]
    print(f'Wartosc N: {n}')
    return signal

def zad4(sygnal):
    widmowa_gestosc_mocy = 2 * np.abs(np.fft.rfft(sygnal)/N)
    rysuj("Widmo gestosci mocy", widmowa_gestosc_mocy)

if __name__ == "__main__":
    zad4(stworz_sygnal(N))
    zad4(stworz_sygnal(int(N * 3 / 2)))