import numpy as np
from matplotlib import pyplot as plt
from numpy import sign, sin, pi, abs
from funkcje import widmo_fazowe, widmo_amplitudowe, rysuj

A = 4
N = 52
n = [0, N/4, N/2, (3*N)/4]
s1 = [A * sin(2 * pi * (x / N)) for x in range(N)]

def eliminacja_szumow(widmo_faz, widmo_amp):
    for i in range(N):
        if abs(widmo_faz[i]) < 1e-4:
            widmo_faz[i] = 0
        if abs(widmo_amp[i]) < 1e-4:
            widmo_amp[i] = 0
            widmo_faz[i] = 0
    return widmo_faz, widmo_amp

def przesuniecie(sygnal, n0):
    nowy_sygnal = sygnal[n0:] + sygnal[:n0]
    return nowy_sygnal

def zad2(sygnal1):
    widmo_amp = widmo_amplitudowe(sygnal1)
    widmo_faz = widmo_fazowe(sygnal1)
    widmo_faz, widmo_amp = eliminacja_szumow(widmo_faz, widmo_amp)

    rysuj("Widmo fazowe sygnalu", widmo_faz)
    rysuj("Widmo amplitudowe sygnalu", widmo_amp)

def zad2_przesuniecia(signal1):
    for n0 in n:
        print(f'Aktualne przesuniecie: {n0}')
        signal = przesuniecie(signal1, int(n0))
        zad2(signal)


if __name__ == "__main__":
    zad2(s1)
    zad2_przesuniecia(s1)