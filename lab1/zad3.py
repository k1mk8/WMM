import numpy as np
from matplotlib import pyplot as plt
from funkcje import widmo_amplitudowe, widmo_fazowe, rysuj

A = 3
N = 11
n = [0, N, 4 * N, 9 * N]
s1 = [A * (x % N) / N  for x in range(N)]

def dopisanie_zer(sygnal, ilosc):
    return sygnal + [0.0 for i in range(ilosc)]

def zad3(sygnal):
    widmo_amp = widmo_amplitudowe(sygnal)
    widmo_faz = widmo_fazowe(sygnal)

    rysuj("Widmo fazowe sygnalu", widmo_faz)
    rysuj("Widmo amplitudowe sygnalu", widmo_amp)


def zad3_dopisanie(sygnal):
    for n0 in n:
        sygnal1 = dopisanie_zer(sygnal, int(n0))
        print(f'Dopisana ilosc zer: {n0}')
        zad3(sygnal1)

if __name__ == "__main__":
    zad3_dopisanie(s1)