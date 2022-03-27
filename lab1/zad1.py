import numpy as np
from matplotlib import pyplot as plt
from funkcje import widmo_amplitudowe, widmo_fazowe, moc_sygnalu, twierdzenie_Parsevala, rysuj

N = 4
s1 = [2, 0, 1, 3]
s2 = [1, 0, 3, 0]

def zad1_a_sygnal_1():
    rysuj("Widmo amplitudowe sygnalu 1", widmo_amplitudowe(s1))
    rysuj("Widmo fazowe sygnalu 1", widmo_fazowe(s1))
    print(f'Moc sygnalu 1: {moc_sygnalu(s1)}')
    print(f'Zachodzi twierdzenie Parsevala dla sygnalu 1: {twierdzenie_Parsevala(s1)}')

def zad1_a_sygnal_2():
    rysuj("Widmo amplitudowe sygnalu 2", widmo_amplitudowe(s2))
    rysuj("Widmo fazowe sygnalu 2", widmo_fazowe(s2))
    print(f'Moc sygnalu 2: {moc_sygnalu(s2)}')
    print(f'Zachodzi twierdzenie Parsevala dla sygnalu 2: {twierdzenie_Parsevala(s2)}')


def splot_kolowy_recznie(sygnal1, sygnal2):
    splot = [0.0 for i in range(N)]
    for s1 in range(N):
        for s2 in range(N):
            splot[s1] += sygnal1[s2] * sygnal2[s1-s2]
    return splot

def splot_kolowy_fft(sygnal1, sygnal2):
    x = np.fft.fft(sygnal1) * np.fft.fft(sygnal2)
    return np.abs(np.fft.ifft(x))

def zad1_b(s1, s2):
    print(splot_kolowy_fft(s1, s2))
    print(splot_kolowy_recznie(s1, s2))

if __name__ == "__main__":
    zad1_a_sygnal_1()
    zad1_a_sygnal_2()
    zad1_b(s1, s2)
