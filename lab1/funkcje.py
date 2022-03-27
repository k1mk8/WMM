import numpy as np
from matplotlib import pyplot as plt

def rysuj(tytul, lista):
    plt.title(tytul)
    plt.stem(lista)
    plt.show()

def widmo_amplitudowe(sygnal):
    widmo = np.fft.fft(sygnal)
    return np.abs(widmo)

def widmo_fazowe(sygnal):
    widmo = np.fft.fft(sygnal)
    return np.angle(widmo)

def moc_sygnalu(sygnal):
    return sum([i**2 for i in sygnal]) / len(sygnal)

def twierdzenie_Parsevala(sygnal):
    fft = np.fft.fft(sygnal)
    parseval = sum([np.abs(n)**2 for n in fft]) / len(sygnal)
    moc = moc_sygnalu(sygnal) * 4
    if parseval == moc:
        return "Prawda"
    else:
        return "Fałsz"