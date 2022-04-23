import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

file = "./image/barbara_mono.png"
image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

def cv_imshow(img, img_title="image"):
    """
    Funkcja do wyświetlania obrazu w wykorzystaniem okna OpenCV.
    Wykonywane jest przeskalowanie obrazu z rzeczywistymi lub 16-bitowymi całkowitoliczbowymi wartościami pikseli,
    żeby jedną funkcją wywietlać obrazy różnych typów.
    """
    # cv2.namedWindow(img_title, cv2.WINDOW_AUTOSIZE) # cv2.WINDOW_NORMAL

    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    elif img.dtype == np.int16:
        img_ = img*128
    else:
        img_ = img
    plt.figure()
    cv2.imshow(img_title, img_)
    cv2.waitKey(10)  ### oczekiwanie przez bardzo krótki czas - okno się wyświetli, ale program się nie zablokuje, tylko będzie kontynuowany

def printi(img, img_title="image"):
    """ Pomocnicza funkcja do wypisania informacji o obrazie. """
    print(f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}")

def histogram(image):
    return cv2.calcHist([image], [0], None, [256], [0, 256])

def calculate_bitrate(image):
    bitrate = 8*os.stat(file).st_size/(image.shape[0]*image.shape[1])
    print(f"bitrate: {bitrate:.4f}")


def calc_entropy(hist):
    pdf = hist/hist.sum() ### normalizacja histogramu -> rozkład prawdopodobieństwa; UWAGA: niebezpieczeństwo '/0' dla 'zerowego' histogramu!!!
    # entropy = -(pdf*np.log2(pdf)).sum() ### zapis na tablicach, ale problem z '/0'
    entropy = -sum([x*np.log2(x) for x in pdf if x != 0])
    return entropy

def hdiff():
    img_tmp1 = image[:, 1:]  ### wszystkie wiersze (':'), kolumny od 'pierwszej' do ostatniej ('1:')
    img_tmp2 = image[:, :-1] ### wszystkie wiersze, kolumny od 'zerowej' do przedostatniej (':-1')
    image_hdiff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0, dtype=cv2.CV_16S)
    image_hdiff_0 = cv2.addWeighted(image[:, 0], 1, 0, 0, -127, dtype=cv2.CV_16S) ### od 'zerowej' kolumny obrazu oryginalnego odejmowana stała wartość '127'
    image_hdiff = np.hstack((image_hdiff_0, image_hdiff)) ### połączenie tablic w kierunku poziomym, czyli 'kolumna za kolumną'
    cv_imshow(image_hdiff, "image_hdiff")
    printi(image_hdiff, "image_hdiff")

def nic():
    img_tmp1 = image[:, 1:]
    img_tmp2 = image[:, :-1]
    image_hdiff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0, dtype=cv2.CV_16S)
    image_hdiff_0 = cv2.addWeighted(image[:, 0], 1, 0, 0, -127, dtype=cv2.CV_16S) ### od 'zerowej' kolumny obrazu oryginalnego odejmowana stała wartość '127'
    image_hdiff = np.hstack((image_hdiff_0, image_hdiff)) ### połączenie tablic w kierunku poziomym, czyli 'kolumna za kolumną'
    printi(image_hdiff, "image_hdiff")
    cv_imshow(image_hdiff, "image_hdiff")
    image_hdiff = cv2.calcHist([(image_hdiff + 255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
    histo = histogram(image)
    plt.plot(np.arange(0, 256), histo, color="blue", label="obrazu wejściowy")
    plt.plot(np.arange(-255, 256), image_hdiff, color="red", label="obraz różnicowy")
    plt.legend()
    plt.gcf().set_dpi(150)
    plt.show()

if __name__ == "__main__":
    #calculate_bitrate(image)
    #print(calc_entropy(histogram(image)))
    nic()