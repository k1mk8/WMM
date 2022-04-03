import cv2
import numpy as np
import matplotlib.pyplot as plt

standar_img = "./images/barbara_col.png"
inoise_img = "./images/barbara_col_inoise.png"
inoise2_img = "./images/barbara_col_inoise2.png"
noise_img = "./images/barbara_col_noise.png"

change_img = "./output/zad1/"

unchanged_img = cv2.imread(standar_img, cv2.IMREAD_UNCHANGED)

def save_image(name, image):
    cv2.imwrite(change_img+name, image)

def plt_imshow(img, img_title="image"): # funkcja służąca do rysowania obrazu
    plt.figure() 
    plt.title(img_title) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bez tej linijki obraz staje się niebieski
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.xticks([]), plt.    yticks([])
    plt.show()

def calcPSNR(img1, img2): # funkcja służąca do obliczania PSNR
    imax = 255.**2
    mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size
    return 10.0*np.log10(imax/mse)

def print_histogram(image): # funkcja służąca do rysowania histogramu
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.flatten()
    plt.figure()
    plt.plot(histogram)
    plt.xlim([0,256])
    plt.show()

def zad_1_Gauss(which_image): # funkcja zwracająca PSNR i obraz dla podanego obrazu z zastosowaniem filtru Gaussa
    image = cv2.imread(which_image, cv2.IMREAD_UNCHANGED)
    for i in [3, 5, 7]: # pętla sprawdzająca wyniki dla mask 3, 5 oraz 7
        print(f'Maska: {i} x {i}')
        gauss_blur = cv2.GaussianBlur(image, (i,i), 0)
        print(calcPSNR(unchanged_img, gauss_blur)) # liczenie PSNR
        plt_imshow(gauss_blur)
        x = str(f"{which_image[:10]}{i}x{i}.png")
        save_image(x, gauss_blur)


def zad_1_median(which_image): # funkcja zwracająca PSNR i obraz dla podanego obrazu z zastosowaniem filtru medianowego
    image = cv2.imread(which_image, cv2.IMREAD_UNCHANGED)
    for i in [3, 5, 7]: # pętla sprawdzająca wyniki dla mask 3, 5 oraz 7
        print(f'Maska: {i} x {i}')
        median_blur = cv2.medianBlur(image, i)
        print(calcPSNR(unchanged_img, median_blur)) # liczenie PSNR
        plt_imshow(median_blur) # rysowanie obrazu

def zad2(image):
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_YCrCb[:, :, 0] = cv2.equalizeHist(image_YCrCb[:, :, 0])
    image_end = cv2.cvtColor(image_YCrCb, cv2.COLOR_YCrCb2BGR)
    plt_imshow(image)
    plt_imshow(image_end)
    print_histogram(image)
    print_histogram(image_end)
    save_image("normal_hist.png", image_end)


def zad3():
    W = -10
    image = unchanged_img
    gauss_image = cv2.GaussianBlur(image, (3,3), 0)
    laplacian_image = cv2.Laplacian(gauss_image, cv2.CV_64F)
    img = np.asarray(image, np.float64)
    img_out = cv2.addWeighted(img, 1, laplacian_image, W, 0)
    cv2.imwrite(change_img+"laplacian-10.png", img_out)


if __name__ == "__main__":
    zad_1_median(noise_img)