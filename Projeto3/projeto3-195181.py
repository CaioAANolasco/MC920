#! Caio Augusto Alves Nolasco - RA:195181
#! Projeto 3 - MC920 - Introdução ao Processamento de Imagem Digital 

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial import distance as distance

def lowPassFilter(radius, spectrum_center):
    lpfilter = np.zeros(spectrum_center.shape) #!Create a matrix of zeros to build the low pass filter
    center = (lpfilter.shape[0] / 2, lpfilter.shape[1] / 2) #!Save the coordinates of the very center of the image
    for index in np.ndindex(lpfilter.shape): #!Iterate through the filter matrix
        if(distance.euclidean(index, center) <= radius):
            lpfilter[index] = 1 #!If distance from indexed point to center is within the radius value, it is set to 1

    return lpfilter

def highPassFilter(radius, spectrum_center):
    hpfilter = np.zeros(spectrum_center.shape)  #!Create a matrix of zeros to build the high pass filter
    center = (hpfilter.shape[0] / 2, hpfilter.shape[1] / 2) #!Save the coordinates of the very center of the image
    for index in np.ndindex(hpfilter.shape): #!Iterate through the filter matrix
        if(distance.euclidean(index, center) >= radius):
            hpfilter[index] = 1  #!If distance from indexed point to center is not within the radius value, it is set to 1

    return hpfilter

def passBandFilter(inner_radius, outer_radius, spectrum_center):
    pbfilter = np.zeros(spectrum_center.shape) #!Create a matrix of zeros to build the high pass filter
    center = (pbfilter.shape[0] / 2, pbfilter.shape[1] / 2)
    for index in np.ndindex(pbfilter.shape): #!Iterate through the filter matrix
        if(distance.euclidean(index, center) >= inner_radius and distance.euclidean(index, center) <= outer_radius):
            pbfilter[index] = 1 #!If distance from indexed point to center is within [inner_radius, outer_radius], it is set to 1

    return pbfilter

def fastFourierTransform(image):
    spectrum = np.fft.fft2(image) #! Used Numpy FFT functionalities to apply Fourier transform
    spectrum_center = np.fft.fftshift(spectrum) #! Centralizes the spectrum gotten from the Fourier transform

    print(spectrum_center)
    plt.imshow(np.log(1+np.abs(spectrum_center)), "gray"), plt.title("Centered Spectrum") #! Convert complex numbers from FFT to valid matplotlib float parameters
    plt.savefig(output_Path + 'centered_spectrum.png') #! Saves centralized spectrum image

    inverse_filter = np.fft.ifft2(spectrum)
    plt.imshow(np.abs(inverse_filter), "gray"), plt.title("Inverse Fourier Transform")
    plt.savefig(output_Path + 'inversed_fourier.png') #! Saves image obtained from inverse FFT aplied to the spectrum
    return spectrum_center


def applyLowPassFilter(spectrum_center, output_Path):
    radius = int(input("Enter low pass filter radius:")) #! Read radius for filter from user input
    lpfilter = lowPassFilter (radius, spectrum_center)
    lowPassCenter = spectrum_center * lpfilter #! Created low pass filter and aplies it to centralized FFT spectrum
    plt.imshow(np.log(1+np.abs(lowPassCenter)), "gray"), plt.title("Low Pass Filter Center")
    plt.savefig(output_Path + 'low_pass_center_' + str(radius) + '.png') #! Saves image of product of lower pass filter and centralized spectrum.

    descentralizedLowPass = np.fft.ifftshift(lowPassCenter) #! Descentralize spectrum to apply inverse FFT

    inverse_LowPass = np.fft.ifft2(descentralizedLowPass)
    plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Low Pass Filter Processed Image")
    plt.savefig(output_Path + 'low_pass_processed_image_' + str(radius) + '.png') #! Apply inverse FFT to filter results and save filtered image

def applyHighPassFilter(spectrum_center, output_Path):
    radius = int(input("Enter high pass filter radius:")) #! Read radius for filter from user input
    hpfilter = highPassFilter(radius, spectrum_center) 
    highPassCenter = spectrum_center * hpfilter #! Created high pass filter and aplies it to centralized FFT spectrum
    plt.imshow(np.log(1+np.abs(highPassCenter)), "gray"), plt.title("High Pass Filter Center")
    plt.savefig(output_Path + 'high_pass_center_' + str(radius) + '.png') #! Saves image of product of lower pass filter and centralized spectrum.

    descentralizedHighPass = np.fft.ifftshift(highPassCenter) #! Descentralize spectrum to apply inverse FFT

    inverse_highPass = np.fft.ifft2(descentralizedHighPass)
    plt.imshow(np.abs(inverse_highPass), "gray"), plt.title("High Pass Processed Image")
    plt.savefig(output_Path + 'high_pass_processed_image_' + str(radius) + '.png') #! Apply inverse FFT to filter results and save filtered image


def applyPassBandFilter(spectrum_center, output_Path):
    inner_radius = int(input("Enter pass band filter inner radius:"))
    outer_radius = int(input("Enter pass band filter outer radius:"))  #! Read inner and outer radius for filter from user input
    pbfilter = passBandFilter(inner_radius, outer_radius, spectrum_center)
    passBandCenter = spectrum_center * pbfilter #! Created pass band filter and aplies it to centralized FFT spectrum
    plt.imshow(np.log(1+np.abs(passBandCenter)), "gray"), plt.title("Pass Band Filter Center")
    plt.savefig(output_Path + 'pass_band_center_' + str(inner_radius) + '_' + str(outer_radius) + '.png') #! Saves image of product of pass band filter and centralized spectrum.

    descentralizedPassBand = np.fft.ifftshift(passBandCenter) #! Descentralize spectrum to apply inverse FFT

    inverse_PassBand = np.fft.ifft2(descentralizedPassBand)
    plt.imshow(np.abs(inverse_PassBand), "gray"), plt.title("Pass Band Processed Image")
    plt.savefig(output_Path + 'pass_band_processed_image_' + str(inner_radius) + '_' + str(outer_radius) + '.png') #! Apply inverse FFT to filter results and save filtered image

def compressImage(image):
    threshold = int(input("Enter threshold for compression:")) #! Read threshold for image compression
    compressed_image = np.copy(image)

    for index in np.ndindex(compressed_image.shape): #!Iterate through the iamge matriz
        if(np.abs(compressed_image[index]) < threshold): 
            compressed_image[index] = 0 #! If absolue value of frequency is lower than threshold, set it to 0

    compressed_image = np.fft.ifftshift(compressed_image)
    compressed_image = np.fft.ifft2(compressed_image) #! Inverse transform operations
    plt.imshow(np.abs(compressed_image), "gray"), plt.title("Compressed Image")
    plt.savefig(output_Path + 'compressed_image_' + str(threshold) + '.png') #! Apply inverse FFT to compressed image and save filtered image

image_path = input("Enter image path:")
output_Path = input("Enter image output path:")
#! Read, from user input, source image directory and desired directory for filters output

image = Image.open(image_path) #! Open image from given path

#! Calls FFT and filtering functions
spectrum_center = fastFourierTransform(image)
applyLowPassFilter(spectrum_center, output_Path)
applyHighPassFilter(spectrum_center, output_Path)
applyPassBandFilter(spectrum_center, output_Path)
compressImage(spectrum_center)
