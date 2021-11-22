#! Caio Augusto Alves Nolasco - RA:195181
#! Projeto 2 - MC920 - Introdução ao Processamento de Imagem Digital 

import numpy as np
import cv2
#! Used libraries for this project are numpy and OpenCV2  


def filterH1(cv2_image):
    km = np.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]])
    #! Define mask for h1 filter as a numpy array

    
    new_image = cv2.filter2D(cv2_image,-1,km)
    #! Use OpenCV2 filter2D() function to apply the mask defined by the numpy array km

    output_path_extension = image_output_path + "/filterh1.png"
    cv2.imwrite(output_path_extension, new_image)
    #! Save result image with proper name extension in given directory 


def filterH2(cv2_image):
    km = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
            ])
    
    km = km / 256
    #! Define mask for h2 filter as a numpy array

    new_image = cv2.filter2D(cv2_image,-1,km)
    #! Use OpenCV2 filter2D() function to apply the mask defined by the numpy array km

    output_path_extension = image_output_path + "/filterh2.png"
    cv2.imwrite(output_path_extension, new_image)
    #! Save result image with proper name extension in given directory 

def filterH3(cv2_image):
    km = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]) 
    #! Define mask for h3 filter as a numpy array   
    
    new_image = cv2.filter2D(cv2_image,-1,km)
    #! Use OpenCV2 filter2D() function to apply the mask defined by the numpy array km

    output_path_extension = image_output_path + "/filterh3.png"
    cv2.imwrite(output_path_extension, new_image)
    #! Save result image with proper name extension in given directory 

    return new_image
    #! Filtered image is returned for future used (combine_h3_h4 filterss)

def filterH4(cv2_image):
    km = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]]) 
    #! Define mask for h4 filter as a numpy array   

    new_image = cv2.filter2D(cv2_image,-1,km)
    #! Use OpenCV2 filter2D() function to apply the mask defined by the numpy array km

    output_path_extension = image_output_path + "/filterh4.png"
    cv2.imwrite(output_path_extension, new_image)
    #! Save result image with proper name extension in given directory 

    return new_image
    #! Filtered image is returned for future used (combine_h3_h4 filterss)

def filterH5(cv2_image):
    km = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]])
    #! Define mask for h5 filter as a numpy array     

    new_image = cv2.filter2D(cv2_image,-1,km)
    #! Use OpenCV2 filter2D() function to apply the mask defined by the numpy array km

    output_path_extension = image_output_path + "/filterh5.png"
    cv2.imwrite(output_path_extension, new_image)
    #! Save result image with proper name extension in given directory 

def filterH6(cv2_image):
    km = np.ones((3,3))  
    km = km / 9
    #! Define mask for h6 filter as a numpy array

    new_image = cv2.filter2D(cv2_image,-1,km)
    #! Use OpenCV2 filter2D() function to apply the mask defined by the numpy array km

    output_path_extension = image_output_path + "/filterh6.png"
    cv2.imwrite(output_path_extension, new_image)
    #! Save result image with proper name extension in given directory

def filterH7(cv2_image):
    km = np.array([
            [-1, -1, 2],
            [-1, 2, -1],
            [2, -1, -1]])
     #! Define mask for h7 filter as a numpy array  

    new_image = cv2.filter2D(cv2_image,-1,km)
    #! Use OpenCV2 filter2D() function to apply the mask defined by the numpy array km

    output_path_extension = image_output_path + "/filterh7.png"
    cv2.imwrite(output_path_extension, new_image)
    #! Save result image with proper name extension in given directory

def filterH8(cv2_image):
    km = np.array([
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2]])  
    #! Define mask for h8 filter as a numpy array 

    new_image = cv2.filter2D(cv2_image,-1,km)
    #! Use OpenCV2 filter2D() function to apply the mask defined by the numpy array km

    output_path_extension = image_output_path + "/filterh8.png"
    cv2.imwrite(output_path_extension, new_image)
    #! Save result image with proper name extension in given directory

def combine_H3_H4 (filterH3_image, filterH4_image):
    #! Receive results from filters h3 and h4 as pamaters in numpy array format

    sqrdH3 = np.array(filterH3_image, dtype='uint16')
    sqrdH4 = np.array(filterH4_image, dtype='uint16')
    #! Arrays are copied from a int8 array to int16 array to avoid overflow

    sqrdH3 = sqrdH3 ** 2
    sqrdH4 = sqrdH4 ** 2

    sumh3_h4 = sqrdH3 + sqrdH4
    #! Geometric mean of results from h3 and h4

    combined_image = sumh3_h4 ** (1/2)
    #! Apply desired algorithmic transformation to created a new image from parameters: new_image = [(h3^2 + h4^2) ^ (1/2)]
    combined_image = combined_image.astype(int)
    #! Convert numbers from float to integers

    output_path_extension = image_output_path + "/combinedh3h4.png"
    cv2.imwrite(output_path_extension, combined_image)
    #! Save result image with proper name extension in given directory

def main():
    cv2_image = cv2.imread(image_path, 0)
    #! Open image as numpy array using OpenCV2 method imread()

    filterH1(cv2_image)
    filterH2(cv2_image)
    filterH3_image = filterH3(cv2_image)
    filterH4_image = filterH4(cv2_image)
    filterH5(cv2_image)
    filterH6(cv2_image)
    filterH7(cv2_image)
    filterH8(cv2_image)
    #! Apply all defined filters

    combine_H3_H4(filterH3_image, filterH4_image)
    #! Apply special filter from filters h3 and h4

image_path = input("Enter image path:")
image_output_path = input("Enter image output path:")
#! Read, from user input, source image directory and desired directory for filters output
if __name__ == "__main__":
    main()