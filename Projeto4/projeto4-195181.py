#! Caio Augusto Alves Nolasco - RA:195181
#! Projeto 4 - MC920 - Introdução ao Processamento de Imagem Digital

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2

#! Read image path and output path.

image_path = input("Enter image path:")
output_path = input("Enter image output path:")


def dilate_then_erode (kernel, img_threshold):
    #! Apply dilation morphology operation over image with given kernel.
    dil_erode = cv2.dilate(img_threshold, kernel, iterations=1)

    dil_erode_output_path = output_path + 'dilation_' + str(kernel.shape[0]) + '_' + str(kernel.shape[1]) + '.pbm'
    cv2.imwrite(dil_erode_output_path , dil_erode)

    #! Apply erosion morphology operation over image with given kernel.
    dil_erode = cv2.erode(dil_erode, kernel, iterations=1)

    #! Save image.
    dil_erode_output_path = output_path + 'erode_' + str(kernel.shape[0]) + '_' + str(kernel.shape[1]) + '.pbm'
    cv2.imwrite(dil_erode_output_path , dil_erode)

    return dil_erode

def bitwise_and_image (img1, img2):

    #! Apply bitwise and to two images. In this case, both images subjected to morphological transformations.
    and_img = cv2.bitwise_and(img1, img2) 

    #! Save image.
    and_img_output_path = output_path + 'bitwise_and.pbm'
    cv2.imwrite(and_img_output_path, and_img)

    return and_img

def close_image (kernel, and_image):

    #! Apply closure morph tranformation over image.
    closed_image = cv2.morphologyEx(and_image, cv2.MORPH_CLOSE, kernel)

    #! Save image.
    closed_image_output_path = output_path + 'closed_image.pbm'
    cv2.imwrite(closed_image_output_path , closed_image)

    return closed_image

def count_transitions(box_matrix):

    area = (box_matrix.shape[0] * box_matrix.shape[1])
    black_pixels = area - cv2.countNonZero(box_matrix)

    if black_pixels == 0:
        return 0

    #! Count how many vertical and horizontal transition from a white pixel to a black pixel.
    color_transitions = 0
    for x in range(box_matrix.shape[0]):
        for y in range(box_matrix.shape[1]):
            if box_matrix[x,y] == 255:
                if x+1 in range(box_matrix.shape[0]):
                    if box_matrix[x+1, y] == 0:
                        color_transitions += 1
                if y+1 in range(box_matrix.shape[1]):
                    if box_matrix[x, y+1] == 0:
                        color_transitions += 1

    #! Calculate ratio between transitions found and the area of the rectangle that covers the connected component.
    ratio = color_transitions/black_pixels
    return ratio

def black_pixel_ratio(box_matrix):
    area = box_matrix.shape[0] * box_matrix.shape[1]

    #! Count the number of black pixels inside the rectangle and calculate the ratio between number found and total area,
    black_pixels = area - cv2.countNonZero(box_matrix)
    ratio = black_pixels/area

    return ratio

def text_test(box_matrix):
    transition_ratio = count_transitions(box_matrix)
    pixel_ratio = black_pixel_ratio(box_matrix)

    #! Text if current connected component is a text block. If ratios calculated are within the boundary established, it is block of text.
    if 0.03 <= transition_ratio <= 0.3 and 0.1 <= pixel_ratio <= 0.7:
        return True

    #! If is not within boundaries, it is not text.
    return False

def draw_rectangles_over_text (result_image, original_image):

    #! Find all connected components of closed image.
    contours = cv2.findContours(result_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    text_blocks = 0
    for c in contours:

        #! For every connected component, find the coordinates of the envolving rectangle.
        box = cv2.boundingRect(c)
        x,y,w,h = box
        box_matrix = np.array(result_image[y:y+h, x:x+w])

        #! Submit found rectangle to test to determine if it is text.
        is_text = text_test(box_matrix)
        if is_text:

            #! If it is text, draw rectangle over block of text in original image.
            text_blocks += 1
            cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 0, 0), 2)


    #! Save image with identified text block drawn.
    final_image_output_path = output_path + 'final_image.pbm'
    cv2.imwrite(final_image_output_path, original_image)

    return text_blocks


def main():
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_threshold = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel1 = np.ones((1,100), np.uint8)
    first_morph_image = dilate_then_erode(kernel1, img_threshold)

    kernel2 = np.ones((200,1), np.uint8)
    second_morph_image = dilate_then_erode(kernel2, img_threshold)

    and_img = bitwise_and_image (first_morph_image, second_morph_image)

    kernel3 = np.ones((1,30))
    closed_image = close_image (kernel3, and_img)

    text_boxes = draw_rectangles_over_text (closed_image, original_image)
    print(text_boxes)


if __name__ == '__main__':
    main()





    


