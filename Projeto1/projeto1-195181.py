import imageio as imgio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import PIL
import math

### 1.1) Image Resolution
def image_resolution(pil_image):

    #image, in this case, is a PIL image object
    new_output_path = image_output_path + '/resolution256.png'
    pil_image.save(new_output_path, dpi=(256,256))

    new_output_path = image_output_path + '/resolution128.png'
    pil_image.save(new_output_path, dpi=(128,128))

    new_output_path = image_output_path + '/resolution64.png'
    pil_image.save(new_output_path, dpi=(64,64))

    new_output_path = image_output_path + '/resolution32.png'
    pil_image.save(new_output_path, dpi=(32,32))

### 1.2) Quantize an image
def quantize (pil_image):
    #image, in this case, is a PIL image object
        
    # quantize a image for values 256, 128, 64, 32, 16
    quantized_image = pil_image.quantize(256)
    new_output_path = image_output_path + '/quantized_image256.png'
    quantized_image.save(new_output_path)

    quantized_image = pil_image.quantize(128)  
    new_output_path = image_output_path + '/quantized_image128.png' 
    quantized_image.save(new_output_path)

    quantized_image = pil_image.quantize(64)  
    new_output_path = image_output_path + '/quantized_image64.png' 
    quantized_image.save(new_output_path)

    quantized_image = pil_image.quantize(32) 
    new_output_path = image_output_path + '/quantized_image32.png'  
    quantized_image.save(new_output_path)

    quantized_image = pil_image.quantize(16) 
    new_output_path = image_output_path + '/quantized_image16.png'  
    quantized_image.save(new_output_path)

###### 1.3)
### log transformation
def log_transformation(numpy_image):
    #constant c equals the maximum gray value divided by g(x,y) for the maximum value
    c = 255/math.log10(numpy_image.max() + 1)
    log_image = c * np.log10(numpy_image + 1)
    log_image = Image.fromarray(log_image.astype(np.uint8))
    new_output_path = image_output_path + '/log_image.png'
    log_image.save(new_output_path)


### exp transformation
def exp_transformation (numpy_image):
    #normalize values so they fit in numpy array (overflow)
    arraydivided = numpy_image / 255

    #constant c equals the maximum gray value divided by g(x,y) for the maximum value
    c = 1/math.exp(arraydivided.max())
    exp_image = c * np.exp(arraydivided)
    exp_image *= 255
    exp_image = Image.fromarray(exp_image.astype(np.uint8))
    new_output_path = image_output_path + '/exp_image.png'
    exp_image.save(new_output_path)


### quadratic transformation
def quadratic_transformation (numpy_image):
    #constant c equals the maximum gray value divided by g(x,y) for the maximum value
    c = math.pow(255, 2)/math.pow(numpy_image.max(), 2)
    quadratic_image = c * np.square(numpy_image)
    quadratic_image = Image.fromarray(quadratic_image.astype(np.uint8))
    new_output_path = image_output_path + '/quadratic_image.png'
    quadratic_image.save(new_output_path)

### square root transformation
def square_root_transformatio (numpy_image):
    #constant c equals the maximum gray value divided by g(x,y) for the maximum value
    c = 255/math.sqrt(numpy_image.max())
    sqrt_image = c * np.sqrt(numpy_image)
    sqrt_image = Image.fromarray(sqrt_image.astype(np.uint8))
    new_output_path = image_output_path + '/sqrt_image.png'
    sqrt_image.save(new_output_path)


### contrast enhancement
def contrast_enhancement (numpy_image):

    #read inputs parameters
    a = int(input("Enter 'a' value:"))
    b = int(input("Enter 'b' value:"))
    alfa = float(input("Enter 'alfa' value:"))
    beta = float(input("Enter 'beta' value:"))
    gama = float(input("Enter 'gama' value:"))

    zeroes = np.zeros((512, 512))

    #create three altered matrices, each for every function
    cond1 = alfa * numpy_image
    cond2 = beta * (numpy_image - a) + alfa * a
    cond3 = gama * (numpy_image - b) + beta * (b - a) + alfa * a

    #apply the new value when relevant, zero it out when it is not
    cond1 = np.where((numpy_image <= a) & (numpy_image >= 0), cond1, zeroes)
    cond2 = np.where((numpy_image > a) & (numpy_image <= b), cond2, zeroes)
    cond3 = np.where((numpy_image > b) & (numpy_image < 255), cond3, zeroes)

    #add the three intermediate matrices
    contrast_image = cond1 + cond2 + cond3

    contrast_image = Image.fromarray(contrast_image.astype(np.uint8))
    new_output_path = image_output_path + '/constrat_image.png'
    contrast_image.save(new_output_path)

def main():
    pil_image = Image.open('/home/cnolasco/Documents/city.png')
    numpy_image = imgio.imread(image_path)

    image_resolution(pil_image)

    quantize(pil_image)

    log_transformation(numpy_image)
    exp_transformation(numpy_image)
    quadratic_transformation(numpy_image)
    square_root_transformatio(numpy_image)
    contrast_enhancement(numpy_image)


image_path = input("Enter image path:")
image_output_path = input("Enter image output path:")
main()