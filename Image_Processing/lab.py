#!/usr/bin/env python3

import math

from PIL import Image as Image

# NO ADDITIONAL IMPORTS ALLOWED!

'''
Gets 2d position of a pixel from a 1d pixel representation; also accounts for out of bounds pixels using min() and max() during the correlation process
'''
def get_pixel(image, x, y):
    w = image['width']
    return image['pixels'][min(image['height']-1,max(x,0))*w + min(image['width']-1,max(y,0))]

'''
Sets certain pixel values; does not allow for out of bounds coordinates to be provided
'''
def set_pixel(image, x, y, c):
    w = image['width']
    image['pixels'][x*w + y] = c

def load_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_image(image, filename, mode='png'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save("{}.{}".format(filename,mode))
    else:
        out.save("{}.{}".format(filename,mode))
    out.close()


def apply_per_pixel(image, func):
    if(type(image)==str):
        image = load_image(image)
    result = {'height': image['height'], 'width': image['width'], 'pixels': image['pixels'][:]}
    for x in range(image['height']):
        for y in range(image['width']):
            #print(x,y,len(image['pixels']))
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)
    return result


def inverted(image):
    if(type(image)==str):
        image = load_image(image)
    return (apply_per_pixel(image, lambda c: 255-c))


def correlate(image, kernel, do_clip):
    if(type(image)==str):
        image = load_image(image)
    temp_result = {'height':image['height'], 'width':image['width'], 'pixels':[0 for i in range(len(image['pixels'])) ] }
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE:

    Kernel will be represented as an n*m matrix (i.e. a list of lists)
    """
    add = 0
    if(len(kernel)%2!=0):
        add = 1
    for j in range(0,image['height'],1):
        for k in range(0,image['width'],1):
            res = 0.0
            for l in range(-(len(kernel)//2),(len(kernel)//2)+add,1):
                for m in range(-(len(kernel[l])//2),(len(kernel[l])//2)+add,1):
                    #print(j+l,k+m)
                    res += (float)((get_pixel(image,j+l,k+m))*(kernel[l+((len(kernel)//2))][m+((len(kernel[l])//2))]))
            set_pixel(temp_result,j,k,res)
            #print("-------------------------------------------------------------------------------------------")
    if(do_clip):
        round_and_clip_image(temp_result)
    return temp_result
'''
Sharpens the image; basically uses the functionality from blurred() and genblur() with some of the modifications for sharpening
'''
def sharpened(image,n):
    if(type(image)==str):
        image = load_image(image)
    blur = blurred(image,n)
    new_img = {'height':blur['height'], 'width':blur['width'], 'pixels':[]}
    for j in range(len(blur['pixels'])):
        new_img['pixels'].append(2*image['pixels'][j] - blur['pixels'][j])
    round_and_clip_image(new_img)
    return new_img

'''
Uses the Sobel operator to perform edge detection; added an additional do_clip parameter to test_correlation in order to allow img1 and img2 to not be clipped prior to the combining (caused some WAs before)
'''
def edges(image):
    kern1 = [ [1,0,-1], [2,0,-2], [1,0,-1]]
    kern2 = [ [1,2,1], [0,0,0], [-1,-2,-1]]
    if(type(image)==str):
        image = load_image(image)
    img1 = test_correlation(image,kern1)
    #print(img1['pixels'])
    img2 = test_correlation(image,kern2)
    #print(img2['pixels'])
    res_img = {'height':img1['height'], 'width':img1['width'], 'pixels':[]}
    for pix in range(len(img1['pixels'])):
        result = round( math.sqrt(img1['pixels'][pix]*img1['pixels'][pix] + img2['pixels'][pix]*img2['pixels'][pix]) ) 
        res_img['pixels'].append(result)
        #print("PIXEL VALUE: {}".format(result))
    round_and_clip_image(res_img)
    #print("RESULT: {}".format(res_img['pixels']))
    save_image(res_img, 'kernel_final_edge')
    return res_img

'''
Bounds all pixels within the 'image' parameter to be strictly between 0 and 255
'''
def round_and_clip_image(image):
    if(type(image)==str):
        image = load_image(image)
    for pix in range(len(image['pixels'])):
        image['pixels'][pix] = min(255,max(0,round(image['pixels'][pix])))



'''
Generates a box blur filter to use in correlate method
'''
def genblur(n):
    res = []
    for j in range(n):
        tmp = []
        for k in range(n):
            tmp.append(1/(n*n))
        res.append(tmp)
    return res

'''
Runs the blur operation on an image given a size of a box blur filter
'''
def blurred(image, n):
    if(type(image)==str):
        image = load_image(image)
    bblur = genblur(n)
    new_img = test_correlation(image,bblur,True)
    return new_img


# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES
'''
This method acts as a runner for the invert() function; I also used it for small-batch testing instead of directly using test.py
'''
def run_inversion(image):
    if(type(image)==str):
        image = load_image(image)
    #print(load_image(file_name))
    save_image(inverted(image),"bluegill")

'''
This method acts as a runner for the correlate() function; I also used it for small-batch testing instead of directly using test.py
'''
def test_correlation(image,kernel,do_clip=False):
    if(type(image)==str):
        image = load_image(image)
    #kernel = [ [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0] ]
    #the above kernel was used for an initial subproblem, now obsolete
    rimg = correlate(image,kernel,do_clip)
    #save_image(rimg,'kernel_final_correlation')
    return rimg




if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    #ret = blurred(load_image('test_images/twocats.png'),3)
    #res = load_image('test_results/twocats_blur_03.png')
    #for p in range(len(res['pixels'])):
     #   print("{},{}".format(ret['pixels'][p], res['pixels'][p]))
    blurred(load_image('test_images/centered_pixel.png'),4)
    #edges('test_images/construct.png')
    #compare_images('test_results/chess_edges.png','kernel_final_edge.png')
    #compare_images('test_results/mushroom_sharp_03.png', 'kernel_final_sharpen.png')
    #test_image = {'height':3, 'width':3, 'pixels':[3,9,3,3,9,3,3,9,3]}
    #res = correlate(test_image,genblur(3))
    #print(res['pixels'])
    pass
