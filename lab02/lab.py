#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# CODE FROM LAB 1 (replace with your code)
def get_pixel(image, x, y):
    w = image['width']
    return image['pixels'][min(image['height']-1,max(x,0))*w + min(image['width']-1,max(y,0))]

'''
Sets certain pixel values; does not allow for out of bounds coordinates to be provided
'''
def set_pixel(image, x, y, c):
    w = image['width']
    image['pixels'][x*w + y] = c

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
        image = load_color_image(image)
    return (apply_per_pixel(image, lambda c: 255-c))

def correlate(image, kernel,do_clip=True):
    if(type(image)==str):
        image = load_color_image(image)
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
            #print(j,k)
            for l in range(-(len(kernel)//2),(len(kernel)//2)+add,1):
                for m in range(-(len(kernel[l])//2),(len(kernel[l])//2)+add,1):
                    res += (float)((get_pixel(image,j+l,k+m))*(kernel[l+((len(kernel)//2))][m+((len(kernel[l])//2))]))
            set_pixel(temp_result,j,k,res)
            #print("-------------------------------------------------------------------------------------------")
    if(do_clip):
        round_and_clip_image(temp_result)
    return temp_result

def round_and_clip_image(image):
    if(type(image)==str):
        image = load_color_image(image)
    for pix in range(len(image['pixels'])):
        image['pixels'][pix] = min(255,max(0,round(image['pixels'][pix])))


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
    bblur = genblur(n)
    new_img = correlate(image,bblur)
    return new_img

def sharpened(image,n):
    if(type(image)==str):
        image = load_greyscale_image(image)
    blur = blurred(image,n)
    new_img = {'height':blur['height'], 'width':blur['width'], 'pixels':[]}
    for j in range(len(blur['pixels'])):
        new_img['pixels'].append(2*image['pixels'][j] - blur['pixels'][j])
    round_and_clip_image(new_img)
    return new_img
def edges(image):
    if(type(image)==str):
        image = load_greyscale_image(image)
    kern1 = [ [1,0,-1], [2,0,-2], [1,0,-1]]
    kern2 = [ [1,2,1], [0,0,0], [-1,-2,-1]]
    img1 = correlate(image,kern1,False)
    #print(img1['pixels'])
    img2 = correlate(image,kern2,False)
    #print(img2['pixels'])
    res_img = {'height':img1['height'], 'width':img1['width'], 'pixels':[]}
    for pix in range(len(img1['pixels'])):
        result = round( math.sqrt(img1['pixels'][pix]*img1['pixels'][pix] + img2['pixels'][pix]*img2['pixels'][pix]) ) 
        res_img['pixels'].append(result)
        #print("PIXEL VALUE: {}".format(result))
    round_and_clip_image(res_img)
    #print("RESULT: {}".format(res_img['pixels']))
    return res_img



# LAB 2 FILTERS


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color_converter(image):
        if(len(image['pixels'])==0):
            return image
        img1 = {"height":image['height'], "width":image['width'], 'pixels':[]}
        img2 = {"height":image['height'], 'width':image['width'], 'pixels':[]}
        img3 = {"height":image['height'], 'width':image['width'], 'pixels':[]}
        for (i,j,k) in image['pixels']:
            img1['pixels'].append(i)
            img2['pixels'].append(j)
            img3['pixels'].append(k)
        i_1 = filt(img1)
        i_2 = filt(img2)
        i_3 = filt(img3)
        ret_img = {'height':image['height'], 'width':image['width'], 'pixels':[]}
        for i in range(len(i_1['pixels'])):
            ret_img['pixels'].append( (i_1['pixels'][i],i_2['pixels'][i],i_3['pixels'][i]) )
        return ret_img
    return color_converter


def make_blur_filter(n):
    def apply_blur(image):
        if(len(image['pixels'])==0):
            return image
        img1 = {"height":image['height'], "width":image['width'], 'pixels':[]}
        for i in image['pixels']:
            img1['pixels'].append(i)
        i_1 = blurred(img1,n)
        ret_img = {'height':image['height'], 'width':image['width'], 'pixels':[]}
        for i in range(len(i_1['pixels'])):
            ret_img['pixels'].append(i_1['pixels'][i])
        return ret_img
    return apply_blur


def make_sharpen_filter(n):
    def apply_sharpen(image):
        if(len(image['pixels'])==0):
            return image
        img1 = {"height":image['height'], "width":image['width'], 'pixels':[]}
        for i in image['pixels']:
            img1['pixels'].append(i)
        i_1 = sharpened(img1,n)
        ret_img = {'height':image['height'], 'width':image['width'], 'pixels':[]}
        for i in range(len(i_1['pixels'])):
            ret_img['pixels'].append(i_1['pixels'][i])
        return ret_img
    return apply_sharpen





def filter_cascade(filters):
    def apply_cascade(image):
        if(len(image['pixels'])==0):
            return image
        for flt in filters:
            print(image['height'], image['width'])
            image = flt(image)
            print("success")
        return image
    return apply_cascade



# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(img, n_iter):
    if(len(img['pixels']) ==  0):
        return img
    for i in range(n_iter):
        gimg = greyscale_image_from_color_image(img)
        edge_detect = compute_energy(gimg)
        #print("EDGE IMAGE: ", edge_detect)
        energy_map = cumulative_energy_map(edge_detect)
        #print(energy_map['pixels'])
        save_greyscale_image(energy_map, 'nrg.png')
        #print("ENERGY MAP: ", energy_map)
        seam = minimum_energy_seam(energy_map)
        print(seam)
        seamless = image_without_seam(img,seam)
        img = seamless
    
    save_color_image(img, 'twocats_seamremoved.png')
    return img


# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    grey_img = {'height':image['height'], 'width':image['width'], 'pixels':[] }
    for (r,g,b) in image['pixels']:
        grey_img['pixels'].append(round(.299*r+.587*g+ .114*b))
    return grey_img



def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    energy_map = edges(grey)
    return energy_map


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    print('----------------------------------------------------------------------------------------------------------------')
    dx = [-1,-1,-1]
    dy = [1,0,-1]
    cumulative_energy_map = {'height':energy['height'], 'width':energy['width'], 'pixels':[0 for i in range(len(energy['pixels']))]}
    for j in range(energy['width']):
        set_pixel(cumulative_energy_map,0,j,energy['pixels'][j])
    for i in range(1,energy['height']):
        for j in range(energy['width']):
            p = [ (i+dx[0],j+dy[0]), (i+dx[1],j+dy[1]), (i+dx[2],j+dy[2]) ]
            min_res = 1000000009
            for (x,y) in p:
                if(x >= 0 and x < energy['height'] and y >= 0 and y < energy['width']):
                    min_res = min(min_res, get_pixel(cumulative_energy_map,x,y))
            set_pixel(cumulative_energy_map, i, j, min_res+get_pixel(energy, i, j))
    return cumulative_energy_map

'''
Locates the minimum index of some value in pix
'''
def locate_minimum_energy(pix):
    m = 1000000009
    v = -1
    for idx in range(len(pix)):
        if(pix[idx] < m):
            m=pix[idx]
            v=idx
    return v

def minimum_energy_seam(cem):
    seam = []
    pix = []
    for j in range(cem['width']):
        pix.append(get_pixel(cem, cem['height'] -1, j))
    #print('PIX:', pix)
    ptr = (cem['height'] - 1, locate_minimum_energy(pix))
    seam.append(ptr)
    dx = [-1,-1,-1]
    dy = [1,0,-1]
    for j in range(cem['height'] - 2, -1, -1):
        new_pt = ()
        cmin = float('inf')
        for x in range(len(dx)):
            (nx,ny) = ptr
            if(nx+dx[x] >= 0 and nx+dx[x] < cem['height'] and ny+dy[x] >= 0 and ny + dy[x] < cem['width']):
                pix_val = get_pixel(cem,nx+dx[x],ny+dy[x])
                if(min(cmin,pix_val) == pix_val):
                    new_pt = (nx+dx[x],ny+dy[x])
                    cmin = pix_val
        seam.append(new_pt)
        ptr = new_pt
    
    idx_seam = []
    for i in range(len(seam)-1,-1,-1):
        (a,b) = seam[i]
        idx_seam.append(a*cem['width'] + b)
    return idx_seam
    """
    Given a cumulative energg map, returns a list of the indices into the
    Given a cumulative energg map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup.
    """


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    cnt = set()
    conv_seam = []
    for i in range(len(seam)):
        x = math.floor(seam[i]/image['width'])
        y = seam[i] - image['width']*x
        conv_seam.append( (x,y) )
    print(conv_seam)
    new_image = {'height':image['height'], 'width':image['width']-1, 'pixels':[]}
    #print(image)
    twodim_rep = []
    for x in range(image['height']):
        tmp = []
        for y in range(image['width']):
            tmp.append(get_pixel(image,x,y))
        twodim_rep.append(tmp)
    #print(twodim_rep)
    forbidden_set = []
    for (x,y) in conv_seam:
        forbidden_set.append( (x,y) )
    for x in range(len(twodim_rep)):
        for y in range(len(twodim_rep[x])):
            if( (x,y) in forbidden_set ):
                continue
            new_image['pixels'].append(twodim_rep[x][y])
    #print(new_image)
    #print(len(new_image['pixels']))
    return new_image
    


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
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


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    '''
    fnc = color_filter_from_greyscale_filter(inverted)
    conv_image = fnc(load_color_image('test_images/cat.png'))
    save_color_image(conv_image, 'colored_cat.png')
    '''
    '''
    mk_blr = make_blur_filter(9)
    apply_blur = color_filter_from_greyscale_image(mk_blr)(load_color_image('test_images/python.png'))
    save_color_image(apply_blur, 'blurred_python.png')
    '''
    '''
    mk_shrp = make_sharpen_filter(7)
    apply_sharp = color_filter_from_greyscale_filter(mk_shrp)(load_color_image('test_images/sparrowchick.png'))
    save_color_image(apply_sharp,'sharpened_chick_2.png')
    '''



    '''
    filter1 = color_filter_from_greyscale_filter(edges)
    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))    
    filt = filter_cascade([filter1, filter1, filter2, filter1])
    cascade_image = filt(load_color_image('test_images/frog.png'))
    save_color_image(cascade_image, 'cascade_result.png')
    '''
    seam_carving(load_color_image('test_images/twocats.png'),100)
    '''
    ig = load_color_image('twocats_seamremoved.png')
    cmp = load_color_image('twocats_1seam.png')
    for i in range(len(ig['pixels'])):
        print(ig['pixels'][i], cmp['pixels'][i])
    '''

    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    pass
