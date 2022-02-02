import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def load_img(file_name):
    Img = cv.imread(file_name)
    return Img


def convert_gray(file_name):
    img = cv.imread(file_name)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray_img


def salt_pepper_noise(file_name):
    img = cv.imread(file_name)
    row = img.shape[0]
    col = img.shape[1]
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        # Color that pixel to black
        img[y_coord][x_coord] = 0
          
    return img


def gaussian_noise(file_name):
    img = cv.imread(file_name)
    # Generate Gaussian noise
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    img = cv.add(img,gauss)

    return img


def poisson_noise(file_name):
    img = cv.imread(file_name)
    values = len(np.unique(img))
    values = 2 ** np.ceil(np.log2(values))
    img = np.random.poisson(img * values) / float(values)
    return img


''' Point Transformation '''

def brightness_adjustment(file_name, beta=30):
    img = cv.imread(file_name)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    
    lim = 255 - beta
    v[v > lim] = 255
    v[v <= lim] += beta

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)

    return img


def contrast_adjustment(file_name, alpha=2):
    img = cv.imread(file_name)
    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y,x,c] = np.clip(alpha*img[y,x,c], 0, 255)

    return new_image


def histogram(file_name):
    img = cv.imread(file_name)
    # show histogram of an img
    if img.ndim == 3:
        blank = np.zeros(img.shape[:2], dtype='uint8')
        mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
        plt.figure()
        plt.title('Colour Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of pixels')
        colors = ('b', 'g', 'r')
        for i,col in enumerate(colors):
            hist = cv.calcHist([img], [i], mask, [256], [0,256])
            plt.plot(hist, color=col)
            plt.xlim([0,256])
        
        plt.show()
        cv.waitKey(0)
    elif img.ndim == 2:
        gray_hist = cv.calcHist([img], [0], None, [256], [0,256] )
        plt.figure(facecolor='black')
        plt.title('Grayscale Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of pixels')
        plt.plot(gray_hist)
        plt.xlim([0,256])
        plt.show()
        cv.waitKey(0)

    return img


def histogram_equalization(file_name):
    img = cv.imread(file_name)
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
 
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
 
    # convert the YUV image back to RGB format
    img = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img


''' Local Transformation '''

def low_pass_filter(file_name):
    img = cv.imread(file_name)
    # low pass filter to blur the image
    return cv.blur(img, (5, 5))


def high_pass_filter(file_name):
    img = cv.imread(file_name)
    # high pass filter to highlight edges
    kernel = np.array([[0.0, -1.0, 0.0],
                        [-1.0, 4.0, -1.0],
                        [0.0, -1.0, 0.0]])
    return cv.filter2D(img, -1, kernel)


def median(file_name):
    img = cv.imread(file_name)
    return  cv.medianBlur(img, 3)


def average(file_name):
    img = cv.imread(file_name)
    return cv.blur(img,(5,5))


def laplacian(file_name):
    img = cv.imread(file_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.convertScaleAbs(cv.Laplacian(img, cv.CV_64F))


def gaussian(file_name):
    img = cv.imread(file_name)
    return cv.GaussianBlur(img, (3, 3), 0)


def vertical_sobel(file_name):
    img = cv.imread(file_name)
    # x
    return cv.convertScaleAbs(cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3))


def horizontal_sobel(file_name):
    img = cv.imread(file_name)
    # y
    return cv.convertScaleAbs(cv.Sobel(img, cv.CV_16S, 0, 1, ksize=3))


#def sobel(file_name):
    img = cv.imread(file_name)
    return cv.addWeighted(vertical_sobel(img), 0.5, horizontal_sobel(img), 0.5, 0)


def vertical_prewitt(file_name):
    img = cv.imread(file_name)
    kernelx = np.array([[1,1,1],
                        [0,0,0],
                        [-1,-1,-1]])
    return cv.filter2D(img, -1, kernelx)


def horizontal_prewitt(file_name):
    img = cv.imread(file_name)
    kernely = np.array([[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]])
    return cv.filter2D(img, -1, kernely)


#def prewitt(file_name):
    img = cv.imread(file_name)
    return vertical_prewitt(img) + horizontal_prewitt(img)


def laplacian_of_gaussian(file_name):
    # Apply Gaussian Blur
    img = gaussian(file_name)
 
    # Apply Laplacian operator in some higher datatype
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.convertScaleAbs(cv.Laplacian(img, cv.CV_64F))
    return img


def canny(file_name):
    img = cv.imread(file_name)
    return cv.Canny(convert_gray(file_name),150,175)


def zero_cross(file_name):
    img = cv.imread(file_name)
    return laplacian_of_gaussian(file_name)


def skeleton(file_name):
    img = cv.imread(file_name)
    img = convert_gray(file_name)
    threshold, img = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    skel = np.zeros(img.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    while True:
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        temp = cv.subtract(img, opening)
        eroded = cv.erode(img, element)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy()
        if cv.countNonZero(img)==0:
            break
    return skel


def Hough_Line_Transform(file_name):
    img = cv.imread(file_name)
    # Find the edges in the image using canny detector
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 300, 600, None, 3)

    # Detect points that form a line
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=20)

    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0,0,255) , 3)

    return img


def Hough_Circle_Transform(file_name):
    img = cv.imread(file_name)
    img_gray = convert_gray(file_name)

    # Blur the image to reduce noise
    img_blur = cv.medianBlur(img_gray, 3)

    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)

    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            # Draw outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    return img

    
def dilation(file_name):
    img = cv.imread(file_name)
    kernel = np.ones((5, 5), 'uint8')
    img = cv.dilate(img, kernel, iterations=1)
    
    return img


def erosion(file_name):
    img = cv.imread(file_name)
    kernel = np.ones((5, 5), 'uint8')
    img = cv.erode(img, kernel, iterations=1)
    
    return img


def opening(file_name):
    img = cv.imread(file_name)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    opening_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    return opening_img


def closing(file_name):
    img = cv.imread(file_name)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    closing_img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    return closing_img
