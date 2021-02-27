import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

path = os.getcwd()+'/auto_driving/finding_lanes/source/'

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        

def canny_test(image):
    #convert to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite(path+'gray_image.jpg',gray)

    #guassianblur
    #We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively. If only sigmaX is specified, sigmaY is taken as equal to sigmaX. If both are given as zeros, they are calculated from the kernel size.
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # cv2.imwrite(path+'blur_image.jpg',blur)

    #bilateral filtering
    bf = cv2.bilateralFilter(gray,9,75,75)
    # cv2.imwrite(path+'bf_image.jpg',bf)

    #gradient
    canny_img = cv2.Canny(blur, 50, 150)
    # cv2.imwrite(path+'canny_image.jpg',canny)
    return canny_img

def roi(image):
    height = image.shape[0]
    polygons = np.array([
        [(200,height),(1100,height),(500,240)]
    ])
    print(polygons.shape)
    #create same size image as input image with all pixels to be black, then fill triangle area with white color
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)

    #using bitwise & filter to preserver pixel values only in triangle(white) area
    mask_img = cv2.bitwise_and(mask,image)
    return mask_img

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            print(line)
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),10)
    
    return line_image

img = cv2.imread(path+'test_image.jpg')
lane_image = np.copy(img)
canny_img = canny_test(lane_image)
roi_img = roi(canny_img)

#hough line transform
#tutorial: https://www.youtube.com/watch?v=eLTLtUVuuy4 45:00min
#two pixels as one bin, precision would be 1 degree which is the same as np.pi/180
lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)


#display lines
line_image = display_lines(lane_image, lines)

#blend to original image
blend_img = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
plt.imshow(blend_img)
plt.show()
