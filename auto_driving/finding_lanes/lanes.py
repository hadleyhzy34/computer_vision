import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

path = os.getcwd()+'/auto_driving/finding_lanes/source/'

def make_coordinates(image, line_parameters):
    print(line_parameters,type(line_parameters))
    if len(line_parameters) == 0:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    #coordinates of lines on the left
    left_fit = []
    #coordinates of lines on the right
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        #slope and y intercept
        print("shape of parameter is: ",parameters.shape)
        print("coefficient parameters are: ",parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    
    print("left_fit is: ")
    print(left_fit)
    print("right fit is: ")
    print(right_fit)

    #average columns for slope and y_intercept for lines on the left side
    if len(left_fit) !=0:
        left_fit_average = np.average(left_fit, axis=0)
    else:
        left_fit_average = []

    #average columns for slope and y_intercept for lines on the right side
    if len(right_fit) != 0:
        right_fit_average = np.average(right_fit, axis=0)
    else:
        right_fit_average = []
    
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)

    return np.array([left_line, right_line])




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
        [(200,height),(1100,height),(500,250)]
    ])
    # print(polygons.shape)
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
            #check lines shape, if lines coming from houghline, then its three dimension array, otherwise its two dimension array
            if line is not None:
                x1,y1,x2,y2 = line.reshape(4)
                cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),10)
    
    return line_image

# img = cv2.imread(path+'test_image.jpg')
# lane_image = np.copy(img)
# canny_img = canny_test(lane_image)
# roi_img = roi(canny_img)
# cv2.imwrite(path+'roi_img.jpg',roi_img)

# #hough line transform
# #tutorial: https://www.youtube.com/watch?v=eLTLtUVuuy4 45:00min
# #two pixels as one bin, precision would be 1 degree which is the same as np.pi/180
# lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# averaged_lines = average_slope_intercept(lane_image, lines)

# #display lines
# line_image = display_lines(lane_image, averaged_lines)
# cv2.imwrite(path+'line_image.jpg',line_image)

# print("shape of lines is: ",lines.shape)
# print("shape of line_image is: ",averaged_lines.shape)

# #blend to original image
# blend_img = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# plt.imshow(blend_img)
# plt.show()

#video road lanes processing
cap = cv2.VideoCapture(path+'test2.mp4')

#prepare the video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(path+'output.avi', fourcc, 20.0, (width,height))
cv2.VideoWriter()


while(cap.isOpened()):
    #decode every frame of video
    ret,frame = cap.read()
    if not ret:
        print(frame)
        print("can't receive frame")
        break
    canny_img = canny_test(frame)
    roi_img = roi(canny_img)

    lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)

    #blend to original image
    blend_img = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    out.write(blend_img)

cap.release()
out.release()