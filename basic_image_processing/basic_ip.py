import cv2
import numpy as np

# print("current opencv version is: %s"%(cv2.__version__))
# path = '../data'
# # Create a VideoCapture object
# cap = cv2.VideoCapture(0)
 
# # Check if camera opened successfully
# if not cap.isOpened(): 
#   print("Unable to read camera feed")
 
# # Default resolutions of the frame are obtained.The default resolutions are system dependent.
# # We convert the resolutions from float to integer.
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
 
# # keep in mind video writer has to keep original video size otherwise video will be corrupted, you could resize every frame later on
# # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# out = cv2.VideoWriter(path+'outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
 
# while True:
#   ret, frame = cap.read()
 
#   if ret: 
     
#     frame = cv2.flip(frame, 0)
#     # Write the frame into the file 'output.avi'
#     out.write(frame)
 
#     # Display the resulting frame    
#     cv2.imshow('frame',frame)
 
#     # Press Q on keyboard to stop recording
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
 
#   # Break the loop
#   else:
#     break 
 
# # When everything done, release the video capture and video write objects
# cap.release()
# out.release()
 
# # Closes all the frames
# cv2.destroyAllWindows() 


# cv2.bitwise_and()
# specifies elements of the output array to be changed.

# cv2.countNonZero()


# img = cv2.imread('../data/gradient.png',0)
# ret1, thresh1 = cv2.threshold()


# a= np.uint8([1])
# from matplotlib import pyplot as plt

# img = cv2.imread('data/gradient.png',0)
# a = 'nannan'
# b= 'katherine'
# c= 'bella'
# d= a+b
# print(d)
# if img.all():
#     print('image is fine')
# else:
#     print('image is not fine')
# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('data/sudoku.png')
kernel1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
scharr_kernel = np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
laplacian_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
dst1 = cv.filter2D(img,-1,kernel1)
dst2 = cv.filter2D(img,-1,kernel2)
dst3 = cv.filter2D(img,-1,scharr_kernel)
dst4 = cv.filter2D(img,-1,laplacian_kernel)
dst5 = cv.Laplacian(img,cv.CV_64F)
dst6 = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
dst7 = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.subplot(241),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(242),plt.imshow(dst1),plt.title('Sobel X')
plt.xticks([]), plt.yticks([])
plt.subplot(243),plt.imshow(dst2),plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])
plt.subplot(244),plt.imshow(dst3),plt.title('scharr')
plt.xticks([]), plt.yticks([])
plt.subplot(245),plt.imshow(dst4),plt.title('laplacian')
plt.xticks([]), plt.yticks([])
plt.subplot(246),plt.imshow(dst5),plt.title('laplacian')
plt.xticks([]), plt.yticks([])
plt.subplot(247),plt.imshow(dst6),plt.title('sobel')
plt.xticks([]), plt.yticks([])
plt.subplot(248),plt.imshow(dst7),plt.title('sobel')
plt.xticks([]), plt.yticks([])
plt.show()