import cv2
import numpy as np

def displayIMG(img, windowName):
    cv2.namedWindow( windowName, cv2.WINDOW_NORMAL )
    cv2.resizeWindow(windowName, 500, 300)
    cv2.imshow(windowName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Read img
print('Read image...')
img = cv2.imread('home.jpg')
displayIMG(img,'original')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
displayIMG(gray,'gray')
# Laplacian method
print('Laplacian method...')
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
displayIMG(lap, 'Lap')

# Sobel method
print('Sobel method...')
sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)
displayIMG(sobelX, 'SibelX')
displayIMG(sobelY, 'SibelY')
displayIMG(sobelCombined, 'SibelXY')

# Canny method
print('Canny method...')
canny = cv2.Canny(gray, 30, 150)
displayIMG(canny, 'Canny')