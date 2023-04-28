import numpy as np
import cv2

def displayIMG(img, windowName):
    cv2.namedWindow( windowName, cv2.WINDOW_NORMAL )
    cv2.resizeWindow(windowName, 600, 600)
    cv2.imshow(windowName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 讀取圖檔
image = cv2.imread('images.jpg')
displayIMG(image, 'Original')        

# 轉為灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
displayIMG(gray, 'Gray')

# 模糊化圖片
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
displayIMG(blurred, 'Blur')

# Canny 尋找邊緣
edged = cv2.Canny(blurred, 30, 150)
displayIMG(edged, 'Edged')


#findContour 確定輪廓
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img2 = image.copy()
cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)
displayIMG(img2, 'contour')

# 取出圖形
for (i, c) in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(c)
    thing = image[y:y + h, x:x + w]
    cv2.imshow('thing', thing)
    mask = np.zeros(image.shape[:2], dtype = 'uint8')
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    displayIMG(cv2.bitwise_and(thing, thing, mask = mask),'final')