import cv2

image_path = 'C:/Users/Admin/Downloads/cy34.jpe'
img = cv2.imread(image_path)
resize = cv2.resize(img, (720, 360))

hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)
hsv_res = cv2.resize(hsv, (720, 360))
cv2.imshow("Original", resize)
cv2.imshow("HSV", hsv_res)
cv2.waitKey(0)