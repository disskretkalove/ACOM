import cv2

def Show_image (window, color):
    image = cv2.imread('C:/Users/Admin/Downloads/cy34.jpe', color)
    cv2.namedWindow('image', window)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
Show_image (cv2.WINDOW_NORMAL, 0)
Show_image (cv2.WINDOW_FULLSCREEN,1)
Show_image (cv2.WINDOW_FREERATIO,3)
