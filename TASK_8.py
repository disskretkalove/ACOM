# Задание 8 (самостоятельно) Залить крест одним из 3 цветов – красный,
# зеленый, синий по следующему правилу: НА ОСНОВАНИИ ФОРМАТА RGB
# определить, центральный пиксель ближе к какому из цветов красный,
# зеленый, синий и таким цветом заполнить крест.

import cv2
import numpy as np

img = cv2.imread('C:/Users/Admin/Downloads/cy34.jpe')
resize = cv2.resize(img, (1080, 720))
cv2.imshow("Original", resize)
height, width, _ = img.shape


height, width, _ = img.shape
center_x = width // 2
center_y = height // 2
center_pixel = img[center_y, center_x]



def closest_color(pixel):
    colors = {"0": (0, 0, 255), "1": (0, 255, 0), "2": (255, 0, 0)}
    min_distance = float('inf')
    closest = None

    for color_name, color_value in colors.items():
        distance = sum((pixel - np.array(color_value)) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest = color_name

    return closest



closest = closest_color(center_pixel)
print(closest)

rect_width = 100  # Ширина прямоугольника
rect_height = 300  # Высота прямоугольника

top_left_x = (width - rect_width) // 2
top_left_y = (height - rect_height) // 2
bottom_right_x = top_left_x + rect_width
bottom_right_y = top_left_y + rect_height
if closest == "1":
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), -1)
if closest == "2":
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), -1)
if closest == "3":
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)


rect_width = 300  # Ширина прямоугольника
rect_height = 100  # Высота прямоугольника

top_left_x = (width - rect_width) // 2
top_left_y = (height - rect_height) // 2
bottom_right_x = top_left_x + rect_width
bottom_right_y = top_left_y + rect_height
if closest == "1":
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), -1)
if closest == "2":
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), -1)
if closest == "3":
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Окно с изменяемым размером
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()