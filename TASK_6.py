# Задание 6 (самостоятельно) Прочитать изображение с камеры. Вывести
# в центре на экране Красный крест в формате, как на изображении. Указать
# команды, которые позволяют это сделать.

import cv2

img = cv2.imread('C:/Users/Admin/Downloads/cy34.jpe')
height, width, _ = img.shape

rect_width = 100  # Ширина прямоугольника
rect_height = 300  # Высота прямоугольника

top_left_x = (width - rect_width) // 2
top_left_y = (height - rect_height) // 2
bottom_right_x = top_left_x + rect_width
bottom_right_y = top_left_y + rect_height
cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)

rect_width = 300  # Ширина прямоугольника
rect_height = 100  # Высота прямоугольника

top_left_x = (width - rect_width) // 2
top_left_y = (height - rect_height) // 2
bottom_right_x = top_left_x + rect_width
bottom_right_y = top_left_y + rect_height
cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)
ROI = img[top_left_y:top_left_y + rect_height, top_left_x:top_left_x + rect_width]

# cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
# src	Это входное изображение
# ksize	Это размер ядра по Гауссу. [высота ширина]. Высота и ширина должны быть нечетными и
# могут иметь разные значения. Если ksize установлен в [0 0], то ksize вычисляется из значений сигма.
# sigmaX	Это стандартное отклонение ядра вдоль оси X(горизонтальное направление).
# sigmaY	Это стандартное отклонение ядра по оси Y(вертикальное направление).
# Если sigmaY=0, то для sigmaY берется значение sigmaX.
# borderType	Определяет границы изображения,
# в то время как ядро применяется к границам изображения.
# Возможные значения: cv.BORDER_CONSTANT cv.BORDER_REPLICATE cv.BORDER_REFLECT cv.BORDER_WRAP
# cv.BORDER_REFLECT_101 cv.BORDER_TRANSPARENT cv.BORDER_REFLECT101 cv.BORDER_DEFAULT cv.BORDER_ISOLATED

blur = cv2.GaussianBlur(ROI, (101, 1), 30)
img[top_left_y:top_left_y + rect_height, top_left_x:top_left_x + rect_width] = blur
cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Окно с изменяемым размером
cv2.imshow('image', img)
cv2.waitKey(0)

cv2.destroyAllWindows()