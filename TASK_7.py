# Задание 7 (самостоятельно) Отобразить информацию с вебкамеры,
# записать видео в файл, продемонстрировать видео.

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка при открытии вебкамеры.")
    exit()

output_file = "task_7.avi"  # Имя выходного файла
frame_width = int(cap.get(3))  # Ширина кадра
frame_height = int(cap.get(4))  # Высота кадра
fps = 30.0  # Количество кадров в секунду

center_x = frame_width // 2
center_y = frame_height // 2
ret, frame = cap.read()
center_pixel = frame[center_y, center_x]


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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:

    ret, frame = cap.read()

    if not ret:
        print("Ошибка при захвате кадра.")
        break

    rect_width = 100  # Ширина прямоугольника
    rect_height = 300  # Высота прямоугольника

    top_left_x = (frame_width - rect_width) // 2
    top_left_y = (frame_height - rect_height) // 2
    bottom_right_x = top_left_x + rect_width
    bottom_right_y = top_left_y + rect_height
    if closest == "1":
        cv2.rectangle(cap, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), -1)
    if closest == "2":
        cv2.rectangle(cap, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)
    if closest == "3":
        cv2.rectangle(cap, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), -1)

    rect_width = 300  # Ширина прямоугольника
    rect_height = 100  # Высота прямоугольника

    top_left_x = (frame_width - rect_width) // 2
    top_left_y = (frame_height - rect_height) // 2
    bottom_right_x = top_left_x + rect_width
    bottom_right_y = top_left_y + rect_height
    if closest == "1":
        cv2.rectangle(cap, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), -1)
    if closest == "2":
        cv2.rectangle(cap, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)
    if closest == "3":
        cv2.rectangle(cap, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), -1)

    out.write(frame)

    cv2.imshow("Webcam Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()