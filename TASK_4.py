# Задание 4 Записать видео из файла в другой файл.
# Так как в задании не уточнено, я буду записывать видео с вебки в файл

import cv2


output_file = "output_video.avi"  # Имя выходного файла
frame_width = 640  # Ширина кадра
frame_height = 480  # Высота кадра
fps = 30.0  # Количество кадров в секунду


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Ошибка при открытии вебкамеры.")
    exit()


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при захвате кадра.")
        break

    # Запись кадра в выходное видео
    out.write(frame)

    # зеркало
    cv2.imshow("Webcam Video", frame)

    # выход
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# выход
cap.release()
out.release()
cv2.destroyAllWindows()