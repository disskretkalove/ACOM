import cv2
import numpy as np
import requests

def blur_rectangle(frame, x, y, width, height, blur_kernel_size=(15, 15)):
    # Создаем копию фрейма, чтобы не изменять оригинальный кадр.
    blurred_frame = frame.copy()

    # Определяем координаты верхнего левого и нижнего правого углов прямоугольника.
    top_left = (x, y)
    bottom_right = (x + width, y + height)

    # Выделяем прямоугольник на копии фрейма.
    roi = blurred_frame[y:y+height, x:x+width]

    # Применяем GaussianBlur к выделенному прямоугольнику.
    blurred_roi = cv2.GaussianBlur(roi, blur_kernel_size, 0)

    # Заменяем выделенную область на фрейме на размытую версию.
    blurred_frame[y:y+height, x:x+width] = blurred_roi

    return blurred_frame

def get_points_for_drow_x_on_center_frame_on_center(w,h,width_rectangle_h,height_rectangle_h,width_rectangle_v,height_rectangle_v):


    x = int (((w - width_rectangle_h)/2))
    y = int(((h - height_rectangle_h)/2))

    top_left_for_h = (x,y)
    bottom_right_for_h = (x+width_rectangle_h, y + height_rectangle_h)

    x1 = int (((h - width_rectangle_v)/2))
    y1 = int (((w - height_rectangle_v)/2))

    top_left_for_v = (y1,x1)
    bottom_right_for_v = (y1+ height_rectangle_v, x1 +  int((width_rectangle_v - height_rectangle_h) / 2))

    x2 = int (((h - width_rectangle_v )/2) + width_rectangle_v/2 - height_rectangle_h/2 + height_rectangle_h)
    y2 = int (((w - height_rectangle_v)/2))

    top_left_for_v1 = (y2,x2)
    bottom_right_for_v1 = (y2+ height_rectangle_v, x2 +  int((width_rectangle_v - height_rectangle_h) / 2))

    h_coords =  (top_left_for_h, bottom_right_for_h)
    v_coords = (top_left_for_v, bottom_right_for_v)
    v1_coords = (top_left_for_v1, bottom_right_for_v1)

    return h_coords, v_coords, v1_coords

def drow_X(frame, width_rectangle_h, height_rectangle_h, width_rectangle_v, height_rectangle_v, color):
    height, width, channels = frame.shape
    h_coords, v_coords, v1_coords = get_points_for_drow_x_on_center_frame_on_center(width,height,width_rectangle_h,height_rectangle_h,width_rectangle_v,height_rectangle_v)
    cv2.rectangle(frame, v_coords[0], v_coords[1], color, 2)
    cv2.rectangle(frame, v1_coords[0], v1_coords[1], color, 2)
    cv2.rectangle(frame, h_coords[0], h_coords[1], color, 2)

def drow_RGB_X(frame, width_rectangle_h, height_rectangle_h, width_rectangle_v, height_rectangle_v, color):
    height, width, channels = frame.shape
    h_coords, v_coords, v1_coords = get_points_for_drow_x_on_center_frame_on_center(width,height,width_rectangle_h,height_rectangle_h,width_rectangle_v,height_rectangle_v)
    cv2.rectangle(frame, v_coords[0], v_coords[1], color, -1)
    cv2.rectangle(frame, v1_coords[0], v1_coords[1], color, -1)
    cv2.rectangle(frame, h_coords[0], h_coords[1], color, -1)




def readWriteTOFile():

    video = cv2.VideoCapture(0)
    ok, frame = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))

    width_rectangle_h = 80
    height_rectangle_h = 20

    width_rectangle_v = 100
    height_rectangle_v = 20

    h_coords, v_coords, v1_coords = get_points_for_drow_x_on_center_frame_on_center(w,h,width_rectangle_h,height_rectangle_h,width_rectangle_v,height_rectangle_v)



    while True:
        ok, frame = video.read()
        center_x = w // 2
        center_y = h // 2

        # Определяем цвет, ближайший к центральному пикселю
        central_pixel_color = frame[center_y, center_x]
        if central_pixel_color.max() == central_pixel_color[0]:
            color = (255,0,0)
        elif central_pixel_color.max() == central_pixel_color[1]:
            color = (0,255,0)
        else:
            color = (0,0,255)
        print (central_pixel_color)
        color1 = (0,0,255)
        drow_RGB_X(frame, width_rectangle_h, height_rectangle_h, width_rectangle_v, height_rectangle_v, color)
#        drow_X (frame, width_rectangle_h, height_rectangle_h, width_rectangle_v, height_rectangle_v, color1)

#        blurred_frame = blur_rectangle(frame, h_coords[0][0], h_coords[0][1], width_rectangle_h, height_rectangle_h)


#        cv2.imshow('frame', blurred_frame)
#        video_writer.write(blurred_frame)
        cv2.imshow('frame', frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def print_cam():
    # Создаем объект VideoCapture для чтения с камеры.
    cap = cv2.VideoCapture(0)  # 0 - индекс камеры по умолчанию

    # Устанавливаем размеры кадра (ширина и высота) на 640x480.
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        # Захватываем кадр с камеры.
        ret, frame = cap.read()



        # Преобразуем кадр в черно-белый.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Отображаем кадр в окне с именем "frame".

        cv2.imshow('frame', frame)

        if not ret:
            print("Ошибка при захвате кадра.")
            break
        # Если нажата клавиша 'Esc' (код 27 в ASCII), то выходим из цикла.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Когда всё закончено, освобождаем ресурсы и закрываем окно.
    cap.release()
    cv2.destroyAllWindows()

# Вызываем функцию для запуска камеры.
#print_cam()

readWriteTOFile()