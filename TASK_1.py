import cv2 as cv
import numpy as np
# Путь, размерность гауссовского ядра, среднеквадратичное отклонение,
# делители для вычисления значений порогов
def canny(path, ksize, sigma, divs):

    orig_image = cv.imread(path)

    if orig_image is None:
        print('Изображение отсутствует')
        return

    # 1 TASK - гауссовское размытие бинарного изображения
    gray_image = cv.cvtColor(orig_image, cv.COLOR_BGR2GRAY)

    cv.imshow('Gray Image', gray_image)

    gaus_blur_image = cv.GaussianBlur(gray_image, (ksize, ksize), sigma)

    cv.imshow('Gaussian Blur Image', gaus_blur_image)




    # 2 TASK - выведем 2 матрицы: значений длин градиентов и значений углов градиентов
    # Оператор Собеля для вычисления частных производных по x и y

    ker_Sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    ker_Sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # Вычисление частных производных по x и y (градиентов)
    gradient_x = convolution(gaus_blur_image, ker_Sobel_x)
    gradient_y = convolution(gaus_blur_image, ker_Sobel_y)

    # Вычисление значений градиентов
    gradient_value_matrix = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    max_gradient = np.max(gradient_value_matrix)
    gradient_value_matrix_print = orig_image.copy()
    for i in range(orig_image.shape[0]):
        for j in range(orig_image.shape[1]):
            gradient_value_matrix_print[i][j] = (float(gradient_value_matrix[i][j]) / max_gradient) * 255

    cv.imshow('Values of gradients', gradient_value_matrix_print)
    print("Значения матрицы градиентов:\n", gradient_value_matrix_print)


    # Вычисления матрицы значений углов градиентов
    gradient_angle_matrix = np.zeros(orig_image.shape)
    for i in range(orig_image.shape[0]):
        for j in range(orig_image.shape[1]):
            gradient_angle_matrix[i, j] = get_gradient_angle(gradient_x[i, j], gradient_y[i, j])
    cv.imshow('Values of angles', gradient_angle_matrix.astype(np.uint8))
    print('Значение углов градиентов:\n', gradient_angle_matrix)

    # 3 - TASK подавление не максимумов

    non_max_suppression = orig_image.copy()
    for i in range(orig_image.shape[0]):
        for j in range(orig_image.shape[1]):
            angle = gradient_angle_matrix[i][j]
            gradient = gradient_value_matrix[i][j]
            if (i == 0 or i == orig_image.shape[0] - 1 or j == 0 or j == orig_image.shape[1] - 1):
                non_max_suppression[i][j] = 0
            else:

                if np.logical_or(angle == 0, angle == 4).any():
                    y_shift = 0
                elif np.logical_and(angle > 0, angle < 4).any():
                    y_shift = 1
                else:
                    y_shift = -1

                if np.logical_or(angle == 2, angle == 6).any():
                    x_shift = 0
                elif np.logical_and(angle > 2, angle < 6).any():
                    x_shift = -1
                else:
                    x_shift = 1
                if (gradient >= gradient_value_matrix[i + x_shift][j + y_shift] and gradient >= gradient_value_matrix[i - x_shift][j - y_shift]):
                    is_max = True
                else:
                    is_max = False
                non_max_suppression[i][j] = 255 if is_max else 0
    cv.imshow('Non Max Suppression', non_max_suppression)


    # 4 - Task пороговая фильтрация

    low_level = max_gradient // divs[0]
    high_level = max_gradient // divs[1]


    porog_filtration = np.zeros(orig_image.shape)
    for i in range(orig_image.shape[0]):
        for j in range(orig_image.shape[1]):
            gradient = gradient_value_matrix[i][j]
            # потенциальная граница изображения?
            if (non_max_suppression[i][j] == 255).any():
                # градиент находится внутри интервала?
                if (gradient >= low_level and gradient <= high_level):
                    flag = False
                    # проверка пикселя с максимальной длиной градиента среди соседей
                    for k in range(-1, 1):
                        for l in range(-1, 1):
                            if (flag):
                                break
                            # поиск границы внутри интервала верхнего и нижнего порога
                            if (non_max_suppression[i + k][j + l] == 255).any():
                                flag = True
                                break
                    if (flag):
                        porog_filtration[i][j] = 255
                # если значение градиента выше - верхней границы, то пиксель точно граница
                elif (gradient > high_level):
                    porog_filtration[i][j] = 255
    cv.imshow('Porog Filtration', porog_filtration)


    cv.waitKey(0)
    cv.destroyAllWindows()


def convolution(img, ker):
    ksize = len(ker)
    # начальные координаты для итераций по пикселям
    # операция свёртки - каждый пиксель умножается на соответствующий элемент ядра свертки,
    # а затем все произведения суммируются
    x0 = ksize // 2
    y0 = ksize // 2
    # переопределение матрицы изображения для работы с каждым внутренним пикселем
    matr = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr[i][j] = img[i][j]
    for i in range(x0, len(matr)-x0):
        for j in range(y0, len(matr[i])-y0):

            val = 0
            for k in range(-(ksize//2), ksize//2+1):
                for l in range(-(ksize//2), ksize//2+1):
                    val += img[i + k][j + l] * ker[k +(ksize//2)][l + (ksize//2)]
            matr[i][j] = val
    return matr

def get_gradient_angle(x, y):
    if x != 0:
        tg = y/x
    else:
        return -1
    if x < 0:
        if y < 0:
            if tg > 2.414:
                return 0
            elif tg < 0.414:
                return 6
            elif tg <= 2.414:
                return 7
        else:
            if tg < -2.414:
                return 4
            elif tg < -0.414:
                return 5
            elif tg >= -0.414:
                return 6
    else:
        if y < 0:
            if tg < -2.414:
                return 0
            elif tg < -0.414:
                return 1
            elif tg >= -0.414:
                return 2
        else:
            if tg < 0.414:
                return 2
            elif tg < 2.414:
                return 3
            elif tg >= 2.414:
                return 4






# canny('images/img.png', 3, 3, (25, 10)) # very good
canny('mig31.jpg', 15, 10, (25, 10)) # > средн. кв. откл. < границы
#canny('mig31.jpg', 7, 7, (10, 1)) # PoroSad result
# canny('images/img.png', 3, 10, (20, 7))
