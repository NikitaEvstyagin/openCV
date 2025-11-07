import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# 1. Загрузка изображения
image = cv2.imread('perec.jpg')  # Замените 'your_image.jpg' на путь к вашему изображению

# 2. Применение Гауссова размытия
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 3. Повышение резкости
# 3.1 Свертка с ядром
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened_kernel = cv2.filter2D(image, -1, kernel)

# 3.2 Маска нерезкости
blurred_for_sharpening = cv2.GaussianBlur(image, (5, 5), 0)
sharpened_unsharp = cv2.addWeighted(image, 1.5, blurred_for_sharpening, -0.5, 0)

# 4. Выделение границ с помощью оператора Собеля
# Преобразуем изображение в градации серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Применяем оператор Собеля
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
# Комбинируем градиенты по X и Y
edges = cv2.magnitude(sobel_x, sobel_y)
# Преобразуем результат в 8-битный формат
edges = cv2.convertScaleAbs(edges)
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# 5. Комбинирование результатов
# Комбинируем размытое изображение и границы
combined = cv2.addWeighted(blurred, 0.5, edges_colored, 0.5, 0)

# Добавляем изображение с повышенной резкостью
combined = cv2.addWeighted(combined, 0.5, sharpened_unsharp, 0.5, 0)

combined = cv2.addWeighted(combined, 0.5, sharpened_unsharp, 0.5, 0)


# 6. Отображение результатов
def show_images(original, blurred, edges, sharpened, combined):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Размытие по Гауссу')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Выделение границ')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости (ядро)')
    plt.imshow(cv2.cvtColor(sharpened_kernel, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Повышение резкости (маска нерезкости)')
    plt.imshow(cv2.cvtColor(sharpened_unsharp, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Комбинация изображений')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Вызов функции для отображения результатов
show_images(image, blurred, edges, sharpened_unsharp, combined)