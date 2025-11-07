import cv2
import numpy as np

image = cv2.imread("photo.jpg")

if image is None:
    print(f"Ошибка: Не удалось загрузить изображение.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

_, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

adaptive_thresh_gaussian_image = cv2.adaptiveThreshold(thresh, 255,
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

contours, hierarchy = cv2.findContours(adaptive_thresh_gaussian_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Подсчет объектов
num_objects = len(contours)
print(f"Количество объектов на изображении: {num_objects}")

# Создание копии изображения для вывода
output_image = image.copy()

# Площади и центры объектов
areas = []
centers = []

# Перебор контуров
for i, contour in enumerate(contours):
    # Расчет площади контура
    area = cv2.contourArea(contour)
    areas.append(area)

    # Расчет центра контура
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

        # Вывод площади контура рядом с контуром
        cv2.putText(
            output_image,
            f"Area: {int(area)}",
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    else:
        print(f"У контура {i} нулевая площадь")

# Поиск самого большого и самого маленького объектов
if areas:
    largest_object_index = np.argmax(areas)
    smallest_object_index = np.argmin(areas)

    largest_center = centers[largest_object_index] if len(centers) > largest_object_index else None
    smallest_center = centers[smallest_object_index] if len(centers) > smallest_object_index else None

    # Вывод информации о самом большом объекте
    if largest_center:
        cv2.circle(output_image, largest_center, 5, (0, 255, 255), -1)
        cv2.putText(
            output_image,
            f"Largest Center: {largest_center}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        print(f"Координаты центра самого большого объекта: {largest_center}")
    else:
        print("Не найден центр самого большого объекта (площадь нулевая)")
    # Вывод информации о самом маленьком объекте
    if smallest_center:
        cv2.circle(output_image, smallest_center, 5, (255, 210, 0), -1)
        cv2.putText(
            output_image,
            f"Smallest Center: {smallest_center}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 210, 0),
            1,
        )
        print(f"Координаты центра самого маленького объекта: {smallest_center}")
    else:
        print("Не найден центр самого маленького объекта (площадь нулевая)")
else:
    print("Контуры не найдены")
# --- Новая часть кода заканчивается здесь ---

cv2.drawContours(output_image, contours, -1, (0, 252, 124), 1)  # рисуем контуры на копии изображения
cv2.imshow("obr", adaptive_thresh_gaussian_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
