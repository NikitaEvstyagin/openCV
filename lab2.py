import cv2
import numpy as np

# Загрузка видеофайла
video_path = "catball.mp4"
cap = cv2.VideoCapture(video_path)

# Захват видеопотока с камеры
# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определение диапазона цвета (например, красный)
    lower_red = np.array([10, 100, 100])  # Нижняя граница красного цвета
    upper_red = np.array([30, 255, 255])  # Верхняя граница красного цвета

    # Создание маски по цвету
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Морфологические операции для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)  # Эрозия
    mask = cv2.dilate(mask, kernel, iterations=2)  # Дилатация

    # Поиск контуров на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры найдены
    if contours:
        # Фильтрация контуров по площади (игнорируем мелкие шумы)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

        if contours:
            # Выбор наибольшего контура
            largest_contour = max(contours, key=cv2.contourArea)

            # Вычисление центра масс
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)

                # Отрисовка контура и центра масс
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # Вывод координат центра масс
                cv2.putText(frame, f"Center: {center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Отображение результата
    cv2.imshow("Camera Feed", frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()