import cv2
import time
import numpy as np



# Загрузка каскадных файлов XML
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Инициализация видеозахвата
cap = cv2.VideoCapture("Every winner from the HLTV Awards 2024.mp4")

# Переменные для расчета FPS
prev_time = 0
fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем изображение в оттенки серого для обнаружения
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Вычисление FPS
    current_time = time.time()
    time_difference = current_time - prev_time
    fps = 1 / time_difference
    prev_time = current_time

    # Отображение FPS на кадре
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Флаги для проверки улыбки и открытых глаз
    smile_detected = False
    eyes_detected = False

    for (x, y, w, h) in faces:
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Область интереса (ROI) для глаз и улыбки
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Обнаружение глаз
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        if len(eyes) >= 2:  # Если обнаружено хотя бы два глаза
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Обнаружение улыбки
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        if len(smiles) > 0:  # Если обнаружена улыбка
            smile_detected = True
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    # Вывод сообщений на экран
    if not smile_detected:
        cv2.putText(frame, "SMILE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if not eyes_detected:
        cv2.putText(frame, "OPEN EYES", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Отображение кадра
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()