import cv2
import mediapipe as mp

# Инициализация MediaPipe для обнаружения лица и рук
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Инициализация видеопотока с веб-камеры
cap = cv2.VideoCapture(1)  # Изменил порядковый номер камеры на 0

# Ваше имя и фамилия
my_name = "Валерия"
my_surname = "Мерзлякова"

# Открываем файл метаданных для чтения
with open("metadata.txt", "r") as file:
    metadata = file.readlines()

# Создаем пустые списки для хранения путей к изображениям и их меток
image_paths = []
labels = []

# Читаем каждую строку в файле метаданных
for line in metadata:
    # Разделяем строку на путь к изображению и метку, используя пробел в качестве разделителя
    image_path, label = line.strip().split(" ", maxsplit=1)
    # Добавляем путь к изображению и метку в соответствующие списки
    image_paths.append(image_path)
    labels.append(label)

# Создание объектов для обнаружения лица и рук
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Перевод изображения в формат RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Обнаружение лица
        face_results = face_detection.process(image)

        # Обнаружение рук
        hand_results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Рисование прямоугольника вокруг лица
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, bbox, (255, 0, 0), 2)

                # Обнаружение поднятых пальцев
                finger_count = 0
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Получение координат кончиков пальцев
                        tips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
                        for tip in tips:
                            if tip.y < hand_landmarks.landmark[0].y:  # Проверка поднятия пальца (выше ладони)
                                finger_count += 1

                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Вывод текста на изображение
                if finger_count == 1:
                    cv2.putText(image, my_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                cv2.LINE_AA)
                elif finger_count == 2:
                    cv2.putText(image, my_surname, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                cv2.LINE_AA)
                else:
                    cv2.putText(image, "Unknown", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                cv2.LINE_AA)

        # Отображение изображения
        cv2.imshow('MediaPipe Face and Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
