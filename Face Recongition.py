import cv2
from fer import FER   # Facial Emotion Recognition

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize FER detector
detector = FER(mtcnn=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions in frame
    results = detector.detect_emotions(frame)

    # Draw results
    for result in results:
        (x, y, w, h) = result["box"]
        emotions = result["emotions"]

        # Get emotion with highest score
        top_emotion = max(emotions, key=emotions.get)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display emotion text
        cv2.putText(frame, top_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Face & Emotion Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
