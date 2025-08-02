import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('bottle_classifier.h5')
class_names = ['Opaque', 'Steel', 'Transparent']
x,y,w,h = 200, 200, 250, 250
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    roi = frame[y:y+h, x:x+w]
    input_img = cv2.resize(roi, (150, 150))
    input_img = input_img.astype('float32') / 255.0
    input_img = np.expand_dims(input_img, axis=0)
    predictions = model.predict(input_img)
    predicted_class = class_names[np.argmax(predictions)]
    cv2.putText(frame, f'Prediction: {predicted_class}', (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow('Bottle Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
