import cv2

# Load the Haar Cascade XML file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture from the default camera
video_cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_cap.read()

    if not ret:
        break

    # Convert the frame to grayscale (Haar cascade works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('video_live', frame)

    # Break the loop if 'a' is pressed
    if cv2.waitKey(10) == ord('a'):
        break

# Release the capture
video_cap.release()
# Destroy all OpenCV windows
cv2.destroyAllWindows()

