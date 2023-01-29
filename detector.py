import cv2

# Initialize camera
cap = cv2.VideoCapture(0)

# Load the cascade for hand detection
hand_cascade = cv2.CascadeClassifier('hand.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands in the image
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw a rectangle around the hands
    for (x,y,w,h) in hands:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('Hand Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
