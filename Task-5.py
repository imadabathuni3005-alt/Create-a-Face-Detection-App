# ==========================================
#         AI FACE DETECTION APP
# ==========================================

import cv2
import time

print("=" * 45)
print("        AI FACE DETECTION APP")
print("=" * 45)
print("Starting Camera...")
print("Press 'Q' to Exit\n")

# Load Haar Cascade
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check if classifier loaded
if face_detector.empty():
    print("Error: Haar Cascade file not found.")
    exit()

# Open Webcam
camera = cv2.VideoCapture(0)

# Check camera
if not camera.isOpened():
    print("Error: Unable to access webcam.")
    exit()

previous_time = time.time()

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to read frame.")
        break

    # Mirror Effect
    frame = cv2.flip(frame, 1)

    # Resize Frame (Compact Window)
    frame = cv2.resize(frame, (800, 450))

    # Convert to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Draw Rectangle Around Faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Calculate FPS
    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time

    # Header Bar
    cv2.rectangle(frame, (0, 0), (800, 45), (40, 40, 40), -1)

    cv2.putText(
        frame,
        "AI FACE DETECTION",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (280, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (420, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Status
    if len(faces) > 0:
        status = "Status: Face Detected"
        color = (0, 255, 0)
    else:
        status = "Status: Searching..."
        color = (0, 0, 255)

    cv2.putText(
        frame,
        status,
        (560, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    # Show Window
    cv2.imshow("AI Face Detection App", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
camera.release()
cv2.destroyAllWindows()

print("\n============================================")
print("   Face Detection Application Closed")
print("============================================")