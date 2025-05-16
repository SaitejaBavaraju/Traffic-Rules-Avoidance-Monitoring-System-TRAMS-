import cv2
import cvzone
import math
from ultralytics import YOLO
import pytesseract
import time
from twilio.rest import Client

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO models
helmet_model = YOLO('helmet_model.pt')  # Replace with actual model path
number_plate_model = YOLO('license_plate_detector.pt')

# Twilio Configuration (Commented out for security)
# account_sid = 'your_account_sid'
# auth_token = 'your_auth_token'
# client = Client(account_sid, auth_token)

def send_warning_message(number_plate, violation_type, recipient_phone):
    """ Sends an SMS warning message using Twilio """
    message_body = f"ALERT: {violation_type} detected for vehicle {number_plate}. Please follow  with traffic rules."
    try:
        # message = client.messages.create(
        # messaging_service_sid='your_service_sid',
        # body=message_body,
        # to=recipient_phone
        # )
        print(f"Message sent successfully to {recipient_phone}")
    except Exception as e:
        print(f"Failed to send message: {str(e)}")

def extract_number_plate(image):
    """ Extracts text from a number plate using OCR """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 7')
    return text.strip()

# Define class names for helmet detection
classNames = ['With Helmet', 'Without Helmet']

# Video input
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
display_width = frame_width * 2
display_height = frame_height * 2

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (frame_width, frame_height))

# Set up zebra crossing line
zebra_line_y = None
def draw_zebra_line(event, x, y, flags, param):
    global zebra_line_y
    if event == cv2.EVENT_LBUTTONDOWN:
        zebra_line_y = y

cv2.namedWindow("Set Zebra Line", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Set Zebra Line", draw_zebra_line)

ret, frame = cap.read()
while zebra_line_y is None:
    cv2.imshow("Set Zebra Line", frame)
    cv2.waitKey(1)
cv2.destroyWindow("Set Zebra Line")

# Fixed Red-Light Schedule
RED_LIGHT_DURATION = 30  # Red light for 60 seconds
GREEN_LIGHT_DURATION = 30  # Green light for 40 seconds
CYCLE_DURATION = RED_LIGHT_DURATION + GREEN_LIGHT_DURATION  # 100-second cycle

start_time = time.time()  # Store the initial time

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", display_width, display_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Determine current light status
    elapsed_time = int(time.time() - start_time)
    traffic_light_red = elapsed_time % CYCLE_DURATION < RED_LIGHT_DURATION
    print(f"Traffic Light {'RED' if traffic_light_red else 'GREEN'}")

    # Run detection models
    helmet_results = helmet_model(frame)
    plate_results = number_plate_model(frame)

    detected_plates = {}
    for plate in plate_results:
        for pbox in plate.boxes:
            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
            plate_crop = frame[py1:py2, px1:px2]
            number_plate_text = extract_number_plate(plate_crop)
            if number_plate_text:
                detected_plates[(px1, py1, px2, py2)] = number_plate_text
                cvzone.putTextRect(frame, number_plate_text, (px1, py1 - 10), scale=1, thickness=2)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

            # Check for red-light violations
            center_y = (py1 + py2) // 2
            if traffic_light_red and center_y > zebra_line_y:
                send_warning_message(number_plate_text, 'Red Light Violation', '+917093660389')
                with open("violations_log.txt", "a") as log_file:
                    log_file.write(f"{time.ctime()} - Red Light Violation - {number_plate_text}\n")

    # Helmet detection logic
    for result in helmet_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf >= 0.70:
                cls = int(box.cls[0])
                label = classNames[cls]
                cvzone.putTextRect(frame, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if label == 'Without Helmet':
                    for (px1, py1, px2, py2), plate_text in detected_plates.items():
                        send_warning_message(plate_text, 'Helmet Violation', '+917093660389')
                        with open("violations_log.txt", "a") as log_file:
                            log_file.write(f"{time.ctime()} - Helmet Violation - {plate_text}\n")
# Change the zebra line color based on the traffic light status
    if traffic_light_red:
        line_color = (0, 0, 255)  # Red
    else:
        line_color = (0, 255, 0)  # Green

    cv2.line(frame, (0, zebra_line_y), (frame.shape[1], zebra_line_y), line_color, 2)

    out.write(frame)
    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        break



cap.release()
out.release()
cv2.destroyAllWindows()