import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText  # Added for message
from email import encoders
import cv2
import mediapipe as mp
import os
import face_recognition
import ssl

# Specify the folder path where you want to save the images
save_folder = "images"  # Change this to your desired folder path
crm_data = 'crmlist'
crm_files = os.listdir(crm_data)

# Load the face encodings of criminal images from all directories
crm_face_encodings = []

# Create the folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils
fingercoordinates = [(0, 0), (8, 6), (12, 10), (16, 14), (20, 18)]
thumcoordinates = (4, 2)
effect = None

if not cap.isOpened():
    raise IOError("Cannot open webcam")
print("Capture(c)\nQuit(q)")

# Setup port number and server name
smtp_port = 587  # Standard secure SMTP port
smtp_server = "smtp.gmail.com"  # Google SMTP Server

# Set up the email lists
email_from = "icsy200325@gmail.com"  # Replace with your email address
email_list = ["icsy200325@gmail.com"]  # Replace with recipient email address

# Define the password (better to reference externally)
pswd = "ksgw zpqk easf uhhd"  # Replace with your email password

# Create context
simple_email_context = ssl.create_default_context()

# Create a MIME object to define parts of the email (outside the loop)
msg = MIMEMultipart()
msg['From'] = email_from
msg['Subject'] = "Attachment and Message"
message = ""

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandmarks = results.multi_hand_landmarks

    if multiLandmarks:
        handpoints = []
        for handLms in multiLandmarks:
            mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handpoints.append((cx, cy))

        for point in handpoints:
            cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)

        upCount = 0
        for coordinates in fingercoordinates:
            if handpoints[coordinates[0]][1] < handpoints[coordinates[1]][1]:
                upCount += 1
        if handpoints[thumcoordinates[0]][0] > handpoints[thumcoordinates[1]][0]:
            upCount += 1

        cv2.putText(img, str(upCount), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)

        # Check if one finger is pointed to switch to grayscale
        if upCount == 1:
            effect = 'GRAY'
        elif upCount == 2:
            effect = 'HSV'
        elif upCount == 3:
            effect = 'YUV'
        elif upCount == 4:
            effect = 'HSV2'
        elif upCount == 5:
            effect = 'YUV2'
        else:
            effect = None

    # Apply the selected filter
    if effect == 'GRAY':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif effect == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif effect == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif effect == 'HSV2':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = cv2.split(img)[0]  # Extract the existing H channel
        img = cv2.merge([img[:, :, 1], img[:, :, 2], h_channel])
    elif effect == 'YUV2':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = cv2.split(img)[0]  # Extract the existing Y channel
        img = cv2.merge([img[:, :, 1], img[:, :, 2], y_channel])
    # Add more filter conditions here if needed

    cv2.imshow('Combined', img)
    c = cv2.waitKey(1)

    if c == ord('q'):
        break
    if c == ord('c'):
        filenm = input("Enter your name or Quit(q): ")
        checking = os.path.join(save_folder, f"{filenm}.png")
        cv2.imwrite(checking, img)
        print(f"Captured and saved '{filenm}.png' with {effect} effect")

        # Load and compare the test image to criminal images
        test_image = face_recognition.load_image_file(checking)
        rgb_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        # Get face locations in the test image
        face_locations = face_recognition.face_locations(rgb_test_image)

        if len(face_locations) > 0:
            # If at least one face is detected, encode the first face
            test_face_encoding = face_recognition.face_encodings(rgb_test_image)[0]

            # Compare the face encoding of the test image with the list of criminal face encodings
            for crm_file in crm_files:
                crm_image = face_recognition.load_image_file(os.path.join(crm_data, crm_file))
                crm_face_encoding = face_recognition.face_encodings(crm_image)[0]
                result = face_recognition.compare_faces([crm_face_encoding], test_face_encoding)
                if result[0]:
                    print(f"{filenm} have a criminal record.")
                    # Open the file in python as a binary
                    attachment = open(checking, 'rb')  # 'r' for read and 'b' for binary

                    # Encode as base64
                    attachment_package = MIMEBase('application', 'octet-stream')
                    attachment_package.set_payload(attachment.read())
                    encoders.encode_base64(attachment_package)
                    attachment_package.add_header('Content-Disposition', "attachment; filename= " + checking)
                    msg.attach(attachment_package)

                    # Add a message to the email body
                    message = f"{filenm} have a criminal record."

                    # Connect with the server (moved outside the loop)
                    print("Connecting to server...")
                    TIE_server = smtplib.SMTP(smtp_server, smtp_port)
                    TIE_server.starttls()
                    TIE_server.login(email_from, pswd)
                    print("Successfully connected to the server")
                    print()

                    # Send the email with the attachment and message
                    msg.attach(MIMEText(message, 'plain'))
                    TIE_server.sendmail(email_from, email_list, msg.as_string())
                    print(f"Attachment and message sent to: {email_list}")
                    print("found ya!")
                    print()
                    attachment.close()  # Close the attachment file
                    break
            else:
                print("No criminal record.")
        else:
            print("No face detected in the test image")

# Close the SMTP connection
TIE_server.quit()

cap.release()
cv2.destroyAllWindows()
