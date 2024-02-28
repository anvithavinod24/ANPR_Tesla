import cv2
import os

# Create a folder path
save_folder = r"C:\Users\v\Desktop\carplate"

# Check if the folder exists, and create it if not
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cam = cv2.VideoCapture(0)
cv2.namedWindow("Python Webcam Screenshot")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("Escape hit. Closing app")
        break
    elif k % 256 == 32:
        img_name = os.path.join(save_folder, "opencv_frame_{}.jpg".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("Screenshot taken and saved in", img_name)
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
