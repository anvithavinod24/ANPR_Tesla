import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import csv

import util

# Create a folder path
save_folder = r"C:\Users\v\Desktop\carplate"

# Check if the folder exists, and create it if not
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cam = cv2.VideoCapture(0)
cv2.namedWindow("Python Webcam Screenshot")
img_counter = 0

# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

input_dir = r"C:\Users\v\Desktop\carplate"

# load class names
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

# load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# Specify the CSV file path
csv_file_path = r"number_plate.csv"

# Read the CSV file and extract the values you want to check against
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_values = [row[0] for row in csv_reader]

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

        img_path = img_name

        # load image
        img = cv2.imread(img_path)
        H, W, _ = img.shape

        # convert image
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

        # get detections
        net.setInput(blob)
        detections = util.get_outputs(net)

        # bboxes, class_ids, confidences
        bboxes = []
        class_ids = []
        scores = []

        reader = easyocr.Reader(['en'])

        for detection in detections:
            # [x1, x2, x3, x4, x5, x6, ..., x85]
            bbox = detection[:4]

            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

            bbox_confidence = detection[4]

            class_id = np.argmax(detection[5:])
            score = np.amax(detection[5:])

            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)

        # apply nms
        bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

        # plot
        for bbox_, bbox in enumerate(bboxes):
            xc, yc, w, h = bbox
            license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
            img = cv2.rectangle(img,
                                (int(xc - (w / 2)), int(yc - (h / 2))),
                                (int(xc + (w / 2)), int(yc + (h / 2))),
                                (0, 255, 0),
                                10)

            license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
            _, license_plate_thresh = cv2.threshold(license_plate_gray, 66, 255, cv2.THRESH_BINARY_INV)

            output = reader.readtext(license_plate_gray)
            for out in output:
                text_bbox, text, text_score = out
                print("Detected Number Plate:", text, "Score:", text_score)

                # Check if the detected number plate is in the CSV values
                if text in csv_values:
                    print("Yes, present in CSV file.")
                else:
                    print("No, not present in CSV file.")

        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.figure()
        plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))

        plt.figure()
        plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))

        plt.figure()
        plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))

        plt.show()

        img_counter += 1

cam.release()
cv2.destroyAllWindows()
