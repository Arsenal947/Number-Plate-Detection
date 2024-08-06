# from ultralytics import YOLO
# import cv2
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt


# # Load a model
# model = YOLO('yolov8n.pt')  # load an official model
# # model = YOLO('path/to/best.pt')  # load a custom model

# # Predict with the model
# cap = cv2.VideoCapture('sample.mp4')

# ret = True
# frame_nmr = -1
# ret, frame = cap.read()
# while ret:
#     frame_nmr += 1
#     if frame_nmr > 10:
#         break
#     if ret:
#         dets = model(frame)[0]  # Get detections
#         for result in dets.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = result
#             threshold = 0.5

#             if score > threshold:
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#                 cv2.putText(frame, dets.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
#         print(dets.boxes.data.tolist())
#         # cv2.imshow("Yolo", frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#     # results = model(frame)  # predict on an image
    
import easyocr
import cv2

image = cv2.imread('plate.jpg')
# Initialize the OCR reader
reader = easyocr.Reader(['en'])
print(reader)
detections = reader.readtext(image)
print(detections)
for detection in detections:
    bbox, text, score = detection

    text = text.upper().replace(' ', '')
    print(text)

    # if license_complies_format(text):
    #     return format_license(text), score
