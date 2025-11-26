# from ultralytics import YOLO
# import cv2

# model = YOLO('best.pt')
# cap = cv2.VideoCapture(0)  # 0 = default webcam

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame,conf=0.5)
#     annotated_frame = results[0].plot()

#     cv2.imshow("YOLOv8 Detection", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


from flask import Flask, render_template, request, jsonify
from waitress import serve
from PIL import Image
from ultralytics import YOLO 

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file", passes it
    through YOLOv8 object detection network and returns an object
    indicating whether a detection is present.
    :return: a JSON object indicating whether a detection is present
    """
    buf = request.files["image_file"]
    detection_present = detect_objects_on_image(buf.stream)
    return jsonify({"detection_present": detection_present})


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns a boolean indicating whether a detection is present.
    :param buf: Input image file stream
    :return: Boolean indicating whether a detection is present
    """
    model = YOLO("best.pt")
    results = model.predict(Image.open(buf))
    result = results[0]
    return len(result.boxes) > 0


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
