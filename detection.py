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


# from flask import Flask, request, jsonify
# from flask_cors import CORS   # <-- import CORS
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# import json

# app = Flask(__name__)
# CORS(app)   # <-- enable CORS for all routes

# model = YOLO("best.pt")

# @app.route("/predict", methods=["POST"])
# def predict():
#     file = request.files["image"]
#     img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#     if img is None:
#         return jsonify({"error": "Invalid image"}), 400
#     results = model(img, conf=0.5)
#     return jsonify(json.loads(results[0].tojson()))



# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)


from flask import Flask, render_template, request, jsonify
from waitress import serve
from flask_cors import CORS 
from PIL import Image
from ultralytics import YOLO 

app = Flask(__name__)
CORS(app, resources={r"/detect": {"origins": "http://127.0.0.1:5500"}})

@app.route("/")
def root():
    return render_template("index.html")
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