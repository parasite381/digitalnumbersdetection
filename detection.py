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


from flask import Flask, request, jsonify
from flask_cors import CORS   # <-- import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)   # <-- enable CORS for all routes

model = YOLO("best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img, conf=0.5)
    return jsonify(results[0].tojson())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
