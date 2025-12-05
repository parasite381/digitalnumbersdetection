from flask import Flask, render_template, request, jsonify
from waitress import serve
from flask_cors import CORS 
from PIL import Image
from ultralytics import YOLO 

app = Flask(__name__)
CORS(app)

#load
model = YOLO("best.pt")

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
    boxes = detect_objects_on_image(buf.stream)
    return jsonify({"boxes": boxes, "detection_present": len(boxes) > 0})
    # return jsonify({"detection_present": detection_present})


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns a boolean indicating whether a detection is present.
    :param buf: Input image file stream
    :return: Boolean indicating whether a detection is present
    """
    
    results = model.predict(Image.open(buf),conf=0.5)
    boxes = []
    for box in results[0].boxes:
        # xyxy format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,"class_name": cls_name,"confidence": conf})

    return boxes

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)