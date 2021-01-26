import argparse
import re

import cv2
from tflite_runtime.interpreter import Interpreter
from PIL import Image

from datetime import datetime
import time
import numpy as np

from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def annotate_objects(img, results, labels):
    """Draws the bounding box and label for each object in the results."""
    CAMERA_HEIGHT, CAMERA_WIDTH, _ = img.shape

    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        # Overlay the box, label, and score on the camera preview
        # annotator.bounding_box([xmin, ymin, xmax, ymax])
        c1, c2 = (xmin, ymin), (xmax, ymax)
        cv2.rectangle(img, c1, c2, (0, 0, 255), thickness=2)     # Rectangle Object

        label = '%s %.2f' % (labels[obj['class_id']], obj['score'])
        
        if label:
            tl = round(0.002 * (CAMERA_HEIGHT + CAMERA_WIDTH) / 2) + 1  # line thickness
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl/5, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # annotator.text([xmin, ymin], '%s\n%.2f' % (labels[obj['class_id']], obj['score']))
            cv2.rectangle(img, c1, c2, (0, 0, 255), -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl/5, (255, 255, 255), thickness=int(tf), lineType=cv2.LINE_AA)


def generate(opt):
    start_time = time.time()
    time_threshold = 1  # second
    counter = 0
    fps_var = 0

    labels = load_labels(opt.labels)
    interpreter = Interpreter(opt.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cctv = opt.source
    cap = cv2.VideoCapture(cctv)
    while True:
        ret, frame = cap.read()
        if ret:
            h_img, _, _ = frame.shape
            
            # TODO: Core 
            resize = cv2.resize(frame, (input_width, input_height))
            results = detect_objects(interpreter, resize, opt.threshold)
            annotate_objects(frame, results, labels)

            now = datetime.now()
            now = '{}'.format(now.strftime("%d-%m-%Y %H:%M:%S"))

            counter += 1
            time_counter = time.time() - start_time
            if time_counter > time_threshold:
                fps_var = counter / time_counter
                fps_var = int(fps_var)
                counter = 0
                start_time = time.time()

            info_frame = [
                ("FPS", fps_var),
                ("Date", now)
            ]

            x_text = 0
            y_text = h_img
            for (i1, (k, v)) in enumerate(info_frame):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (int(x_text), int(y_text) - ((i1 * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            print('no frame')


@app.route('/video_feed')
def video_feed():
    return Response(generate(opt),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/dev/video0', help='path to input source', required=True) # input file/folder, 0 for webcam
    parser.add_argument('--model', type=str, default='weight/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite', help='File path of .tflite file.', required=True)
    parser.add_argument('--labels', type=str, default='weight/coco_labels.txt', help='File path of labels file.', required=True)
    parser.add_argument('--threshold', type=float, default=0.4, help='Score threshold for detected objects.', required=False)

    opt = parser.parse_args()

    app.run(host="0.0.0.0", port=5000, threaded=True)
    # python3 inference.py --source /dev/video0 --model weight/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite --labels weight/coco_labels.txt