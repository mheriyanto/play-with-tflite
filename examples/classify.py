import argparse
import re

import cv2
from tflite_runtime.interpreter import Interpreter

from datetime import datetime
import time
import numpy as np

from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""

    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def generate(opt):
    start_time = time.time()
    time_threshold = 1  # second
    counter = 0
    fps_var = 0

    labels = load_labels(opt.labels)
    model = Interpreter(opt.model)
    model.allocate_tensors()
    _, input_height, input_width, _ = model.get_input_details()[0]['shape']

    cctv = opt.source
    cap = cv2.VideoCapture(cctv)
    while True:
        ret, frame = cap.read()
        if ret:
            h_img, _, _ = frame.shape
            
            # TODO: Core 
            resize = cv2.resize(frame, (input_width, input_height))
            results = classify_image(model, resize)
            print("results: ", results)

            label_id, prob = results[0]

            text = '%s %.2f' % (labels[label_id], prob)

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
                ("Object", text),
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
    parser.add_argument('--model', type=str, default='../saved/models/mobilenet_v1_1.0_224_quant.tflite', help='File path of .tflite file.', required=True)
    parser.add_argument('--labels', type=str, default='../saved/models/labels_mobilenet_quant_v1_224.txt', help='File path of labels file.', required=True)

    opt = parser.parse_args()

    app.run(host="0.0.0.0", port=5000, threaded=True)
    # python3 classify.py --source /dev/video0 --model ../saved/models/mobilenet_v1_1.0_224_quant.tflite --labels ../saved/models/labels_mobilenet_quant_v1_224.txt