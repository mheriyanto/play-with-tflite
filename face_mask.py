# Ref: https://github.com/tanhouren/Face_mask_detector

import argparse
import re

import cv2
from tflite_runtime.interpreter import Interpreter

from datetime import datetime
import time
import numpy as np

from flask import Flask, render_template, Response

type_list = ['got mask', 'no mask','wear incorrectly']
WIDTH = 640
HEIGHT = 480

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def draw_and_show(box,classes,scores,num,frame):
	for i in range(int(num[0])):
		if scores[0][i] > 0.8:
			y,x,bottom,right = box[0][i]
			x,right = int(x*WIDTH),int(right*WIDTH)
			y,bottom = int(y*HEIGHT),int(bottom*HEIGHT)
			class_type=type_list[int(classes[0][i])]
			label_size = cv2.getTextSize(class_type,cv2.FONT_HERSHEY_DUPLEX,0.5,1)
			cv2.rectangle(frame, (x, y), (right, bottom), (0,255,0), thickness=2)
			cv2.rectangle(frame,(x,y-18),(x+label_size[0][0],y),(0,255,0),thickness=-1)
			cv2.putText(frame,class_type,(x,y-5),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
	return frame


def generate(opt):  
    start_time = time.time()
    time_threshold = 1  # second
    counter = 0
    fps_var = 0

    interpreter = Interpreter(opt.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
    shape_input = input_details[0]['shape'][1]

    cctv = opt.source
    cap = cv2.VideoCapture(cctv)

    while True:
        ret, frame = cap.read()
        if ret:
            h_img, _, _ = frame.shape
            
            # TODO: Core 
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_ = cv2.resize((img_*2/255)-1, (shape_input, shape_input))
            output_frame = img_[np.newaxis,:,:,:].astype('float32')

            interpreter.set_tensor(input_details[0]['index'], output_frame)
            interpreter.invoke()
            boxes = interpreter.get_tensor(output_details[0]['index'])
            classes = interpreter.get_tensor(output_details[1]['index'])
            scores = interpreter.get_tensor(output_details[2]['index'])
            num = interpreter.get_tensor(output_details[3]['index'])
            output = [boxes, classes, scores, num]

            frame = draw_and_show(output, frame)

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
    parser.add_argument('--model', type=str, default='weight/ssd_mobilenet_v2_fpnlite.tflite', help='File path of .tflite file.', required=True)
    parser.add_argument('--threshold', type=float, default=0.4, help='Score threshold for detected objects.', required=False)

    opt = parser.parse_args()

    app.run(host="0.0.0.0", port=5000, threaded=True)
    # python3 face_mask.py --source /dev/video0 --model weight/ssd_mobilenet_v2_fpnlite.tflite