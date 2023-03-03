# NOTE: way of communicating with each simulator varies
# this code is taken as it is -- as it is specific to this Udacity Simulator.
# comments are added for understanding.

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__)  # '__main__'
maxSpeed = 10


def preProcess(img):  # same as utils.preprocess()
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    # image that we receive from simulator
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])   # << meaning..??
    # Send the image to model and make prediction
    steering = float(model.predict(image))
    throttle = 1.0 - speed/maxSpeed  # limiting the speed to not exceed max speed
    print('steering: {}, throttle: {}, speed: {}'.format(
        steering, throttle, speed))
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected to simulator..!')
    sendControl(0, 0)   # initial controls..


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()  # << confirm here..
    })


if __name == '__main__':
    model = load_model('model.h5')  # << ensure the model path is correct
    app = socketio.Middleware(sio, app)
    # 4567 is the port number of this simulator
    eventlet.wsgi.server(eventlet.listen('', 4567), app)
