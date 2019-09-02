#!/usr/bin/python3

import time
import cv2
import numpy
import argparse
import copy
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
from tm1637 import TM1637
from io import BytesIO
from PIL import Image

import mvnc.mvncapi as mvnc

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time


#formHtml = b'<img src="camera"/><br><button onClick="window.location.reload();">Refresh</button><form action="photo"><button type="submit">Photo</button></form>'
formHtml = b'<img id="cam_img" src="img_org"/><br><div><input id="img_type" type="checkbox" onclick="document.getElementById(\'btn_refresh\').click()"/><label for="img_type">Warped</label></div><button id="btn_refresh" style="width:200px;height:80px;" onClick="if (document.getElementById(\'img_type\').checked) document.getElementById(\'cam_img\').src = \'img_warped\' + (new Date()).getTime(); else document.getElementById(\'cam_img\').src = \'img_org\' + (new Date()).getTime();">Refresh</button><button style="width:200px;height:80px;" onClick="url = (document.getElementById(\'img_type\').checked) ? \'/photo_warped\' : \'/photo_org\'; http = new XMLHttpRequest(); http.open(\'GET\', url); http.send();">Photo</button>'
PORT = 8054

# Number of top predictions to print
NUM_PREDICTIONS = 2

# Variable to store commandline arguments
ARGS = None
cam = None
rawCapture = None

# GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

LAST_IMG_DATA_WARPED = None
LAST_IMG_DATA_ORG = None
IMAGE_LOCK = threading.Lock()


# HTTP server
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    global LAST_IMG_DATA_WARPED
    global LAST_IMG_DATA_ORG
    global IMAGE_LOCK

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', "text/html")
            self.end_headers()
            self.wfile.write(formHtml)

        elif self.path == "/photo_org":
            if LAST_IMG_DATA_ORG is not None:
                IMAGE_LOCK.acquire()
                image = Image.fromarray(LAST_IMG_DATA_ORG)
                IMAGE_LOCK.release()
                file_name = "image_org_" + str(int(time.time())) + ".jpeg"
                image.save(file_name, format='JPEG')

                print("Image " + file_name + " saved")

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<body>ok</body>")

            else:
                self.send_error(404, 'No image')

        elif self.path == "/photo_warped":
            if LAST_IMG_DATA_WARPED is not None:
                IMAGE_LOCK.acquire()
                image = Image.fromarray(LAST_IMG_DATA_WARPED)
                IMAGE_LOCK.release()
                file_name = "image_warped_" + str(int(time.time())) + ".jpeg"
                image.save(file_name, format='JPEG')

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<body>ok</body>")

                print("Image " + file_name + " saved")

            else:
                self.send_error(404, 'No image')

        elif self.path.startswith("/img_warped"):
            if LAST_IMG_DATA_WARPED is not None:
                # create JPG in memory
                IMAGE_LOCK.acquire()
                image_bytes = BytesIO()
                image = Image.fromarray(LAST_IMG_DATA_WARPED)
                image.save(image_bytes, format='JPEG')
                img_content = image_bytes.getvalue()
                IMAGE_LOCK.release()

                self.send_response(200)
                self.send_header('Content-type', "image/jpg")
                self.end_headers()
                self.wfile.write(img_content)

            else:
                self.send_error(404, 'No image')

        elif self.path.startswith("/img_org"):
            if LAST_IMG_DATA_ORG is not None:
                # create JPG in memory
                IMAGE_LOCK.acquire()
                image_bytes = BytesIO()
                image = Image.fromarray(LAST_IMG_DATA_ORG)
                image.save(image_bytes, format='JPEG')
                img_content = image_bytes.getvalue()
                IMAGE_LOCK.release()

                self.send_response(200)
                self.send_header('Content-type', "image/jpg")
                self.end_headers()
                self.wfile.write(img_content)

            else:
                self.send_error(404, 'No image')

def open_ncs_device():
    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print("No devices found")
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    return device


def load_graph(device):
    # Read the graph file into a buffer
    with open(ARGS.graph, mode='rb') as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    return device.AllocateGraph(blob)


def get_perspective_matrixes(w, h):
    src_points = numpy.float32(
        [[w * 0.301, h * 0.883],
         [w * 0.329, h * 0.25],
         [w * 0.831, h * 0.364],
         [w * 0.842, h * 0.854]])
    dst_points = numpy.float32(
        [[0, h],
         [0, 0],
         [w, 0],
         [w, h]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)

    return M, Minv


def warp_image(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_AREA)  # keep same size as input image
    return warped


def pre_process_image(img):
    global LAST_IMG_DATA_WARPED
    global LAST_IMG_DATA_ORG
    global IMAGE_LOCK

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    img = img[:, :, ::-1]

    # warp image
    img_warped = warp_image(img, M)

    IMAGE_LOCK.acquire()
    LAST_IMG_DATA_ORG = copy.copy(img)
    LAST_IMG_DATA_WARPED = copy.copy(img_warped)
    IMAGE_LOCK.release()

    img = img_warped

    # crop image
    # height, width, channels = img.shape
    # crop_size = 0.6

    # crop_y1 = int(height * (1 - crop_size) / 2)
    # crop_height = int(height * crop_size)
    # crop_x1 = int(width * (1 - crop_size) / 2)
    # crop_width = int(width * crop_size)

    # img = img[crop_y1:crop_y1 + crop_height, crop_x1:crop_x1 + crop_width]

    # Resize image [Image size is defined during training]
    img = cv2.resize(img, tuple(ARGS.dim))

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype(numpy.float16)
    img = (img - numpy.float16(ARGS.mean)) * ARGS.scale

    return img


def infer_image(graph, img, frame, captureTime):
    # Load the image as a half-precision floating point array
    graph.LoadTensor(img, 'user object')

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Sort the indices of top predictions
    # order = output.argsort()[::-1][:NUM_PREDICTIONS]
    top = output.argmax()

    # Get execution time
    inference_time = graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN)

    print(labels[top] + " %3.1f%% time=%d+%dms            \r" % (100.0 * output[top], int(captureTime * 1000), numpy.sum(inference_time)), end='')
    top_fixed = top + 1
    if (output[top] < 0.9) or (labels[top] == "empty"):
        top_fixed = 0

    if top_fixed == 0:
        display.Clear()
    else:
        code = int(codes[top])
        display.ShowInt(code)


def close_ncs_device(device, graph):
    graph.DeallocateGraph()
    device.CloseDevice()


def main():
    start_time = 0

    for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = frame.array

        img = pre_process_image(image)
        end_time = time.time()

        infer_image(graph, img, image, end_time - start_time)

        rawCapture.truncate(0)
        start_time = time.time()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="SmartScale using \
                         Intel® Movidius™ Neural Compute Stick.")

    parser.add_argument('-g', '--graph', type=str,
                        default='../../caffe/GoogLeNet/graph',
                        help="Absolute path to the neural network graph file.")

    parser.add_argument('-l', '--labels', type=str,
                        default='../../data/ilsvrc12/synset_words.txt',
                        help="Absolute path to labels file.")

    parser.add_argument('-M', '--mean', type=float,
                        nargs='+',
                        default=[104.00698793, 116.66876762, 122.67891434],
                        help="',' delimited floating point values for image mean.")

    parser.add_argument('-S', '--scale', type=float,
                        default=1,
                        help="Absolute path to labels file.")

    parser.add_argument('-D', '--dim', type=int,
                        nargs='+',
                        default=[224, 224],
                        help="Image dimensions. ex. -D 224 224")

    ARGS = parser.parse_args()

    try:
        # Load the labels file
        labels = [line.rstrip('\n').split(':')[1] for line in
                  open(ARGS.labels) if line != 'classes\n']

        # load codes
        codes = [line.rstrip('\n').split(':')[2] for line in
                  open(ARGS.labels) if line != 'classes\n']
        

        display = TM1637(CLK=21, DIO=20, brightness=1.0)
        display.SetBrightness(1)
        display.Clear()

        M, Minv = get_perspective_matrixes(1640, 1232)

        device = open_ncs_device()
        graph = load_graph(device)

        cam = PiCamera()
        cam.resolution = (1640, 1232)
        cam.framerate = 5
        rawCapture = PiRGBArray(cam, size=(1640, 1232))

        # init HTTP server
        server = HTTPServer(("", PORT), SimpleHTTPRequestHandler)
        # server.serve_forever()

        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

        # fire
        main()

    except KeyboardInterrupt:
        close_ncs_device(device, graph)
        server.shutdown()
        server.server_close()
