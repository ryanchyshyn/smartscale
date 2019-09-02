#!/usr/bin/python3

import cv2
import os
import numpy
import numpy as np
import argparse
from picamera.array import PiRGBArray
from picamera import PiCamera

import mvnc.mvncapi as mvnc

# Number of top predictions to print
NUM_PREDICTIONS = 2

# Variable to store commandline arguments
ARGS = None
cam = None
rawCapture = None


def open_ncs_device():

    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print("No devices found")
        quit()

    device = mvnc.Device(devices[0])
    device.OpenDevice()

    return device


def load_graph(device):
    # Read the graph file into a buffer
    with open(ARGS.graph, mode='rb') as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph(blob)

    return graph


def getPerspectiveMatrixes(w, h):
    srcPoints = np.float32(
        [[w * 0.301, h * 0.883],
         [w * 0.329, h * 0.25],
         [w * 0.831, h * 0.364],
         [w * 0.842, h * 0.854]])
    dstPoints = np.float32(
        [[0, h],
         [0, 0],
         [w, 0],
         [w, h]])
    
    M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    Minv = cv2.getPerspectiveTransform(dstPoints, srcPoints)

    return M, Minv


def warpImage(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_AREA)  # keep same size as input image
    return warped


def pre_process_image(img):

    # crop image
    #height, width, channels = img.shape
    #crop_size = 0.4

    #crop_y1 = int(height * (1 - crop_size) / 2)
    #crop_height = int(height * crop_size)
    #crop_x1 = int(width * (1 - crop_size) / 2)
    #crop_width = int(width * crop_size)

    #img = img[crop_y1:crop_y1 + crop_height, crop_x1:crop_x1 + crop_width]

    # Resize image [Image size is defined during training]
    img = cv2.resize( img, tuple(ARGS.dim) )

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    if( ARGS.colormode == "bgr" ):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img


def infer_image(graph, img, frame):

    # Load the image as a half-precision floating point array
    graph.LoadTensor(img, 'user object')

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Sort the indices of top predictions
    #order = output.argsort()[::-1][:NUM_PREDICTIONS]
    top = output.argmax()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        cv2.putText(frame, labels[top] + " %3.1f%%" % (100.0 * output[top]), (25, 25), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 255))
        cv2.imshow('live inference', frame)


def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()
    cv2.destroyAllWindows()


def main():

    device = open_ncs_device()
    graph = load_graph(device)

    M, Minv = getPerspectiveMatrixes(640, 480)

    while (True):
        #ret, frame = cam.read()
        cam.capture(rawCapture, format="bgr")
        frame = rawCapture.array
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # warp image
        frame = warpImage(frame, M)

        # crop image
        #height, width, channels = frame.shape
        #crop_size = 0.6

        #crop_y1 = int(height * (1 - crop_size) / 2)
        #crop_height = int(height * crop_size)
        #crop_x1 = int(width * (1 - crop_size) / 2)
        #crop_width = int(width * crop_size)

        # crop
        #frame = frame[crop_y1:crop_y1 + crop_height, crop_x1:crop_x1 + crop_width]

        infer_data = pre_process_image( frame )
        infer_image( graph, infer_data, frame )

        rawCapture.truncate(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    close_ncs_device( device, graph )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Image classifier using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/GoogLeNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../data/ilsvrc12/synset_words.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[104.00698793, 116.66876762, 122.67891434],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=1,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[224, 224],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="bgr",
                         help="RGB vs BGR color sequence. This is network dependent." )

    ARGS = parser.parse_args()

    # Load the labels file
    labels = [line.rstrip('\n') for line in
              open(ARGS.labels) if line != 'classes\n']

    cam = PiCamera()
    cam.resolution = (640, 480)
    rawCapture = PiRGBArray(cam, size=(640, 480))

    main()
