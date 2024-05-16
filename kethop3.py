import cv2
import numpy as np
import math
import time
import os
from datetime import datetime
import threading
import sys
import glob
import importlib.util
import argparse
import RPi.GPIO as GPIO

# Define GPIO pins for each traffic light module
PIN_RED = [13, 0, 10, 17]
PIN_YELLOW = [6, 11, 22, 4]
PIN_GREEN = [5, 9, 27, 3]

# CLock
# Set up pins
sdi_pins = [24, 25, 12, 21]
rclk_pins = [18, 7, 20, 19] #LOAD
srclk_pins = [23, 8, 16, 26]

# Define segment codes from 0 to 9 in Hexadecimal
seg_code = [0xC0, 0xF9, 0xA4, 0xB0, 0x99, 0x92, 0x82, 0xF8, 0x80, 0x90]
counter = [10, 10, 10, 10]


# Set up GPIO
GPIO.setmode(GPIO.BCM)
# Set up GPIO pins for each traffic light module
for pin in PIN_RED + PIN_YELLOW + PIN_GREEN:
    GPIO.setup(pin, GPIO.OUT)

# Initialize counters for each vehicle type
vehicle_counts = {"car": 0, "motor": 0, "bus": 0, "truck": 0}
#total_vehicle = 0

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)',
                    action='store_false')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

#save_results = args.save_results # Defaults to False
show_results = args.noshow_results # Defaults to True


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])


# read image
# img = cv2.imread('images/image7.jpg')

# Capture image from camera
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

# Save image captured and detected
def save_image(image):
    # Generate image file name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"image_{timestamp}.jpg"
    # check folder exists
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")
    # filepath
    filepath = os.path.join("captured_images", filename)
    cv2.imwrite(filepath, image)
    print("saved image", filepath)

class VehicleDetector:
    # detect function
    def detect_vehicles(self, img):
        # Load the Tensorflow Lite model.
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        outname = output_details[0]['name']

        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0

        # Load image and resize to expected shape [1xHxWx3]
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imH, imW, _ = img.shape 
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        detections = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                #total_vehicle += 1
                if object_name == 'car':
                    vehicle_counts["car"] += 1
                elif object_name == 'motor':
                    vehicle_counts["motor"] += 1
                elif object_name == 'bus':
                    vehicle_counts["bus"] += 1
                elif object_name == 'truck':
                    vehicle_counts["truck"] += 1
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
            

        # Print the total count for each vehicle type 
        for vehicle_type, count in vehicle_counts.items(): 
                print(f"Total {vehicle_type}: {count}")
            
        # print("total vehicle", total_vehicle)   

        # cv2.imshow('Object detector', img)
        # cv2.waitKey(0)


### code control traffic signal ###
# From above code we have the numbers of vehicles of each class, variable "vehicle_counts["class_name"]"

# Default values of signal times
defaultRed = 80
defaultYellow = 5
defaultGreen = 10
defaultMinimum = 10
defaultMaximum = 60

temp = 0 # for detect second time

signals = []
noOfSignals = 4

currentGreen = 0   # Indicates which signal is green (like pointer)
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0   # Indicates whether yellow signal is on or off

# Average times for vehicles to pass the intersection, modify this later
carTime = 3
bikeTime = 2
busTime = 3
truckTime = 3

# Red signal time at which cars will be detected at a signal
detectionTime = 5   # prevent two settime running at the same time

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0



# Set time according to formula
def setTime():
    print("In settime(): \n")
    global vehicle_counts, total_vehicle
    global carTime, busTime, truckTime, bikeTime
    vehicle_counts = {"car": 0, "motor": 0, "bus": 0, "truck": 0}       # reset counter for new image
    #total_vehicle = 0
    # Load Veichle Detector
    vd = VehicleDetector()

    # Shots the road
    img = capture_image()

    # call detect func
    vd.detect_vehicles(img)
    save_image(img)

    # calculate time
    greenTime = math.ceil((vehicle_counts["car"]*carTime) + (vehicle_counts["motor"]*bikeTime) + (vehicle_counts["bus"]*busTime)+ (vehicle_counts["truck"]*truckTime))  #formula
    print('Green Time: ',greenTime)
    # normalize
    if(greenTime<defaultMinimum):
        greenTime = defaultMinimum
    elif(greenTime>defaultMaximum):
        greenTime = defaultMaximum

    print("Green time after normalize", greenTime)
        
    # set time for next green
    signals[(currentGreen+1)%(noOfSignals)].green = greenTime

def setTime2():
    print("In settime2(): \n")
    global vehicle_counts
    global carTime, busTime, truckTime, bikeTime
    global temp
    vehicle_counts = {"car": 0, "motor": 0, "bus": 0, "truck": 0}       # reset counter for new image
    # Load Veichle Detector
    vd = VehicleDetector()

    # Shots the road
    img = capture_image()

    # call detect func
    vd.detect_vehicles(img)
    save_image(img)

    # calculate time
    greenTime = math.ceil((vehicle_counts["car"]*carTime) + (vehicle_counts["motor"]*bikeTime) + (vehicle_counts["bus"]*busTime)+ (vehicle_counts["truck"]*truckTime))  #formula
    print('Green Time: ',greenTime)
    # normalize
    if(greenTime<defaultMinimum):
        greenTime = defaultMinimum
    elif(greenTime>defaultMaximum):
        greenTime = defaultMaximum

    print("Green time after normalize", greenTime)
        
    # set time for current green
    signals[currentGreen].green = math.ceil((greenTime+temp) /2)
    # set time for next red
    signals[(currentGreen+1)%(noOfSignals)].red = signals[currentGreen].yellow+signals[currentGreen].green


def repeat():
    print("repeat run")
    global currentGreen, currentYellow, nextGreen, temp
    flag = False    # use for the second detection, second set time
    temp = signals[currentGreen].green/2
    while(signals[currentGreen].green>0):   # while the timer of current green signal is not zero  
        printStatus()
        updateValues()
        if(signals[(currentGreen+1)%(noOfSignals)].red==detectionTime):    # set time of next green signal 
            thread = threading.Thread(name="detection",target=setTime, args=())
            # thread.daemon = True
            thread.start()
            # setTime()

        # After 1/2 timer set time again for current signal
        if(flag == False):
            if(signals[currentGreen].green == temp):
                thread = threading.Thread(name="detection2",target=setTime2, args=())
                thread.start()
                flag = True

        time.sleep(1)
    currentYellow = 1   # set yellow signal on
    
    while(signals[currentGreen].yellow>0):  # while the timer of current yellow signal is not zero
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0   # set yellow signal off
    
    # reset all signal times of current signal to default times
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen # set next signal as green signal
    nextGreen = (currentGreen+1)%noOfSignals    # set next green signal
    signals[nextGreen].red = signals[currentGreen].yellow+signals[currentGreen].green    # set the red time of next to next signal as (yellow time + green time) of next signal
    repeat()  


def printStatus():
    global counter
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                print(" GREEN TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
                counter[i] = signals[i].green
                print ("counter : %d"%(counter[i]))
            else:
                print("YELLOW TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
                counter[i] = signals[i].yellow
                print ("counter : %d"%(counter[i]))
        else:
            print("   RED TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
            counter[i] = signals[i].red
            print ("counter : %d"%(counter[i]))
    print()

# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):               # if yellow light off
                signals[i].green-=1             # decreasing time for green
                signals[i].totalGreenTime+=1
            else:
                signals[i].yellow-=1            
        else:
            signals[i].red-=1


# Initialization of signals with default values
def initialize():
    print("initialize func run")
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    print("\nline befor repeat()")
    repeat()

# Clock func
def setup():
    GPIO.setmode(GPIO.BCM)
    for sdi_pin, rclk_pin, srclk_pin in zip(sdi_pins, rclk_pins, srclk_pins):
        GPIO.setup(sdi_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(rclk_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(srclk_pin, GPIO.OUT, initial=GPIO.LOW)


# Shift the data to 74HC595
# def hc595_shift(sdi_pin, rclk_pin, srclk_pin, dat):
#     for bit in range(0, 8):
#         GPIO.output(sdi_pin, 0x80 & (dat << bit))
#         GPIO.output(srclk_pin, GPIO.HIGH)
#         GPIO.output(srclk_pin, GPIO.LOW)
#     GPIO.output(rclk_pin, GPIO.HIGH)
#     GPIO.output(rclk_pin, GPIO.LOW)
    
def hc595_shift(sdi_pin, rclk_pin, srclk_pin, data):
    # Dịch bit cho chữ số hàng chục
    units_digit = data % 10
    for bit in range(0, 8):
        GPIO.output(sdi_pin, (0x80 & (seg_code[units_digit] << bit)))
        GPIO.output(srclk_pin, GPIO.HIGH)

        GPIO.output(srclk_pin, GPIO.LOW)
    
    # Dịch bit cho chữ số hàng đơn vị
    tens_digit = data // 10
    for bit in range(0, 8):
        GPIO.output(sdi_pin, (0x80 & (seg_code[tens_digit] << bit)))
        GPIO.output(srclk_pin, GPIO.HIGH)

        GPIO.output(srclk_pin, GPIO.LOW)
    
    # Latch dữ liệu
    GPIO.output(rclk_pin, GPIO.HIGH)
    GPIO.output(rclk_pin, GPIO.LOW)


def clearDisplay(sdi_pin, rclk_pin, srclk_pin):
    for i in range(8):
        GPIO.output(sdi_pin, 1)
        GPIO.output(srclk_pin, GPIO.HIGH)
        GPIO.output(srclk_pin, GPIO.LOW)
    GPIO.output(rclk_pin, GPIO.HIGH)
    GPIO.output(rclk_pin, GPIO.LOW)


# def display_number(set_index, num):
#     tens_digit = num % 100//10
#     units_digit = num % 10
#     sdi_pin = sdi_pins[set_index]
#     rclk_pin = rclk_pins[set_index]
#     srclk_pin = srclk_pins[set_index]
#     # Shift data for units digit
#     clearDisplay(sdi_pin, rclk_pin, srclk_pin)
#     hc595_shift(sdi_pin, rclk_pin, srclk_pin, seg_code[units_digit])
#     time.sleep(0.03)
#     
#     # Shift data for tens digit
#     hc595_shift(sdi_pin, rclk_pin, srclk_pin, seg_code[tens_digit])
#     time.sleep(0.03)
    
def display_number(set_index, num):
    sdi_pin = sdi_pins[set_index]
    rclk_pin = rclk_pins[set_index]
    srclk_pin = srclk_pins[set_index]
    hc595_shift(sdi_pin, rclk_pin, srclk_pin, num)


def loop():
    while True:
        for set_index in range(len(sdi_pins)):
            display_number(set_index, counter[set_index])
        

def destroy():   # When "Ctrl+C" is pressed, the function is executed.
    global timer1
    GPIO.cleanup()
    timer1.cancel()      #cancel the timer



def controlLED():
    # global yellowLight, redLight, greenLight
    while True: 
        for i in range(0,noOfSignals):  # display signal and set timer according to current status: green, yello, or red
            if(i==currentGreen):
                if(currentYellow==1):
                    
                        # continue yellow timer
                    signals[i].signalText = signals[i].yellow
                    #yellowLight = signals[i].yellow
                    # turn on YELLOW
                    GPIO.output(PIN_RED[i], GPIO.LOW)
                    GPIO.output(PIN_YELLOW[i], GPIO.HIGH)
                    GPIO.output(PIN_GREEN[i], GPIO.LOW)
                else:
                    
                    # continue green timer
                    signals[i].signalText = signals[i].green
                    #greenLight = signals[i].green
                    # turn on GREEN
                    GPIO.output(PIN_RED[i], GPIO.LOW)
                    GPIO.output(PIN_YELLOW[i], GPIO.LOW)
                    GPIO.output(PIN_GREEN[i], GPIO.HIGH)
                    
            else: # others traffic signal
                #if(signals[i].red<=10): # if that LED <= 10
                    
                    # continue RED timer
                signals[i].signalText = signals[i].red
                #redLight = signals[i].red
                #else:
                    # just display "---"
                    #signals[i].signalText = 99
                # turn on RED
                GPIO.output(PIN_RED[i], GPIO.HIGH)
                GPIO.output(PIN_YELLOW[i], GPIO.LOW)
                GPIO.output(PIN_GREEN[i], GPIO.LOW)
        
        # yellowLight = signals[0].yellow
        # greenLight = signals[0].green
        # redLight = signals[0].red


class Main:
    print("Class main run")
    setup()
    thread4 = threading.Thread(name="controlClock",target=loop, args=())    # controlClock
    thread4.start()
    # initialize()
    thread2 = threading.Thread(name="initialization",target=initialize, args=())    # initialization
    # thread2.daemon = True
    thread2.start()

    thread3 = threading.Thread(name="controlLED",target=controlLED, args=())    # controlLED
    thread3.start()
    
try:
    Main()
except KeyboardInterrupt:
    destroy()

