import numpy as np
import cv2
#import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

#global variables
BLACK = 0           # value for black.
WHITE = 255         # the value for white.
LEFT = 1            # leftmost column on the screen.
RIGHT = 1279        # rightmost column on the screen.
TOP = 1             # top row on the screen.
BOTTOM = 1023       # bottom row on the screen. 
HALF = 400          # the column value for the half of the screen.
RIGHT_QUARTER = 1000   # the column value for the right quarter of the screen.
TOP_THIRD_VERT = 300
BOTTOM_THIRD_VERT = 600
PIXELS_IN_INCREMENT = 25
PRINT_RIGHT_SIDE = 850
column_scan = []    # list used as a global variable to store the column location during scans.
film = []     
column = 1280
grayValues = []
RGBValues = []


def FindRGBValues(filename):
    image = cv2.imread(filename,1)       # Set wetFrame to the image in the given file, where 1 allows for channels that show color.    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                              # turn it gray
    global RGBValues                        # Include global list to hold the RGB values.
    global grayValues                       # Include global list to hold the grayscale values.
    for i in range(LEFT,RIGHT):             # Scanning the screen from left to right.
        px = image[400,i]                   # Save the pixel at row 600 from the colored picture of the device.
        px2 = gray[400,i]                   # Save the pixel at row 600 from the gray picture of the device.
        RGBValues.append(px)                # Append the value of the tuple [red, green, blue] to the RGBValues list. 
        grayValues.append(px2)              # Append the value of the grayscale pixel to the list.
    cv2.line(gray, (200,399), (1000,399), (178, 34, 34), 1)                   # Draw a line to show the scanning region.
    cv2.imshow("Device",image)
    cv2.imshow("Device",gray)
    cv2.waitKey(0)                          # Wait for a key press from the user.
    dataGray = np.array(grayValues)         # Convert a list to a numpy array called dataGray
    dataRGB = np.array(RGBValues)           # Convert a list to a numpy array/matrix called dataRGB
    
    START_OF_SCAN = 770
    LENGTH_OF_SCAN = 40
    gray = np.zeros(LENGTH_OF_SCAN)
    uncertaintyRange = np.zeros(LENGTH_OF_SCAN)
    for i in range(0,LENGTH_OF_SCAN):                  # For 100 increments, store data points from the numpy arrays to the important portion.
        gray[i] = dataGray[i+START_OF_SCAN]           # Gray just needs the row to be selected properly, it is not a matrix like the dataRGB numpy array is.
    for i in range(15,24):
        uncertaintyRange[i] = gray[i]
    leftSum = 0
    leftAvg = 227.43

    rightSum = 0
    rightAvg = 21.17
    print("leftAvg: ",leftAvg," rightAvg: ",rightAvg)

    midpoint = (rightAvg + leftAvg) / 2
    theRange = leftAvg - rightAvg 
    fortyPercent = .4 * theRange
    print("midpoint: ", midpoint, "range: ", theRange, "forty percent: ", fortyPercent)

    x1, y1 = [26,28] , [midpoint-fortyPercent,midpoint-fortyPercent]
    x2, y2 = [26,28] , [midpoint+fortyPercent,midpoint+fortyPercent]
    x3, y3 = [17,19] , [midpoint,midpoint]
    x5, y5 = [28,28] , [midpoint-fortyPercent, midpoint+fortyPercent]
    plt.plot(x1,y1,'b',x2,y2,'b',x3,y3,'r',x5,y5,'b')

    uncertainty = (9 * .177277418) / 2
    print("uncertainty plus or minus: ", uncertainty)

    # For plotting microns on the x-axis
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.plot(gray,'ko')
    ax1.plot(uncertaintyRange,'bo')
    width = np.arange(len(gray))
    ax1.set_xlabel('Microns (um)',fontsize = 24)
    ax1.set_ylabel('Grayscale Pixel Value (0-255)',fontsize = 24)
    ax1.set_title('Pixel Uncertainty in the Edge of the Device',fontsize = 28)
  
    def format_func(x,y):
        scalebar = .165
        return x * scalebar

    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))


    # plt.plot(gray,'ko')                     # Plot the gray values with gray dots.
    # plt.plot(uncertaintyRange,'bo')
    # # plt.plot(112,gray[112],'rs')
    # # plt.plot(127,gray[127],'rs')
    # plt.title('Pixel Uncertainty in the Edge of the Device',fontsize = 28)
    # plt.ylabel('Grayscale Pixel Value (0-255)',fontsize=24)     # 
    # plt.xlabel('Microns (um)',fontsize=24)
    # plt.xticks(np.arange(0,6.6,0.5))
    plt.show()                              # Show the plot, it also waits for a 'q' input from the user.
    cv2.destroyAllWindows()                 # Before leaving the RGBValues function, clear all windows.


FindRGBValues("Dry2.jpg")


