# This code will measure the whole device from top to bottom in increments, then prints out the film depth to get a general idea of the film depth

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# GLOBAL VARIABLES and DEFINITIONS
BLACK = 0                                                                                           # Value for black.
WHITE = 255                                                                                         # The value for white.
LEFT = 1                                                                                            # Leftmost column on the screen.
RIGHT = 1279                                                                                        # Rightmost column on the screen.
TOP = 1                                                                                             # Top row on the screen.
BOTTOM = 1023                                                                                       # Bottom row on the screen. 
LEFT_SIDE = 400                                                                                     # The column value for the left quarter of the screen. We will scan between the two quartiles.
RIGHT_SIDE = 1000                                                                                   # The column value for the right quarter of the screen.
SHOW = 1                                                                                            # If the show variable (eg showFiber) is one, display more detail.
HIDE = 0                                                                                            # If the show variable is zero, refrain from displaying screens.
MEAN = 2
TOP_THIRD_VERT = 300
BOTTOM_THIRD_VERT = 600
PIXELS_IN_INCREMENT = 25
PRINT_RIGHT_SIDE = 1150
column_scan = []    # list used as a global variable to store the column location during scans.
film = []     

# Function for finding the width of the fiber in pixels, then finding the width of each pixel in microns
# based on the known width of the fiber as 125 microns.
def FiberWidthTest(frame, show):
    start = []                                                                                      # Initializing empty lists
    end = []
    width = []
    fiber_in_microns = 125                                                                          # The actual width of the fiber is known to be 125 microns.
    fiber_width = 0                                                                                 # Number of pixels.
    width_sum = 0                                                                                   # Sum of the width measurements.

    for i in range (LEFT,RIGHT):                                                                    # Scan all columns from left to right.
        for j in range(TOP,BOTTOM):                                                                 # Scan all rows fromt top to bottom.
            px = frame[j,i]                                                                         # Save the pixel value, where [y,x] since it is inverted.
            if j != BOTTOM:                                                                         # If the for loop is not on the last pixel, save the upcoming pixel.
                pxplusone = frame[j+1, i]                                                           # Save the pixel to the right of our current position in the scan
            if px == WHITE:                                                                         # If we are on a white pixel.
                if pxplusone == BLACK:                                                              # And the next one is black.
                    start.append(j + 1)                                                             # Add the pixel location to the 'start' list.
            if px == BLACK:                                                                         # If the for loop is on a black pixel,
                if pxplusone == WHITE:                                                              # and the next is white,
                    end.append(j)                                                                   # add the value of the white pixel to 'end' list.
        width.append(end[len(end)-1]-start[0])                                                      # Finds the width by taking the last value from the end list and the first value of the start list, then appends the difference.
        if show == 2:                                                                               # If the show input var is 2:
            print ('Fiber Width: ', width)                                                          # print to see the different values measured.
        start.clear()                                                                               # Clears list to measure the next column.
        end.clear()                                                                                 # Clear list.

    for x in range (len(width)):                                                                    # Runs through the length of the width list.
        width_sum = width_sum + width[x]                                                            # Sum up the widths we have measured.

    fiber_width = width_sum / len(width)                                                            # Averages the column measurements.
    if show == 1:                                                                                   # If the show parameter is 1:
        print('FIBER WIDTH: ', fiber_width)                                                         # print the value of the fiber_width.
    scalebar = fiber_in_microns / fiber_width                                                       # Divides 125 by the pixel average to give a micron/pixel scale.
    return scalebar

# Function for determing the scanning region on the widest part of the device
# Column_scan should hold the x coordinate of the edge that we find as long as noise is clean enough.
# The list row that is returned will have the y coordinate to match each index of column_scan.
# For example, if at index 7, the column_scan might hold the value 850 and the row will hold 8, reflecting where the edge was located.
def ScanningRegion(frame,show):
    row = []                                                                                        # Initialize a list to store which row we are observing.
    global column_scan                                                                              # Include the column_scan global list in the scope of this function.

    # We scan from left to right across each row looking for the column that has the first black pixel (edge of device) then move downwards to other rows
    for j in range(TOP,BOTTOM):                                                                     # Iterate between the top and bottom of the screen.
        for i in range(LEFT_SIDE,RIGHT_SIDE):                                                       # Iterate from left to right across the majority of the screen, which should include the device.
            px = frame[j,i]                                                                         # Pixels are [row, column] so basically [y, x], yes it's inverted.
            if j != (RIGHT_SIDE - 1):                                                               # If we are not on the last column.
                pxplusone = frame[j,i+1]                                                            # Look ahead at the upcoming pixel in the scan, store it in pxplusone.
            if px == WHITE and pxplusone == BLACK:                                                  # If current pixel in scan is white and upcoming is black.
                column_scan.append(i+1)                                                             # Save the black pixel column location (index value plus one).
                row.append(j)                                                                       # Append the row that we found the edge.
                break
    return row



# Function that averages the grayscale values (between 0 and 255) for each pixel in the two scanning regions
# which are chosen through the input parameters. The midpoint of those two averages will be the value that will
# be used for thresholding.
def findThreshValue(filename,rectOneLeft_x,rectOneRight_x,rectOneTop_y,rectOneBot_y,show,rectTwoLeft_x,rectTwoRight_x,rectTwoTop_y,rectTwoBot_y):
    frame = cv2.imread(filename,1)                                                                  # Read the image with the filename offered in the parameter.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                                  # turn it gray
    pixels = []                                                                                     # Initialize a list to hold the values of the gray image pixels.
    for i in range(rectOneLeft_x,rectOneRight_x):                                                   # Cycle through the rows of allotted region.
        for j in range(rectOneTop_y, rectOneBot_y):                                                 # Cycle through the columns.
            pixels.append(gray[j,i])                                                                # Store the grayscale value at each pixel in a list.
    sumOfPixelValues= 0                                                                             # Variable to hold the sum of each pixel value.
    for x in range(len(pixels)):                                                                    # For the number of pixels measured.
        sumOfPixelValues = sumOfPixelValues + pixels[x]                                             # Sum the values.
    foregroundAverage = sumOfPixelValues / len(pixels)                                              # Average over the number of measurements.
    pixels.clear()                                                                                  # Clear the pixel list to measure the background rectangle.
    for i in range(rectTwoLeft_x,rectTwoRight_x):                                                   # Cycle through the rows of allotted region.
        for j in range(rectTwoTop_y,rectTwoBot_y):                                                  # Cycle through the columns.
            pixels.append(gray[j,i])                                                                # Store the grayscale value at each pixel in a list.
    sumOfPixelValues = 0                                                                            # Reset the sum variable.
    for x in range(len(pixels)):                                                                    # For the number of pixels measured.
        sumOfPixelValues = sumOfPixelValues + pixels[x]                                             # Sum the values.
    backgroundAverage = sumOfPixelValues / len(pixels)                                              # Average over the number of measurements.
    thresholdValue = (foregroundAverage + backgroundAverage) / MEAN                                 # Find the mean of the two average pixel values.
    if (show == SHOW):                                                                              # If the show variable is high.  
        cv2.rectangle(frame,(rectOneLeft_x,rectOneTop_y),(rectOneRight_x,rectOneBot_y),(0,0,255),1) # Draws a blue rectangle
        cv2.rectangle(frame,(rectTwoLeft_x,rectTwoTop_y),(rectTwoRight_x,rectTwoBot_y),(0,0,255),1) # To show the scanning regions.
        cv2.imshow('Original with Scan Regions', frame)                                             # Show the image with the scan regions.
        print("Threshold value for ", filename, " :", thresholdValue)                               # Output the threshold value calculated.
        cv2.waitKey(0)                                                                              # Wait for user key press.
    cv2.destroyAllWindows()                                                                         # Clear the windows.
    return thresholdValue                                                                           # Return the threshold value.                                                                      # Return the threshold value.



def filmDepth(fiberFilename, dryFilename, wetFilename, showFiber, showDry, showWet):
    # this block of code is for finding the widest points on the device according to our tolerance
    dryFrame = cv2.imread(dryFilename,1)                                                            # Set dryFrame to the image in the given file, where 1 allows for channels that show color.               
    gray = cv2.cvtColor(dryFrame, cv2.COLOR_BGR2GRAY)                                               # Turn it gray.
    dryThreshVal = findThreshValue(dryFilename,850,890,410,500,showDry,670,710,410,500)             # Find the thresholding value by calling findThreshDry function and setting it to dryThresh.
    ret, dryThresh = cv2.threshold(gray, dryThreshVal, 255, cv2.THRESH_BINARY)                      # Create an image dryThresh using the thresholding value just found.
    scan_range = ScanningRegion(dryThresh, showDry)                                                 # Calls a function to find the scanning range, changing the global list column_scan and the local list scan_range.
    if showDry == 1:                                                                                # If the parameter showDry is 1.
        for x in range(len(column_scan)):                                                           # For the length of the scanning list plot each coordinate with a blue pixel
            cv2.line(dryFrame, (column_scan[x],scan_range[x]), (column_scan[x],scan_range[x]), (178, 34, 34), 1)
        cv2.line(dryFrame, (LEFT_SIDE,TOP_THIRD_VERT), (LEFT_SIDE,BOTTOM_THIRD_VERT), (178, 34, 34), 1)                    # Draw a line to show the start of the scanning region.
        cv2.line(dryFrame, (RIGHT_SIDE,TOP_THIRD_VERT), (RIGHT_SIDE,BOTTOM_THIRD_VERT), (178, 34, 34), 1)  # Draw a line to show the end of the scanning region.
        cv2.imshow('Dry Device Edited', dryFrame)                                                   # Print a color frame to show the pixels being measured.
        cv2.imshow('Dry Device Binary', dryThresh)                                                  # Print the binary thresholded image for comparison.
        cv2.waitKey(0)                                                                              # Wait for a key press from the user.
    # this block of code is for finding the width of the fiber
    fiberFrame = cv2.imread(fiberFilename,1)                                                        # Set fiberFrame to the image in the given file, where 1 allows for channels that show color.               
    gray = cv2.cvtColor(fiberFrame, cv2.COLOR_BGR2GRAY)                                             # Turn it gray.
    fiberThreshVal = findThreshValue(fiberFilename,100,1000,500,840,showFiber,100,1000,870,1000)
    ret, thresh = cv2.threshold(gray, fiberThreshVal, 255, cv2.THRESH_BINARY)                       # Binary threshold the image with the value found above.
    if showFiber == 1:                                                                              # If the parameter showFiber is 1.
        cv2.imshow('Fiber Thresholded', thresh)                                                     # Show the thresholded fiber.
        cv2.imshow('Fiber Original', fiberFrame)                                                    # Show the original image of the fiber.
    scalebar = FiberWidthTest(thresh, showFiber)                                                    # Call the FiberWidthTest function to find the pixel/micron scale.
    print('SCALEBAR: ', scalebar)                                                                   # Print out the scale found above.
    cv2.waitKey(0)                                                                                  # Wait for a key press from the user.
    # this block of code is for finding the wet device and the film depth compared to our scanning region
    wetFrame = cv2.imread(wetFilename,1)                                                            # Set wetFrame to the image in the given file, where 1 allows for channels that show color.    
    wetThreshVal = findThreshValue(wetFilename,850,890,410,500,showWet,670,710,410,500)             # Calls the findThreshWet function to find the thresholding value.
    gray = cv2.cvtColor(wetFrame, cv2.COLOR_BGR2GRAY)                                               # turn it gray
    ret, wetThresh = cv2.threshold(gray, wetThreshVal, 255, cv2.THRESH_BINARY)                      # binary thresholding 127 and 255 are the standard values used
    if showWet == 1:                                                                                # If showWet is enabled.
        cv2.line(wetFrame, (LEFT_SIDE,TOP_THIRD_VERT), (LEFT_SIDE,BOTTOM_THIRD_VERT), (178, 34, 34), 1)                   # Draw a line to show the scanning region.
        cv2.line(wetFrame, (RIGHT_SIDE,TOP_THIRD_VERT), (RIGHT_SIDE,BOTTOM_THIRD_VERT), (178, 34, 34), 1) # Draw a line to show the scanning region.
        cv2.imshow('Wet Device Original', wetFrame)                                                 # Show the original device.
        cv2.imshow('Wet Device Thresholded', wetThresh)                                             # Show the thresholded device.
        cv2.waitKey(0)                                                                              # Wait for a key press from the user.

    microns = 0                                                                                     # depth of film in microns
    measurements = []                                                                               # list holds the measurements of the scan
    global film                                                                                     # Include the global list film.
    measure_sum = 0                                                                                 # sum of the said list
    measure_avg = 0                                                                                 # avg value of depth in pixels
    counter = 0                                                                                     # Initialize a counter.

    for i in range(len(scan_range)):                                                                # Scan each row on the device.
        for j in range(LEFT_SIDE,RIGHT_SIDE):                                                       # Scan the area about LEFT_SIDEway to the right quarter of the screen.
            px = wetThresh[scan_range[i], j]                                                        # Pixel currently scanning.
            pxplusone = wetThresh[scan_range[i], j + 1]                                             # Look one pixel ahead of scanning location.
            if px == WHITE and pxplusone == BLACK:                                                  # If the scanning location is white and the next pixel is black.
                measurements.append(column_scan[i] - (j + 1))                                       # Save the difference of the dry column and wet column (thickness of film in pixels).
                break
        counter = counter + 1                                                                       # Increment counter.
        if counter == PIXELS_IN_INCREMENT:                                                          # If counter is equal to the number desired:
            counter = 0                                                                             # reset counter.
            for x in range(len(measurements)):                                                      # For all of the measurements.
                measure_sum = measure_sum + measurements[x]                                         # Sum each in the list.
            if len(measurements) != 0:                                                              # If the measurement list actually has values:
                measure_avg = measure_sum / len(measurements)                                       # take the average by dividing the sum by the number of measurements.
                microns = measure_avg * scalebar                                                    # Convert the pixel depth of film to microns using the scalebar found previously.
                film.append(microns)                                                                # Every 25 pixel sections are averaged, converted to microns, and stored in the list film.
                measurements.clear()                                                                # Clear the measurements list for the next batch of 25 pixels.
                measure_sum = 0                                                                     # Reset the measure_sum variable for the next batch.                 
    cv2.destroyAllWindows()                                                                         # Before leaving the filmDepth function, clear all windows.
    return

filmDepth("Fiber5.jpg","Dry5.jpg","Wet5.jpg",1,1,1) # Call the function with three files and three ints that tell the function whether to print images and debugging

showFrame = cv2.imread("Wet5.jpg",1)                                                                # Set showFrame to the image in the given file, where 1 allows for channels that show color.
y = TOP                                                                                             # Set the variable y to the value of the top of the screen.
spacer = 24                                                                                         # Have a spacer of 24 for printing the measurements.
font = cv2.FONT_HERSHEY_SIMPLEX                                                                     # Normal size sans-serif font.
print("The number of measurements: ",len(film))
print("Measurements: ",film)
stringList = []                                                                                     # Initialize a list to hold the measurements converted to strings.
for x in range(len(film)):                                                                          # For the length of the measurements taken
    roundedFloat = round(film[x],2)
    stringList.append(str(roundedFloat))                                                            # Use the function str() to convert the floating point numbers to strings.
    cv2.putText(showFrame,stringList[x],(PRINT_RIGHT_SIDE,y+spacer), font , 1,(255,255,255),1,cv2.LINE_AA) 
    # Puts text to the screen of a file, with string, at a certain origin, with a font, a text size, color, and line thickness
    cv2.line(showFrame, (760,y), (1150,y), (178, 34, 34), 1)  # Draw lines incrementing every 25 pixels to show which regions we are measuring.
    y = y + PIXELS_IN_INCREMENT                                                                     # Increment y by 25.
cv2.imshow('Wet Device With 25 Px Ticks', showFrame)    # Show the edited frame.

data = np.array(film)                                                                               # Convert the python list to a numpy array.
plt.plot(data, 'ro')                                                                                # Plot the data with red dots.
plt.ylabel("Film Depth in Microns")                                                                 # Change the y axis lable.
plt.xlabel("Measurements in 25 pixel Increments")                                                   # Change the x axis label.
# plt.show()                                                                                        # Show the plot. 'q' will exit from it.

cv2.waitKey(0)                                                                                      # Wait for any key press.
cv2.destroyAllWindows()                                                                             # Destroy all windows.