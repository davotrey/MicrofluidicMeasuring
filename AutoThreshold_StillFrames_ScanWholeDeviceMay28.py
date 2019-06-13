import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

#global variables
BLACK = 0           # value for black.
WHITE = 255         # the value for white.
LEFT = 1            # leftmost column on the screen.
RIGHT = 1279        # rightmost column on the screen.
TOP = 1             # top row on the screen.
BOTTOM = 1023       # bottom row on the screen. 
HALF = 400    # the column value for the half of the screen.
RIGHT_QUARTER = 1000   # the column value for the right quarter of the screen.
TOP_THIRD_VERT = 300
BOTTOM_THIRD_VERT = 600
PIXELS_IN_INCREMENT = 25
PRINT_RIGHT_SIDE = 850
column_scan = []    # list used as a global variable to store the column location during scans.
film = []     
column = 1280

# Function for finding the width of the fiber in pixels, then finding the width of each pixel in microns
# based on the known width of the fiber as 125 microns.
def FiberWidthTest(frame, show):
    start = []              # initializing empty lists
    end = []
    width = []
    fiber_in_microns = 125  # The actual width of the fiber is known to be 125 microns.
    fiber_width = 0         # Number of pixels.
    width_sum = 0           # Sum of the width measurements.

    for j in range (LEFT,RIGHT):                    # Scan all columns from left to right.
        for i in range(TOP,BOTTOM):                 # Scan all rows fromt top to bottom.
            px = frame[i,j]                         # Save the pixel value, where [y,x] since it is inverted.
            if i != BOTTOM:                         # If the for loop is not on the last pixel, save the upcoming pixel.
                pxplusone = frame[i+1, j]           # Save the pixel to the right of our current position in the scan
            if px == WHITE:                         # If we are on a white pixel.
                if pxplusone == BLACK:              # And the next one is black.
                    start.append(i + 1)             # Add the pixel location to the 'start' list.
            if px == BLACK:                         # If the for loop is on a black pixel,
                if pxplusone == WHITE:              # and the next is white,
                    end.append(i)                   # add the value of the white pixel to 'end' list.
        width.append(end[len(end)-1]-start[0])      # Finds the width by taking the last value from the end list and the first value of the start list, then appends the difference.
        if show == 2:                               # If the show input var is 2:
            print ('Fiber Width: ', width)          # print to see the different values measured.
        start.clear()                               # Clears list to measure the next column.
        end.clear()                                 # Clear list.

    for x in range (len(width)):                # Runs through the length of the width list.
        width_sum = width_sum + width[x]        # Sum up the widths we have measured.

    fiber_width = width_sum / len(width)        # Averages the column measurements.
    if show == 1:                               # If the show parameter is 1:
        print('FIBER WIDTH: ', fiber_width)     # print the value of the fiber_width.
    scalebar = fiber_in_microns / fiber_width   # Divides 125 by the pixel average to give a micron/pixel scale.
    return scalebar

# Function for determing the scanning region on the widest part of the device
# Column_scan should hold the x coordinate of the edge that we find as long as noise is clean enough.
# The list row that is returned will have the y coordinate to match each index of column_scan.
# For example, if at index 7, the column_scan might hold the value 850 and the row will hold 8, reflecting where the edge was located.
def ScanningRegion(frame,show):
    row = []            # Initialize a list to store which row we are observing.
    global column_scan  # Include the column_scan global list in the scope of this function.

    # we scan from left to right across each row, then move downwards looking for the column that has the first black pixel
    for i in range(TOP,BOTTOM):                     # Iterate between the top and bottom of the screen.
        for j in range(HALF,RIGHT_QUARTER):         # Iterate between halfway on the screen to the right quarter of the screen, which should include the device.
            px = frame[i,j]                         # Pixels are [row, column] so basically [y, x], yes it's inverted.
            if j != (RIGHT_QUARTER - 1):            # If we are not on the last column.
                pxplusone = frame[i,j+1]            # Look ahead at the upcoming pixel in the scan, store it in pxplusone.
            if px == WHITE and pxplusone == BLACK:  # If current pixel in scan is white and upcoming is black.
                column_scan.append(j+1)             # Save the black pixel column location (index value plus one).
                row.append(i)                       # Append the row that we found the edge.
                break
    return row

# Function for finding the threshold value for the dry image.
def findThreshDry(filename, showDry):
    frame = cv2.imread(filename,1)                          # Read the image with the filename offered in the parameter.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # Turn it gray.
    # Rectangles for the oil wetted device from May 23 experiment.
    cv2.rectangle(frame,(600,410),(640,500),(0,255,0),1)    # Draw a rectangle.
    cv2.rectangle(frame,(500,410),(540,500),(0,255,0),1)    # Draw a rectangle.
    # Rectangles for the water wetted device from May 23 experiment.
    # cv2.rectangle(frame,(750,410),(790,500),(0,255,0),1)    # Draw a rectangle.
    # cv2.rectangle(frame,(670,410),(710,500),(0,255,0),1)    # Draw a rectangle.
    dryDevicePixels = []                                    # initialize region
    for i in range(600,640):                                   # cycle through the rows
        for j in range(410, 500):                           # cycle through the columns
            dryDevicePixels.append(gray[j,i])               # store the grayscale value at each pixel in a list
    dryEdgeSum = 0
    for x in range(len(dryDevicePixels)):                   # for each index in said list
        dryEdgeSum = dryEdgeSum + dryDevicePixels[x]        # add up the value
    dryEdgeAvg = dryEdgeSum / len(dryDevicePixels)          # then avg

    dryBackgroundPixels = []
    for i in range(500,540):
        for j in range(410,500):
            dryBackgroundPixels.append(gray[j,i])
    dryBackgroundSum = 0
    for x in range(len(dryBackgroundPixels)):
        dryBackgroundSum = dryBackgroundSum + dryBackgroundPixels[x]
    dryBackgroundAverage = dryBackgroundSum / len(dryBackgroundPixels)
    
    dryThresh = (dryEdgeAvg + dryBackgroundAverage) / 2     # midpoint between the two values
    if (showDry == 1):
        cv2.imshow('Dry Device Original', frame)            # show the edited frame
        print("Dry threshold Value for ", filename, " :", dryThresh)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dryThresh

def findThreshWet(filename, showWet):
    frame = cv2.imread(filename,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # turn it gray
    # Rectangles for the oil wetted device from May 23 experiment.
    cv2.rectangle(frame,(600,410),(640,500),(0,255,0),1)    # Draw a rectangle.
    cv2.rectangle(frame,(500,410),(540,500),(0,255,0),1)    # Draw a rectangle.
    # Rectangles for the water wetted device from May 23 experiment.
    # cv2.rectangle(frame,(750,410),(790,500),(0,255,0),1)    # Draw a rectangle.
    # cv2.rectangle(frame,(670,410),(710,500),(0,255,0),1)    # Draw a rectangle.

    wetDevicePixels = []                                    # initialize region
    for i in range(600,640):                                 # cycle through the rows
        for j in range(410, 500):                           # cycle through the columns
            wetDevicePixels.append(gray[j,i])               # store the grayscale value at each pixel in a list
    wetEdgeSum = 0
    for x in range(len(wetDevicePixels)):                   # for each index in said list
        wetEdgeSum = wetEdgeSum + wetDevicePixels[x]        # add up the value
    wetEdgeAvg = wetEdgeSum / len(wetDevicePixels)          # then avg
    
    wetBackgroundPixels = []
    for i in range(500,540):
        for j in range(410,500):
            wetBackgroundPixels.append(gray[j,i])
    wetBackgroundSum = 0
    for x in range(len(wetBackgroundPixels)):
        wetBackgroundSum = wetBackgroundSum + wetBackgroundPixels[x]
    wetBackgroundAverage = wetBackgroundSum / len(wetBackgroundPixels)
    wetThresh = (wetEdgeAvg + wetBackgroundAverage) / 2     # midpoint between the two values
    if (showWet == 1):
        cv2.imshow('Wet Device Original', frame)                # show the edited frame
        print("Wet threshold Value for ", filename, " :", wetThresh)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return wetThresh


def filmDepth(fiberFilename, dryFilename, wetFilename, showFiber, showDry, showWet):
    # this block of code is for finding the widest points on the device according to our tolerance
    dryFrame = cv2.imread(dryFilename,1)                # Set dryFrame to the image in the given file, where 1 allows for channels that show color.               
    gray = cv2.cvtColor(dryFrame, cv2.COLOR_BGR2GRAY)   # Turn it gray.
    dryThreshVal = findThreshDry(dryFilename,showDry)   # Find the thresholding value by calling findThreshDry function and setting it to dryThresh.
    ret, dryThresh = cv2.threshold(gray, dryThreshVal, 255, cv2.THRESH_BINARY)  # Create an image dryThresh using the thresholding value just found.
    scan_range = ScanningRegion(dryThresh, showDry)     # Calls a function to find the scanning range, changing the global list column_scan and the local list scan_range.
    if showDry == 1:                                    # If the parameter showDry is 1.
        for x in range(len(column_scan)):               # For the length of the scanning list plot each coordinate with a blue pixel
            cv2.line(dryFrame, (column_scan[x],scan_range[x]), (column_scan[x],scan_range[x]), (178, 34, 34), 1)
        cv2.line(dryFrame, (HALF,TOP_THIRD_VERT), (HALF,BOTTOM_THIRD_VERT), (178, 34, 34), 1)                    # Draw a line to show the start of the scanning region.
        cv2.line(dryFrame, (RIGHT_QUARTER,TOP_THIRD_VERT), (RIGHT_QUARTER,BOTTOM_THIRD_VERT), (178, 34, 34), 1)  # Draw a line to show the end of the scanning region.
        cv2.imshow('Dry Device Edited', dryFrame)       # Print a color frame to show the pixels being measured.
        cv2.imshow('Dry Device Binary', dryThresh)      # Print the binary thresholded image for comparison.
        cv2.waitKey(0)                                  # Wait for a key press from the user.
    # this block of code is for finding the width of the fiber
    fiberFrame = cv2.imread(fiberFilename,1)            # Set fiberFrame to the image in the given file, where 1 allows for channels that show color.               
    gray = cv2.cvtColor(fiberFrame, cv2.COLOR_BGR2GRAY) # Turn it gray.
    ret, thresh = cv2.threshold(gray, dryThreshVal, 255, cv2.THRESH_BINARY) # Binary threshold the image with the value found above.
    if showFiber == 1:                                  # If the parameter showFiber is 1.
        cv2.imshow('Fiber Thresholded', thresh)         # Show the thresholded fiber.
        cv2.imshow('Fiber Original', fiberFrame)        # Show the original image of the fiber.
    scalebar = FiberWidthTest(thresh, showFiber)        # Call the FiberWidthTest function to find the pixel/micron scale.
    print('SCALEBAR: ', scalebar)                       # Print out the scale found above.
    cv2.waitKey(0)                                      # Wait for a key press from the user.
    # this block of code is for finding the wet device and the film depth compared to our scanning region
    wetFrame = cv2.imread(wetFilename,1)                # Set wetFrame to the image in the given file, where 1 allows for channels that show color.    
    wetThreshVal = findThreshWet(wetFilename,showWet)   # Calls the findThreshWet function to find the thresholding value.
    gray = cv2.cvtColor(wetFrame, cv2.COLOR_BGR2GRAY)                              # turn it gray
    ret, wetThresh = cv2.threshold(gray, wetThreshVal, 255, cv2.THRESH_BINARY)  # binary thresholding 127 and 255 are the standard values used
    if showWet == 1:                                    # If showWet is enabled.
        cv2.line(wetFrame, (HALF,TOP_THIRD_VERT), (HALF,BOTTOM_THIRD_VERT), (178, 34, 34), 1)                   # Draw a line to show the scanning region.
        cv2.line(wetFrame, (RIGHT_QUARTER,TOP_THIRD_VERT), (RIGHT_QUARTER,BOTTOM_THIRD_VERT), (178, 34, 34), 1) # Draw a line to show the scanning region.
        cv2.imshow('Wet Device Original', wetFrame)     # Show the original device.
        cv2.imshow('Wet Device Thresholded', wetThresh) # Show the thresholded device.
        cv2.waitKey(0)                                  # Wait for a key press from the user.

    microns = 0             # depth of film in microns
    measurements = []       # list holds the measurements of the scan
    global film             # Include the global list film.
    measure_sum = 0         # sum of the said list
    measure_avg = 0         # avg value of depth in pixels
    counter = 0             # Initialize a counter.

    for i in range(len(scan_range)):                            # Scan each row on the device.
        for j in range(HALF,RIGHT_QUARTER):                     # Scan the area about halfway to the right quarter of the screen.
            px = wetThresh[scan_range[i], j]                    # Pixel currently scanning.
            pxplusone = wetThresh[scan_range[i], j + 1]         # Look one pixel ahead of scanning location.
            if px == WHITE and pxplusone == BLACK:              # If the scanning location is white and the next pixel is black.
                measurements.append(column_scan[i] - (j + 1))   # Save the difference of the dry column and wet column (thickness of film in pixels).
                break
        counter = counter + 1                                   # Increment counter.
        if counter == PIXELS_IN_INCREMENT:                      # If counter is equal to the number desired:
            counter = 0                                                 # reset counter.
            for x in range(len(measurements)):                          # For all of the measurements.
                measure_sum = measure_sum + measurements[x]             # Sum each in the list.
            if len(measurements) != 0:                                  # If the measurement list actually has values:
                measure_avg = measure_sum / len(measurements)           # take the average by dividing the sum by the number of measurements.
                microns = measure_avg * scalebar                        # Convert the pixel depth of film to microns using the scalebar found previously.
                film.append(microns)                                    # Every 25 pixel sections are averaged, converted to microns, and stored in the list film.
                measurements.clear()                                    # Clear the measurements list for the next batch of 25 pixels.
                measure_sum = 0                                         # Reset the measure_sum variable for the next batch.                 
    cv2.destroyAllWindows()                                             # Before leaving the filmDepth function, clear all windows.
    return

filmDepth("F_6.5_OilFiber.jpg","F_6.5_OilDry.jpg","F_6.5_Stability1Hour_2460.jpg",1,1,1) # Call the function with three files and three ints that tell the function whether to print images and debugging

showFrame = cv2.imread("F_6.5_Stability1Hour_2460.jpg",1)   # Set showFrame to the image in the given file, where 1 allows for channels that show color.
y = TOP                                     # Set the variable y to the value of the top of the screen.
spacer = 24                                 # Have a spacer of 24 for printing the measurements.
font = cv2.FONT_HERSHEY_SIMPLEX             # Normal size sans-serif font.
print("The number of measurements: ",len(film))
print("Measurements: ",film)
stringList = []                             # Initialize a list to hold the measurements converted to strings.
for x in range(len(film)):                  # For the length of the measurements taken
    stringList.append(str(film[x]))         # Use the function str() to convert the floating point numbers to strings.
    cv2.putText(showFrame,stringList[x],(PRINT_RIGHT_SIDE,y+spacer), font , 1,(255,255,255),1,cv2.LINE_AA) 
    # Puts text to the screen of a file, with string, at a certain origin, with a font, a text size, color, and line thickness
    cv2.line(showFrame, (HALF,y), (RIGHT_QUARTER,y), (178, 34, 34), 1)  # Draw lines incrementing every 25 pixels to show which regions we are measuring.
    y = y + PIXELS_IN_INCREMENT             # Increment y by 25.
cv2.imshow('Wet Device With 25 Px Ticks', showFrame)    # Show the edited frame.

data = np.array(film)   # Convert the python list to a numpy array.
plt.plot(data, 'ro')    # Plot the data with red dots.
plt.ylabel("Film Depth in Microns") # Change the y axis lable.
plt.xlabel("Measurements in 25 pixel Increments")   # Change the x axis label.
plt.show()              # Show the plot. 'q' will exit from it.

cv2.waitKey(0)          # Wait for any key press.
cv2.destroyAllWindows() # Destroy all windows.