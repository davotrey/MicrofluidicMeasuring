# Finds film depth at one pixel at three different points, plotting said points.
# Using vertical and horizontal shift data the dry edge will follow the wet edge throughout PEGDA resin expansion.


import numpy as np
import cv2
import pylab as plt
import matplotlib.ticker as ticker

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
MEAN = 2                                                                                            # For finding the mean between two measurements.
IMAGE = 0                                                                                           # Value for when an image is the file input for the function.
VIDEO = 1                                                                                           # Value for a video.
NO_FRAME = 0                                                                                        # When using an image and no frame is specified.
FULL_HUE = 255                                                                                  
NO_HUE = 0
column_scan = []                                                                                    # List used as a global variable to store the column location of dry device during scans.
film1 = []                                                                                          # Lists to hold the film thickness of one frame averaged.
film2 = []   
film3 = []   
mseRawDataList = []                                                                                 # Lists to hold data relevant to the tracking
mseRelativeList = []
yshiftList = []
xshiftList = []

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
def findThreshValue(imageOrVideo,frameFromVideo,filename,rectOneLeft_x,rectOneRight_x,rectOneTop_y,rectOneBot_y,show,rectTwoLeft_x,rectTwoRight_x,rectTwoTop_y,rectTwoBot_y):
    if (imageOrVideo == IMAGE):                                                                     # If the frame is a separate .jpg or .png file
        frame = cv2.imread(filename,1)                                                              # Read the image with the filename offered in the parameter.
    elif (imageOrVideo == VIDEO):                                                                   # Else if the frame is passed in from a video
        frame = frameFromVideo                                                                      # Save the frame to the same local variable that would have been used with the .jpg
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
    return thresholdValue                                                                           # Return the threshold value.


def filmStablity(fiberFilename,dryFilename,videoFilename,showFiber,showDry,showWet):
    y1 = 500 - 175                                                                                  # This will be the row where we take our first measurement.
    y2 = 500 - 333                                                                                  # Currently the code supports three total measurements.
    y3 = 500 + 350  
    IMAGE = 0                                                                                       # A flag that tells other functions if a file is an image or a video.
    VIDEO = 1
    # This block of code is for finding the width of the fiber.
    fiberFrame = cv2.imread(fiberFilename,1)                                                        # Set fiberFrame to the image in the given file, where 1 allows for channels that show color.               
    gray = cv2.cvtColor(fiberFrame, cv2.COLOR_BGR2GRAY)                                             # Turn it gray.
    fiberThreshVal = findThreshValue(IMAGE,0,fiberFilename,100,1000,200,840,showFiber,100,1000,900,1000)
    ret, thresh = cv2.threshold(gray, fiberThreshVal, 255, cv2.THRESH_BINARY)                       # Binary threshold the image with the value found above.
    if showFiber == SHOW:                                                                           # If the parameter showFiber is 1.
        cv2.imshow('Fiber Thresholded', thresh)                                                     # Show the thresholded fiber.
        cv2.imshow('Fiber Original', fiberFrame)                                                    # Show the original image of the fiber.
    scalebar = FiberWidthTest(thresh, showFiber)                                                    # Call the FiberWidthTest function to find the pixel/micron scale.
    print('SCALEBAR: ', scalebar)                                                                   # Print out the scale found above.
    cv2.waitKey(0)                                                                                  # Wait for a key press from the user.
    # This block of code is for finding the edge of the dry device.
    dryFrameJumpback = 510    
    # dryFrameJumpback = 300                                                                        # The dry device frame is taken this many frames from the end of the video after the water is evaporated.
    camera = cv2.VideoCapture(videoFilename)                                                        # Connect to the .avi file provided in the parameter videoFilename.
    endOfVideo = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))                                          # Finds the number of frames in the video.
    print('NUMBER OF FRAMES: ', endOfVideo)                                                         # Print the number of frames.
    camera.set(cv2.CAP_PROP_POS_FRAMES, endOfVideo-dryFrameJumpback)                                # Set the frame to the end of the video if using water (it should have evaporated)
    ret, dryFrame = camera.read()                                                                   # Set the dryFrame by reading the camera at the designated frame.
    dryThreshVal = findThreshValue(VIDEO,dryFrame,dryFilename,750,800,410,500,HIDE,500,550,410,500)   
        # Find the thresholding value, tell whether it is an IMAGE or VIDEO, send the frame if its a video, send the filename if its an image, then put the coordinates for the two boxes.
    grayed = cv2.cvtColor(dryFrame, cv2.COLOR_BGR2GRAY)                                             # Turn it gray.
    ret, dryThresh = cv2.threshold(grayed, dryThreshVal, FULL_HUE, cv2.THRESH_BINARY)               # Binary threshold the grayscale image with the value we found.
    scan_range = ScanningRegion(dryThresh, showDry)                                                 # Calls a function to find the scanning range, changing the global list column_scan and the local list scan_range.
    offset = BOTTOM - len(scan_range)                                                               # The offset is important if the top of the device is visible and no edge could be found at the top of the screen.
    if showDry == SHOW:                                                                             # If the parameter showDry is 1.
        camera.set(cv2.CAP_PROP_POS_FRAMES, endOfVideo-dryFrameJumpback)                            # Set the frame to the end of the video if using water (it should have evaporated)
        ret, dryFrame = camera.read()                                                               # Set the dryFrame by reading the camera at the designated frame.
        for x in range(len(column_scan)):                                                           # For the length of the scanning list plot each coordinate with a blue pixel
            cv2.line(dryFrame, (column_scan[x],scan_range[x]), (column_scan[x],scan_range[x]), (178, 34, 34), 1)
        cv2.line(dryFrame, (LEFT_SIDE,y1-64), (RIGHT_SIDE,y1-64), (NO_HUE, NO_HUE, FULL_HUE), 6)          # Draw a red line to show the start of the scanning region.
        cv2.line(dryFrame, (LEFT_SIDE,y2-64), (RIGHT_SIDE,y2-64), (FULL_HUE, NO_HUE, NO_HUE), 6)          # Draw a blue line to show the end of the scanning region.
        cv2.line(dryFrame, (LEFT_SIDE,y3-64), (RIGHT_SIDE,y3-64), (NO_HUE, 130, NO_HUE), 6)               # Draw a green line to show the end of the scanning region.
        cv2.imshow('Dry Device Edited', dryFrame)                                                   # Print a color frame to show the pixels being measured.
        cv2.imshow('Dry Device Binary', dryThresh)                                                  # Print the binary thresholded image for comparison.
        cv2.waitKey(0)                                                                              # Wait for a key press from the user.


    yShiftData = np.loadtxt('yshift500to12450.txt')                                                 # Load data from txt file.
    yShiftData.tolist()                                                                             # Convert numpy array to python list.
    xShiftData = np.loadtxt('xshift500to12450.txt')
    xShiftData.tolist()

    maxVert = yShiftData[0]                                                                         # The vertical coordinate at frame 0.
    minVert = yShiftData[len(yShiftData)-1]                                                         # The vertical coordinate at the end of the video.
    totalVertShift = maxVert - minVert                                                              # The total shift is the difference between vertical coordinates.
    print("minVert: ",minVert," maxVert: ",maxVert," totalVertShift: ",totalVertShift)    

    # This block of code is for scanning the frames of the video and comparing with the dry device.
    actualStartingFrame = 500
    count = actualStartingFrame                                                                     # Track the frame of the video that is being analyzed.  
    scanTolerance = 30         
    scanToleranceLarge = 650                                                                        # How far left and right of a coordinate we will scan.
    scanHeight = 9
    camera.set(cv2.CAP_PROP_POS_FRAMES, actualStartingFrame)                                        # Set the frame to the starting point
    while True:                                                                                     # Keep looping here until 'q' is pressed or the video is complete.
        measurements1 = []                                                                          # List holds the measurements of the scan for measurement 1.
        measurements2 = []                                                                          # For measurement 2.
        measurements3 = []                                                                          # For measurement 3.
        measure_sum1 = 0                                                                            # Variable to hold the sum of the list of measurements for measurement 1.
        measure_sum2 = 0                                                                            # For measurement 2, resets after every frame.
        measure_sum3 = 0                                                                            # For measurement 3.
        measure_avg1 = 0                                                                            # Variable to hold the average value of depth in pixels for measurement 1.
        measure_avg2 = 0                                                                            # For measurement 2, resets after every frame.
        measure_avg3 = 0                                                                            # For measurement 3.
        row_scan_shifted = []                                                                       # Initialize lists to hold the shifted dry edge of the device.
        column_scan_shifted = []
        measurements1.clear()                                                                       # Reset the first list.
        measurements2.clear()                                                                       # Reset the second list.   
        measurements3.clear() 
        row_scan_shifted.clear()                                                    
        column_scan_shifted.clear()
        count = count + 1                                                                           # Frame counter.
        ret, frame = camera.read()                                                                  # Read a frame.
        y1 = 500 - 175                                                                                  # This will be the row where we take our first measurement.
        y2 = 500 - 333                                                                                  # Currently the code supports three total measurements.
        y3 = 500 + 350

        if ((count == actualStartingFrame + 1)):                                                    # Every 500 pixels and at the beginning of the video, set the threshold value again.
            wetThreshVal = findThreshValue(VIDEO,frame,videoFilename,750,800,410,500,HIDE,500,550,410,500)   
            if (showWet == SHOW):                                                                       
                print("Wet Threshold Value: ", wetThreshVal)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                              # Turn it gray.
        ret, thresh = cv2.threshold(gray, wetThreshVal, FULL_HUE, cv2.THRESH_BINARY)                # Binary thresholding.
        cv2.imshow('Binary Threshold', thresh)                                                      # Draw the result - this is the video playing.

        vertShift = int(500 - yShiftData[count-actualStartingFrame-1])                              # Every frame has the vertical shift accounted for given the data found through tracking.
        localMaxVert = int(yShiftData[count-actualStartingFrame-1])                                 # Each frame will have a different vertical coordinate.
        vertDifference = int(localMaxVert - minVert)                                                # The difference between the vertical coordinate of each frame and the final frame.
        xshift = int(6 - xShiftData[count - actualStartingFrame - 1])                               # Horizontal shift for each frame.
        # Scan is just one pixel.
        for x in range(len(column_scan)):                                                           # For the length of the scanning list plot each coordinate with a blue pixel
            column_scan_shifted.append(column_scan[x]-xshift)                                       # Shift the columns by the horizontal shift loaded.
            row_scan_shifted.append(scan_range[x]+vertDifference)                                   # Shift the rows by the vertical shift loaded.
            if x > 50:
                cv2.line(frame, (column_scan_shifted[x],row_scan_shifted[x]), (column_scan_shifted[x],row_scan_shifted[x]), (0, 255, 0), 1)
        for vert in range(scanHeight):
            y1 = y1 + vert
            y2 = y2 + vert
            y3 = y3 + vert
            xcoordinate1 = column_scan_shifted[y1-vertShift-vertDifference]                             # x coordinate or the column of the edge of the dry device.
            xcoordinate2 = column_scan_shifted[y2-vertShift-vertDifference]
            xcoordinate3 = column_scan_shifted[y3-vertShift-vertDifference]
            for x in range(xcoordinate1 - scanTolerance, xcoordinate1 + scanTolerance):                 # Scan in front of the device and a little behind, in case of noise.
                px1 = thresh[y1-vertShift, x]                                                           # Scanning location, where the coordinate is (y,x), it is reversed which is confusing.
                pxplusone1 = thresh[y1-vertShift, x + 1]                                                # Look one pixel ahead of scanning location.
                if px1 == WHITE and pxplusone1 == BLACK:                                                # If the scanning location is black and the next pixel is white
                    measurements1.append((xcoordinate1 - (x + 1))*scalebar)                                     # Save the difference of the dry column and wet column (thickness of film in pixels).
                    break
            for x in range(xcoordinate2 - scanTolerance, xcoordinate2 + scanTolerance):                 # Just like measurement1.      
                px2 = thresh[y2-vertShift, x]                       
                pxplusone2 = thresh[y2-vertShift, x + 1]            
                if px2 == WHITE and pxplusone2 == BLACK:                    
                    measurements2.append((xcoordinate2 - (x + 1))*scalebar)
                    break
            for x in range(xcoordinate3 - scanToleranceLarge, xcoordinate3 + scanTolerance):            # Just like measurement 1, except the scanning region is a little larger for a measurement at the neck.
                px3 = thresh[y3-vertShift, x]                       
                pxplusone3 = thresh[y3-vertShift, x + 1]            
                if px3 == WHITE and pxplusone3 == BLACK:                    
                    measurements3.append((xcoordinate3 - (x + 1))*scalebar)   
                    cv2.rectangle(frame,(x+1,y3-vertShift-5),(x+1,y3-vertShift),(FULL_HUE,NO_HUE,NO_HUE),1)         # Draw a rectangle to show the vertical scanning region.
                    cv2.rectangle(frame,(xcoordinate3,y3-vertShift-5),(xcoordinate3,y3-vertShift),(0,0,255),1)      # Draw a rectangle to show the vertical scanning region.
                    break
        for x in range(len(measurements1)):                                                         # For all of the measurements in meas1.
            measure_sum1 = measure_sum1 + measurements1[x]                                          # Sum each measurement in the list.
        for x in range(len(measurements2)):                                                         
            measure_sum2 = measure_sum2 + measurements2[x]                                          
        for x in range(len(measurements3)):
            measure_sum3 = measure_sum3 + measurements3[x]                                          
        if len(measurements1) != 0:                                                                 # If the measurement list actually has values
            measure_avg1 = measure_sum1 / len(measurements1)                                        # Take the average by dividing the sum by the number of measurements.
            film1.append(measure_avg1)                                                                  # Add each frame's measurement to a list called film.
        if len(measurements2) != 0:                                                                 
            measure_avg2 = measure_sum2 / len(measurements2)                                        
            film2.append(measure_avg2)                                                                  
        if len(measurements3) != 0:                                                                 
            measure_avg3 = measure_sum3 / len(measurements3)                                        
            film3.append(measure_avg3)
        cv2.rectangle(frame,(LEFT_SIDE,y1-vertShift-1),(RIGHT_SIDE,y1-vertShift+1),(NO_HUE,NO_HUE,FULL_HUE),1)  # Draw red rectangle.
        cv2.rectangle(frame,(LEFT_SIDE,y2-vertShift-1),(RIGHT_SIDE,y2-vertShift+1),(FULL_HUE,NO_HUE,NO_HUE),1)  # Draw blue rectangle.
        cv2.rectangle(frame,(LEFT_SIDE,y3-vertShift-1),(RIGHT_SIDE,y3-vertShift+1),(NO_HUE,FULL_HUE,NO_HUE),1)  # Draw green rectangle.
        cv2.imshow('Original', frame)                                                               # Draw the result.

        if count == 12450:                                                                          # Once we reach the end of the portion of video we want to analyze
            cv2.destroyAllWindows()                                                                 # Clear all windows.
            print('END OF THE VIDEO')   
            print('Count',count)                                                            
            break                                                                                   # Break the while loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):                                                       # Press 'q' to exit the while loop early
            cv2.destroyAllWindows()
            print('END OF THE VIDEO')
            print('Count',count)
            break

print(film1)
filmStablity("FiberFrameJuly20.jpg","Dry.jpg","WetVidJuly20.avi",HIDE,HIDE,SHOW)                          # Call the filmStability function by inputing the correct files.

print(len(film1),len(film2),len(film3))


data1 = np.array(film1)                                                                             # Convert the python lists to a numpy arrays.
data2 = np.array(film2)
data3 = np.array(film3)
mseRelativeData = np.array(mseRelativeList)

average_over = 9                                                                                    # Average over this many frames.
averaged_microns1 = []                                                                              # Intialize a list to store the averaged values.
averaged_microns2 = []
averaged_microns3 = []

counter = 0                                                                                         # Count the number of frames.
summation = 0                                                                                       # Sum temporary values.
for x in range(int(len(film1)/average_over)*average_over):                                          # If there are 15789 frames analyzed, and the average is 10, this rounds it down. e.g. 15780.
    counter = counter + 1                                                                           # Counter.
    summation = film1[x] + summation                                                                # Summation.
    if counter == average_over:                                                                     # When the counter reaches the averaging amount
        counter = 0                                                                                 # Reset the counter.
        averaged_microns1.append(summation/average_over)                                            # Store the averaged value.
        summation = 0                                                                               # Reset the sum for the next batch of measurements.
counter = 0                                                 
summation = 0                                               
for x in range(int(len(film2)/average_over)*average_over):                                          # Same but for another measurement.
    counter = counter + 1                                          
    summation = film2[x] + summation                                 
    if counter == average_over:                                   
        counter = 0                                                 
        averaged_microns2.append(summation/average_over)            
        summation = 0  
counter = 0                                                
summation = 0                                               
for x in range(int(len(film3)/average_over)*average_over):                                          # Same but for another measurement.       
    counter = counter + 1                                           
    summation = film3[x] + summation                                 
    if counter == average_over:                                    
        counter = 0                                                 
        averaged_microns3.append(summation/average_over)             
        summation = 0  

fps = 2                                                                                             # The fps of the video being analyzed.
seconds = 60

time1 = np.arange(len(averaged_microns1))                                                           # Length of the averaged values.
fig = plt.figure(1)


#Same graph option

ax1 = fig.add_subplot(111)
redline, = ax1.plot(time1, averaged_microns1, 'r.')
blueline, = ax1.plot(time1, averaged_microns2, 'b.')
greenline, = ax1.plot(time1, averaged_microns3, 'g.')


ticks_x1 = ticker.FuncFormatter(lambda time1, pos: '{0:g}'.format(time1/(seconds*fps)*average_over)) # Adjust the x-axis to be the correct time in minutes
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Stability of Film Thickness Over Time Multiple Locations",fontsize = 28)
ax1.legend((redline,blueline,greenline),('Middle of Device','Top of Device','Bottom of Device'),loc = 1,fontsize = 18)
plt.show()                                                                                          # Show the plot. 'q' will exit from it.

header = "Film Depth in Microns Data 1"
np.savetxt('TrackingAvg9data1.dat',data1,header = header)
header = "Film Depth in Microns Data 2"
np.savetxt('TrackingAvg9data2.dat',data2,header = header)
header = "Film Depth in Microns Data 3"
np.savetxt('TrackingAvg9data3.dat',data3,header = header)