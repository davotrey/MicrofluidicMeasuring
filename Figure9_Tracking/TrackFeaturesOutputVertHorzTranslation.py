import numpy as np
import cv2
#import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.ticker as ticker

# GLOBAL VARIABLES and DEFINITIONS
BLACK = 0                                                                                           # Value for black.
WHITE = 255                                                                                         # The value for white.
LEFT = 1                                                                                            # Leftmost column on the screen.
RIGHT = 1279                                                                                        # Rightmost column on the screen.
TOP = 1                                                                                             # Top row on the screen.
BOTTOM = 1023                                                                                       # Bottom row on the screen. 
LEFT_QUARTER = 400                                                                                  # The column value for the left quarter of the screen. We will scan between the two quartiles.
RIGHT_QUARTER = 1000                                                                                # The column value for the right quarter of the screen.
SHOW = 1                                                                                            # If the show variable (eg showFiber) is one, display more detail.
column_scan = []                                                                                    # List used as a global variable to store the column location during scans.
film1 = []                                                                                          # Lists to hold the film thickness of one frame averaged.
film2 = []   
film3 = []   
mseRawDataList = []
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

    for j in range (LEFT,RIGHT):                                                                    # Scan all columns from left to right.
        for i in range(TOP,BOTTOM):                                                                 # Scan all rows fromt top to bottom.
            px = frame[i,j]                                                                         # Save the pixel value, where [y,x] since it is inverted.
            if i != BOTTOM:                                                                         # If the for loop is not on the last pixel, save the upcoming pixel.
                pxplusone = frame[i+1, j]                                                           # Save the pixel to the right of our current position in the scan
            if px == WHITE:                                                                         # If we are on a white pixel.
                if pxplusone == BLACK:                                                              # And the next one is black.
                    start.append(i + 1)                                                             # Add the pixel location to the 'start' list.
            if px == BLACK:                                                                         # If the for loop is on a black pixel,
                if pxplusone == WHITE:                                                              # and the next is white,
                    end.append(i)                                                                   # add the value of the white pixel to 'end' list.
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

    # we scan from left to right across each row, then move downwards looking for the column that has the first black pixel
    for i in range(TOP,BOTTOM):                                                                     # Iterate between the top and bottom of the screen.
        for j in range(LEFT_QUARTER,RIGHT_QUARTER):                                                 # Iterate between halfway on the screen to the right quarter of the screen, which should include the device.
            px = frame[i,j]                                                                         # Pixels are [row, column] so basically [y, x], yes it's inverted.
            if j != (RIGHT_QUARTER - 1):                                                            # If we are not on the last column.
                pxplusone = frame[i,j+1]                                                            # Look ahead at the upcoming pixel in the scan, store it in pxplusone.
            if px == WHITE and pxplusone == BLACK:                                                  # If current pixel in scan is white and upcoming is black.
                column_scan.append(j+1)                                                             # Save the black pixel column location (index value plus one).
                row.append(i)                                                                       # Append the row that we found the edge.
                break
    return row



# Function that averages the grayscale values (between 0 and 255) for each pixel in the two scanning regions
# which are chosen through the input parameters. The midpoint of those two averages will be the value that will
# be used for thresholding.
def findThreshValue(imageOrVideo,frameFromVideo,filename,rectOneLeft_x,rectOneRight_x,rectOneTop_y,rectOneBot_y,show,rectTwoLeft_x,rectTwoRight_x,rectTwoTop_y,rectTwoBot_y):
    MEAN = 2                                                                                        # For finding the mean between two measurements.
    IMAGE = 0
    VIDEO = 1
    if (imageOrVideo == IMAGE):                                                                     # If the frame is a separate .jpg or .png file
        frame = cv2.imread(filename,1)                                                              # Read the image with the filename offered in the parameter.
    elif (imageOrVideo == VIDEO):                                                                   # Else if the frame is passed in from a video
        frame = frameFromVideo                                                                      # Save the frame to the same local variable that would have been used with the .jpg
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                                  # turn it gray
    # cv2.rectangle(frame,(rectOneLeft_x,rectOneTop_y),(rectOneRight_x,rectOneBot_y),(0,0,255),1)     # Draws a blue rectangle
    # cv2.rectangle(frame,(rectTwoLeft_x,rectTwoTop_y),(rectTwoRight_x,rectTwoBot_y),(0,0,255),1)     # To show the scanning regions.
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
        cv2.imshow('Original with Scan Regions', frame)                                             # Show the image with the scan regions.
        print("Threshold value for ", filename, " :", thresholdValue)                               # Output the threshold value calculated.
        cv2.waitKey(0)                                                                              # Wait for user key press.
    cv2.destroyAllWindows()                                                                         # Clear the windows.
    return thresholdValue                                                                           # Return the threshold value.


def filmStablity(fiberFilename,dryFilename,videoFilename,showFiber,showDry,showWet):
    meas1 = 300                                                                                     # This will be the row where we take our first measurement.
    meas2 = 350                                                                                     # Currently the code supports three total measurements.
    meas3 = 800
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
    # scalebar = FiberWidthTest(thresh, showFiber)                                                    # Call the FiberWidthTest function to find the pixel/micron scale.
    scalebar = .1732492337
    print('SCALEBAR: ', scalebar)                                                                   # Print out the scale found above.
    cv2.waitKey(0)                                                                                  # Wait for a key press from the user.
    # This block of code is for finding the edge of the dry device.
    dryFrameJumpback = 300                                                                           # The dry device frame is taken this many frames from the end of the video after the water is evaporated.
    camera = cv2.VideoCapture(videoFilename)                                                        # Connect to the .avi file provided in the parameter videoFilename.
    endOfVideo = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))                                          # Finds the number of frames in the video.
    print('NUMBER OF FRAMES: ', endOfVideo)                                                         # Print the number of frames.
    camera.set(cv2.CAP_PROP_POS_FRAMES, endOfVideo-dryFrameJumpback)                                # Set the frame to the end of the video if using water (it should have evaporated)
    ret, dryFrame = camera.read()                                                                 # Set the dryFrame by reading the camera at the designated frame.
    # dryFrame = cv2.imread(dryFilename,1)                                                            # If we are using a separate image (.jpg or .png) set it here.
    dryThreshVal = findThreshValue(VIDEO,dryFrame,dryFilename,750,800,410,500,0,500,550,410,500)   
        # Find the thresholding value, tell whether it is an IMAGE or VIDEO, send the frame if its a video, send the filename if its an image, then put the coordinates for the two boxes.
    grayed = cv2.cvtColor(dryFrame, cv2.COLOR_BGR2GRAY)                                               # Turn it gray.
    ret, dryThresh = cv2.threshold(grayed, dryThreshVal, 255, cv2.THRESH_BINARY)                      # Binary threshold the grayscale image with the value we found.
    scan_range = ScanningRegion(dryThresh, showDry)                                                 # Calls a function to find the scanning range, changing the global list column_scan and the local list scan_range.
    offset = BOTTOM - len(scan_range)                                                               # The offset is important if the top of the device is visible and no edge could be found at the top of the screen.
    if showDry == SHOW:                                                                             # If the parameter showDry is 1.
        camera.set(cv2.CAP_PROP_POS_FRAMES, endOfVideo-dryFrameJumpback)                                # Set the frame to the end of the video if using water (it should have evaporated)
        ret, dryFrame = camera.read()                                                                 # Set the dryFrame by reading the camera at the designated frame.
        for x in range(len(column_scan)):                                                           # For the length of the scanning list plot each coordinate with a blue pixel
            cv2.line(dryFrame, (column_scan[x],scan_range[x]), (column_scan[x],scan_range[x]), (178, 34, 34), 1)
        cv2.line(dryFrame, (LEFT_QUARTER,meas1), (RIGHT_QUARTER,meas1), (0, 0, 255), 6)           # Draw a line to show the start of the scanning region.
        cv2.line(dryFrame, (LEFT_QUARTER,meas2), (RIGHT_QUARTER,meas2), (255, 0, 0), 6)           # Draw a line to show the end of the scanning region.
        cv2.line(dryFrame, (LEFT_QUARTER,meas3), (RIGHT_QUARTER,meas3), (0, 130, 0), 6)           # Draw a line to show the end of the scanning region.
        cv2.imshow('Dry Device Edited', dryFrame)                                                   # Print a color frame to show the pixels being measured.
        cv2.imshow('Dry Device Binary', dryThresh)                                                  # Print the binary thresholded image for comparison.
        print("ScanRangeLength: ",len(scan_range), " Offset: ",offset,"ColumnScanLength: ",len(column_scan))
        cv2.waitKey(0)                                                                              # Wait for a key press from the user.




    # This block of code is for scanning the frames of the video and comparing with the dry device.
    actualStartingFrame = 500
    startingFrame = 11851                                                                           # If there is a part of the video that is not useful, change this to skip that section.
    count = actualStartingFrame                                                                           # Track the frame of the video that is being analyzed.  
    scanTolerance = 30                                                                              # How far left and right of a coordinate we will scan.
    PATTERN_LENGTH = 50
    PATTERN_WIDTH = 50
    PATTERN_X_COORD = 851
    REGION_BOTTOM = 500
    pattern = []
    
    xshift = 0
    y = 500
    x1 = 851
    mseRelative = 0
    camera.set(cv2.CAP_PROP_POS_FRAMES, actualStartingFrame)                                              # Set the frame to the starting point
    while True:                                                                                     # Keep looping here until 'q' is pressed or the video is complete.
        microns1 = 0                                                                                 # Variable to hold the depth of film in microns for measurement 1.
        measurements1 = []                                                                          # List holds the measurements of the scan for measurement 1.
        patternScan = []
        mseLowest = 250000
        xmseLowest = 250000
        diff = 10
        diffSquared = 0
        measure_sum1 = 0                                                                            # Variable to hold the sum of the list of measurements for measurement 1.
        measure_avg1 = 0                                                                            # Variable to hold the average value of depth in pixels for measurement 1.
        measurements1.clear()                                                                       # Reset the first list.
        count = count + 1                                                                           # Frame counter.
        ret, frame = camera.read()                                                                  # Read a frame.
   
        cv2.rectangle(frame,(850,500),(856,500-PATTERN_LENGTH),(0,0,255),1)
        if ((count == actualStartingFrame + 1)):# or (count % 500 == 0)):                                    # Every 500 pixels and at the beginning of the video, set the threshold value again.
            wetThreshVal = findThreshValue(VIDEO,frame,videoFilename,750,800,410,500,0,500,550,410,500)   
            if (showWet == SHOW):                                                                       
                print("Wet Threshold Value: ", wetThreshVal)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                              # turn it gray
        ret, thresh = cv2.threshold(gray, wetThreshVal, 255, cv2.THRESH_BINARY)                     # binary thresholding 127 and 255 are the standard values used
        cv2.imshow('Binary Threshold', thresh)                                                      # draw the result - this is the video playing

        if (count == actualStartingFrame + 1):
            for columns in range(PATTERN_WIDTH):
                for rows in range(PATTERN_LENGTH):
                    pattern.append(gray[y-rows,x1+columns])
                    patternScan.append(gray[y+150-rows,x1+150+columns])
                    # print("I'm Here. Column: ",columns,"Row: ",rows)
            for j in range(len(patternScan)):
                    diff = pattern[j] - patternScan[j]
                    diffSquared = diff * diff
                    mseRelative = diffSquared + mseRelative
            camera.set(cv2.CAP_PROP_POS_FRAMES, startingFrame)                                              # Set the frame to the starting point
            count = startingFrame

        # print(mseRelative)

        for x in range(120):
            z = REGION_BOTTOM - x      # The y coordinate is scanning from bottom to top
            # print(y)
            patternScan.clear()
            # patternScanAvg = 0
            mse = 0
            for columns in range(PATTERN_WIDTH):
                for rows in range(PATTERN_LENGTH):
                    patternScan.append(gray[z-rows,PATTERN_X_COORD+columns])
            for j in range(len(patternScan)):
                diff = pattern[j] - patternScan[j]
                diffSquared = diff * diff
                mse = diffSquared + mse
                    # print("pattern[j]",j, " ",pattern[j],"patternScan[j]",patternScan[j],"diff: ",diff,"difference Squared: ",diffSquared,"mse: ",mse)
                    # print(gray[y-j,PATTERN_X_COORD])
                    # patternScanAvg = patternScan[j] + patternScanAvg            
            # mse = (patternAvg - patternScanAvg) * (patternAvg - patternScanAvg)
            # print("patternAvg: ", patternAvg,"patternScanAvg: ",patternScanAvg,"mse: ",mse)
            if (mse < mseLowest):
                mseLowest = mse
                y = z
                # print("patternScanAvg: ",patternScanAvg)
                # print("mse: ",mseLowest)
        for i in range(10):
            z = PATTERN_X_COORD + i
            patternScan.clear()
            xmse = 0
            for columns in range(PATTERN_WIDTH):
                for rows in range(PATTERN_LENGTH):
                    patternScan.append(gray[y-rows,z+columns])
            for j in range(len(patternScan)):
                diff = pattern[j] - patternScan[j]
                diffSquared = diff * diff
                xmse = diffSquared + xmse
            # print ("i: ",i,"xmse: ",xmse)
            if (xmse < xmseLowest):
                xmseLowest = xmse
                xshift = i
        # print("y: ",y,"xshift: ",xshift)

        # for x in range(len(column_scan)):
        #     column_scan[x] = column_scan[x] + xshift

        cv2.rectangle(frame,(850,y),(890,y),(0,0,255),1)
        cv2.rectangle(frame,(PATTERN_X_COORD+xshift-1,y),(PATTERN_X_COORD+xshift+PATTERN_WIDTH,y-PATTERN_LENGTH),(0,255,0),1)
        mseRawDataList.append(xmseLowest)
        mseRelativeList.append(xmseLowest/mseRelative)
        yshiftList.append(y)
        xshiftList.append(xshift)

        yscan = y - 175
        cv2.rectangle(frame,(400,yscan),(900,yscan+20),(0,0,255),1)
        cv2.imshow('Original', frame)                                                               # Draw the result.
        
        
        for j in range(0,20):                                                                 # Scan a specified height (eg 20 pixels tall).
            xcoordinate1 = column_scan[yscan + j - offset]                                          # x coordinate or the column of the edge of the dry device.
            ycoordinate1 = scan_range[yscan + j - offset]                                           # y coordinate or the row of the edge of the dry device.
            for x in range(xcoordinate1 - scanTolerance, xcoordinate1 + scanTolerance):             # Scan in front of the device and a little behind, in case of noise.
                px1 = thresh[ycoordinate1, x]                                                       # Scanning location, where the coordinate is (y,x), it is reversed which is confusing.
                pxplusone1 = thresh[ycoordinate1, x + 1]                                            # Look one pixel ahead of scanning location.
                if px1 == WHITE and pxplusone1 == BLACK:                                            # If the scanning location is black and the next pixel is white
                    measurements1.append(xcoordinate1 - (x + 1))                                    # Save the difference of the dry column and wet column (thickness of film in pixels).
                    break
        # print("length of measurements1: ", len(measurements1))
        for x in range(len(measurements1)):                                                         # For all of the measurements in meas1.
            measure_sum1 = measure_sum1 + measurements1[x]                                          # Sum each measurement in the list
        if len(measurements1) != 0:                                                                 # If the measurement list actually has values
            measure_avg1 = measure_sum1 / len(measurements1)                                        # Take the average by dividing the sum by the number of measurements.
            microns1 = measure_avg1 * scalebar                                                      # Convert the pixel depth of film to microns using the scalebar found previously.
            film1.append(microns1 + (xshift*scalebar))                                                                  # Add each frame's measurement to a list called film.                                                               
        # if count == 505:
        #     count = 2500
        #     camera.set(cv2.CAP_PROP_POS_FRAMES, 2500)   
        # if count == 2505:
        #     count = 5000
        #     camera.set(cv2.CAP_PROP_POS_FRAMES, 5000)     
        # if count == 5005:   
        #     count = 7500
        #     camera.set(cv2.CAP_PROP_POS_FRAMES, 7500)                                         # Set the frame to the starting point
        # if count == 7505:
        #     count = 10000
        #     camera.set(cv2.CAP_PROP_POS_FRAMES, 10000)                                              # Set the frame to the starting point
        if count == 12450:                                                    # Once we reach the end of the portion of video we want to analyze
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
filmStablity("Fiber.jpg","Dry.jpg","Wet.avi",SHOW,SHOW,SHOW)     # Call the filmStability function by inputing the correct files.

print(len(film1),len(mseRawDataList),len(yshiftList))


data1 = np.array(film1)                                                                             # Convert the python list to a numpy array.

mseRawData = np.array(mseRawDataList)
mseRelativeData = np.array(mseRelativeList)
xshiftData = np.array(xshiftList)
yshiftData = np.array(yshiftList)


average_over = 9
averaged_microns1 = []   # intialize a list to store the averaged values

counter = 0                                                 # int for counting the number of frames
summation = 0                                               # int that takes the temporary sums
for x in range(int(len(film1)/average_over)*average_over):        # if there are 15789 frames analyzed, and the average is 10, this rounds it down. e.g. 15780
    counter = counter + 1                                           # a counter
    summation = film1[x] + summation                                 # a summation
    if counter == average_over:                                     # when the counter reaches the averaging amount
        counter = 0                                                 # reset the counter
        averaged_microns1.append(summation/average_over)             # store the averaged value
        summation = 0                                               # reset the sum for the next batch of x measurements

fps = 2                                # 10 frames per second based on the THOR documentation
seconds = 60

time1 = np.arange(len(data1))    # length of the averaged values
scale_time = (len(data1))/(seconds*fps) # time in minutes of averaged values
fig = plt.figure(1)


#Same graph option

ax1 = fig.add_subplot(311)
redline, = ax1.plot(time1, data1, 'ro')
ax2 = fig.add_subplot(312)
ax2.plot(time1, xshiftData, 'bo')
ax3 = fig.add_subplot(313)
ax3.plot(time1, mseRawData, 'bo')

ticks_x1 = ticker.FuncFormatter(lambda time1, pos: '{0:g}'.format(time1/(seconds*fps))) #adjust the x-axis to be the correct time in minutes
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Averaged Stability of Film Thickness Over Time Single Location",fontsize = 28)

ax2.xaxis.set_major_formatter(ticks_x1)
ax2.set_xlabel("Time in Minutes",fontsize = 24)
ax2.set_ylabel("Shift in Pixels",fontsize = 24)
ax2.set_title("Vertical Shift in Pixels over Time",fontsize = 28)

ax3.xaxis.set_major_formatter(ticks_x1)
ax3.set_xlabel("Time in Minutes",fontsize = 24)
ax3.set_ylabel("Mean Squared Error",fontsize = 24)
ax3.set_title("Smallest MSE over Time",fontsize = 28)
# ax1.legend((redline,blueline,greenline),('Middle of Device','Top of Device','Bottom of Device'),loc = 1,fontsize = 18)

# ticks_x2 = ticker.FuncFormatter(lambda time2, pos: '{0:g}'.format(time2/600)) #adjust the x-axis to be the correct time in minutes
# ax2.xaxis.set_major_formatter(ticks_x2)
# ax2.set_xlabel("Time in Minutes")
# ax2.set_ylabel("Film Thickness (um)")
# ax2.set_title("Averaged Stability of Film Thickness Over Time")
# ax2.legend(('Top Point', '1/3 Down', '2/3 Down'), loc = 'upper right')

plt.subplots_adjust(hspace=.7)          # widen the gap between the two plots

plt.show()                                                                         # Show the plot. 'q' will exit from it.

header = "Film Depth in Microns"
np.savetxt('FilmDepth.dat',data1,header = header)
header = "Vertical Shift Upwards in pixels"
np.savetxt('yshift.dat',yshiftData,header = header)
header = "Horizontal Shift to the Right in Pixels"
np.savetxt('xshift.dat',xshiftData,header = header)
header = "Minimum MSE Values Found In Scan"
np.savetxt('MinimumMSE.dat',mseRawData,header = header)
header = "Relative MSE Values to an Arbitrary Scan Elsewhere on Device"
np.savetxt('RelativeMSE.dat',mseRelativeData,header = header)
