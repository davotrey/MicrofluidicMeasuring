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
    scalebar = .1632492337
    print('SCALEBAR: ', scalebar)                                                                   # Print out the scale found above.
    cv2.waitKey(0)                                                                                  # Wait for a key press from the user.
    # This block of code is for finding the edge of the dry device.
    dryFrameJumpback = 510    
    # dryFrameJumpback = 300                                                                        # The dry device frame is taken this many frames from the end of the video after the water is evaporated.
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
    offset = BOTTOM + 1 - len(scan_range)                                                               # The offset is important if the top of the device is visible and no edge could be found at the top of the screen.
    if showDry == SHOW:                                                                             # If the parameter showDry is 1.
        camera.set(cv2.CAP_PROP_POS_FRAMES, endOfVideo-dryFrameJumpback)                                # Set the frame to the end of the video if using water (it should have evaporated)
        ret, dryFrame = camera.read()                                                                 # Set the dryFrame by reading the camera at the designated frame.
        for x in range(len(column_scan)):                                                           # For the length of the scanning list plot each coordinate with a blue pixel
            cv2.rectangle(dryFrame,(600,260-64),(800,450-64),(0,0,255),1)
            cv2.line(dryFrame, (column_scan[x],scan_range[x]), (column_scan[x],scan_range[x]), (178, 34, 34), 1)
        # cv2.line(dryFrame, (LEFT_QUARTER,meas1), (RIGHT_QUARTER,meas1), (0, 0, 255), 6)           # Draw a line to show the start of the scanning region.
        # cv2.line(dryFrame, (LEFT_QUARTER,meas2), (RIGHT_QUARTER,meas2), (255, 0, 0), 6)           # Draw a line to show the end of the scanning region.
        # cv2.line(dryFrame, (LEFT_QUARTER,meas3), (RIGHT_QUARTER,meas3), (0, 130, 0), 6)           # Draw a line to show the end of the scanning region.
        cv2.imshow('Dry Device Edited', dryFrame)                                                   # Print a color frame to show the pixels being measured.
        cv2.imshow('Dry Device Binary', dryThresh)                                                  # Print the binary thresholded image for comparison.
        print("ScanRangeLength: ",len(scan_range), " Offset: ",offset,"ColumnScanLength: ",len(column_scan))
        cv2.waitKey(0)                                                                              # Wait for a key press from the user.


    yShiftData = np.loadtxt('yshift500to12450.txt')                                                 # Load the vertical shift data for each frame.
    yShiftData.tolist()                                                                             # Convert from numpy array to list.
    xShiftData = np.loadtxt('xshift500to12450.txt')                                                 # Load the horizontal shift data for each frame.
    xShiftData.tolist()                                                                             # Convert from numpy array to list.
    maxVert = yShiftData[0]                                                                         # The vertical coordinate at frame 0.
    minVert = yShiftData[len(yShiftData)-1]                                                         # The vertical coordinate at the end of the video.
    totalVertShift = maxVert - minVert                                                              # The total shift is the difference between vertical coordinates.
    print("minVert: ",minVert," maxVert: ",maxVert," totalVertShift: ",totalVertShift)              # Print out the values

    row_scan_shifted = []                                                                           # Initialize lists to hold the shifted dry edge of the device.
    column_scan_shifted = []

    # This block of code is for scanning the frames of the video and comparing with the dry device.
    actualStartingFrame = 500                                                                       # The frame where we begin measuring
    count = actualStartingFrame                                                                     # Track the frame of the video that is being analyzed with count.
    scanTolerance = 25                                                                              # How wide the region is that we scan for film depth based on the dry location.
    camera.set(cv2.CAP_PROP_POS_FRAMES, actualStartingFrame)                                        # Set the frame to the starting frame.
    while True:                                                                                     # Keep looping here until 'q' is pressed or the video is complete.
        measurements = []                                                                           # List holds the measurements of the scan.
        measurements.clear()                                                                        # Clear the lists every new frame
        row_scan_shifted.clear()                                                    
        column_scan_shifted.clear()
        count = count + 1                                                                           # Frame counter.
        ret, frame = camera.read()                                                                  # Read a frame.


        if ((count == actualStartingFrame + 1)):# or (count % 500 == 0)):                           # Every 500 pixels and at the beginning of the video, set the threshold value again.
            wetThreshVal = findThreshValue(VIDEO,frame,videoFilename,750,800,410,500,0,500,550,410,500) # Call the findThreshValue function with specfic regions.   
            if (showWet == SHOW):                                                                   # If showWet is true.
                print("Wet Threshold Value: ", wetThreshVal)                                        # Print the threshold value we found.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                              # Turn it gray.
        ret, thresh = cv2.threshold(gray, wetThreshVal, 255, cv2.THRESH_BINARY)                     # Binary thresholding using a value between 0 and 255 to convert all pixels to high or low.
        cv2.imshow('Binary Threshold', thresh)                                                      # Draw the result - this is the video playing.
        y = 0                                                                                       # Reset y to zero every frame.
        vertShift = int(500 - yShiftData[count-actualStartingFrame-1])                              # Every frame has the vertical shift accounted for given the data found through tracking.
        localMaxVert = int(yShiftData[count-actualStartingFrame-1])                                 # Each frame will have a different vertical coordinate.
        vertDifference = int(localMaxVert - minVert)                                                # The difference between the vertical coordinate of each frame and the final frame.
        xshift = int(6 - xShiftData[count - actualStartingFrame - 1])                               # Horizontal shift for each frame.
        for x in range(len(column_scan)):                                                           # For the length of the scanning list plot each coordinate with a blue pixel
            column_scan_shifted.append(column_scan[x]-xshift)                                       # Shift the columns by the horizontal shift loaded.
            row_scan_shifted.append(scan_range[x]+vertDifference)                                   # Shift the rows by the vertical shift loaded.
            cv2.line(frame, (column_scan_shifted[x],row_scan_shifted[x]), (column_scan_shifted[x],row_scan_shifted[x]), (0, 255, 0), 1)
        for y1 in range(260,450):                                                                   # Vertical scanning range.
            y = y1 - vertDifference - offset                                                        # Each y1 in the vertical scanning range is adjusted by the current frames vertical shift.
            xcoordinate = column_scan_shifted[y1-vertShift-vertDifference]                          # x coordinate for the dry edge of the device.
            for x in range(xcoordinate - scanTolerance, xcoordinate + scanTolerance):               # Horizontal scanning region using the scanTolerance.
                px = thresh[y1-vertShift,x]                                                         # Scanning pixel for the wet device. 
                pxplusone = thresh[y1-vertShift,x+1]                                                # Scanning pixel one ahead for the wet device.
                if px == WHITE and pxplusone == BLACK:                                              # If the scanning pixel is white and the pixel to the right is black.
                    # print((xcoordinate-xshift-(x+1))*scalebar)                                      # Printout.
                    measurements.append((xcoordinate - (x+1))*scalebar)                    # The measurement is the difference between the dry edge and the wet edge with the shift taken into account.
                    cv2.rectangle(frame,(x+1,y1-vertShift),(x+1,y1-vertShift),(0,0,255),1)                  # Draw a rectangle to show the vertical scanning region.
                    cv2.rectangle(frame,(xcoordinate,y1-vertShift),(xcoordinate,y1-vertShift),(255,0,0),1)                  # Draw a rectangle to show the vertical scanning region.
                    break                                                                           # If we find the wet edge of the device, leave the horizontal scan.
        print("y: ",y," vertDifference: ",vertDifference," vertShift: ", vertShift," xshift: ",xshift)
        cv2.rectangle(frame,(600,y1-vertShift),(800,y1-vertShift-190),(0,0,255),1)                  # Draw a rectangle to show the vertical scanning region.
        cv2.rectangle(frame,(600,260-64),(800,450-64),(0,255,0),1)                                  # Draw a rectangle to show the vertical scanning region.
        cv2.imshow('Original', frame)                                                               # Draw the result.
        cv2.imshow('Dry',dryFrame)                                                                  
        
        xaxis = np.array(measurements)                                                              # The xaxis for our graphs is the film depth.
        yaxis = np.arange(260,450)                                                                  # The yaxis is the points between our vertical scan from row 260 to row 450.
        fig = plt.figure(1)                                                                         # Initializing the figure.
        ax = fig.add_subplot(111)                                                                   # Create a figure that can hold subplots but give it just one figure (111 equates to number of figures, row, column)
        redline, = ax.plot(xaxis, yaxis, 'b')                                                       # Plot the xaxis and yaxis with a blue line.
        ax.set_ylabel("Vertical Coordinate",fontsize = 24)                                          # Set axis and plot labels.
        ax.set_xlabel("Film Thickness (um)",fontsize = 24)
        ax.set_title("Film Depth Along Widest Region of Device",fontsize = 28)
        plt.show()                                                                                  # Show plot, waits for input 'q' from user.

        # The following code jumps to different portions of the video
        if count == 503:    
            count = 1000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 1000) 
        if count == 1003:
            count = 1500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 1500)   
        if count == 1503:
            count = 2000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 2000)     
        if count == 2003:   
            count = 2500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 2500)                                         
        if count == 2503:
            count = 3000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 3000)
        if count == 3003:
            count = 3500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 3500)   
        if count == 3503:
            count = 4000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 4000)     
        if count == 4003:   
            count = 4500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 4500)                                         
        if count == 4503:
            count = 5000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 5000)                                                  
        if count == 5003:   
            count = 5500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 5500)                                         
        if count == 5503:
            count = 6000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 6000)
        if count == 6003:
            count = 6500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 6500)   
        if count == 6503:
            count = 7000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 7000)     
        if count == 7003:   
            count = 7500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 7500)                                         
        if count == 7503:
            count = 8000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 8000) 
        if count == 8003:   
            count = 8500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 8500)                                         
        if count == 8503:
            count = 9000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 9000)
        if count == 9003:
            count = 9500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 9500)   
        if count == 9503:
            count = 11000
            camera.set(cv2.CAP_PROP_POS_FRAMES, 11000)     
        if count == 11003:   
            count = 11500
            camera.set(cv2.CAP_PROP_POS_FRAMES, 10500)                                         
        if count == 11503:
            count = 12447
            camera.set(cv2.CAP_PROP_POS_FRAMES, 12447)                                         
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

filmStablity("Fiber.jpg","Dry.jpg","Wet.avi",SHOW,SHOW,SHOW)     # Call the filmStability function by inputing the correct files.

