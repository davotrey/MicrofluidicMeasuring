# The following code tracks the features of light shining through the device, which maintain their shape throughout an extended video 
# To do this, it scans a rectangular region that is a certain number of pixels tall and wide. The larger the region the longer the computations take.
# Within the region should be some unique features that aren't on the edge of the device.
# An original pattern of gray values (0-255) are stored from the actual starting frame, then each frame analyzed afterwards is compared.
# The same index of the pattern list and of the scanning pattern list are subtracted, giving either a positive or a negative number or 0.
# By squaring this value and summing each pixel for the whole region, a large value will be produced. This is the squared error (mean if we averaged it as well)
# The script will search a bounded region that we know the device shifts within and fill find how far vertically and horizontally...
# by finding the smallest MSE value. Because of how many multiplactions happen, this takes a while to run for longer videos.
# Thus, it outputs files that hold all of the data taken, which can be spliced together or analyzed in a different script.

# The red box shows the orinal region. The green box shows where the software thinks the pattern or feature has shifted.
# Pressing q will end the video early, and you can change the startingFrame to begin from where it left off.




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
mseRawDataList = []                                                                                 # Lists to hold data relevant to the tracking
mseRelativeList = []                                                        
yshiftList = []
xshiftList = []


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
    # This block of code is for scanning the frames of the video and comparing with the dry device.
    actualStartingFrame = 500                                                                       # Where the video finds the initial location of the device.
    startingFrame = 501                                                                             # Skip unuseful or previously analyzed sections by skipping from the actual starting frame to this one.
    count = actualStartingFrame                                                                     # Track the frame of the video that is being analyzed.  
    PATTERN_HEIGHT = 50                                                                             # Length of the region with which patterns will be looked for.
    PATTERN_WIDTH = 50                                                                              # Width of the region with which patterns will be looked for.
    PATTERN_X_COORD = 851                                                                           # X coordinate, or column for the region scanned.
    REGION_BOTTOM = 500                                                                             # Y coordinate, or row for the bottom of the region scanned.
    pattern = []                                                                                    # List to hold the gray values (0-255) of each pixel in a 50 by 50 box (PATTERN_WIDTH/LENGTH).
    patternScan = []                                                                                # List to hold the gray values (0-255) of each pixel in a 50 by 50 box (PATTERN_WIDTH/LENGTH).
    totalVertShift = 80                                                                             # The total vertical shift is no more than 80 pixels.
    totalHorzShift = 10                                                                             # The total horizontal shift is no more than 10 pixels.

    xshift = 0                                                                                      # How much the device has shifted in the x axis.
    ycoordinate = REGION_BOTTOM                                                                     # Tracking the ycoordinate of the most pertinent region.
    xcoordinate = PATTERN_X_COORD                                                                   # For generating the first region.
    camera = cv2.VideoCapture(videoFilename)                                                        # Connect to the video file.
    camera.set(cv2.CAP_PROP_POS_FRAMES, actualStartingFrame)                                        # Set the frame to the starting point
    while True:                                                                                     # Keep looping here until 'q' is pressed or the video is complete.
        mseLowest = 250000                                                                          # The standard that we should be under to even consider the MSE to be close to the pattern.
        xmseLowest = mseLowest                                                                      # Same as above but used for the horizontal scanning.
        mseRelative = 1063888.25                                                                    # Many arbitrary scans were taken during tracking, and the mean value was this.
        diff = 0                                                                                    # Holds the value of the difference between pixel of index [i] for both the pattern and patternScan lists.
        diffSquared = 0                                                                             # Holds the value of the difference (can be positive or negative) squared.
        count = count + 1                                                                           # Frame counter.
        ret, frame = camera.read()                                                                  # Read a frame.
   
        cv2.rectangle(frame,(PATTERN_X_COORD-1,REGION_BOTTOM),(PATTERN_X_COORD+PATTERN_WIDTH,REGION_BOTTOM-PATTERN_HEIGHT),(NO_HUE,NO_HUE,FULL_HUE),1)   # Draw a red box to show the orignal scan for the list pattern[]
        if ((count == actualStartingFrame + 1)):                                                    # At the beginning of the video, set the threshold value.
            wetThreshVal = findThreshValue(VIDEO,frame,videoFilename,750,800,410,500,HIDE,500,550,410,500)   # Find the threshold value.
            if (showWet == SHOW):                                                                   
                print("Wet Threshold Value: ", wetThreshVal)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                              # Turn it gray.
        ret, thresh = cv2.threshold(gray, wetThreshVal, 255, cv2.THRESH_BINARY)                     # Binary thresholding.
        cv2.imshow('Binary Threshold', thresh)                                                      # Draw the result - this is the video playing.

        if (count == actualStartingFrame + 1):                                                      # At the beginning of the video
            for columns in range(PATTERN_WIDTH):                                                    # Iterate through the columns for the width of the desired region.
                for rows in range(PATTERN_HEIGHT):                                                  # Iterate rows for length.
                    pattern.append(gray[ycoordinate-rows,xcoordinate+columns])                      # Store the pattern into a list of gray values (0-255).
            camera.set(cv2.CAP_PROP_POS_FRAMES, startingFrame)                                      # Set the frame to the starting point.
            count = startingFrame                                                                   # Jump the counter ahead as well.
        
        # Scan a 50 by 50 pixel region each time and iterate by one pixel vertically, looking for the coordinate with a region most similar to the original.
        for vert in range(totalVertShift):
            scan = REGION_BOTTOM - vert                                                             # The y coordinate is scanning from bottom to top.
            patternScan.clear()                                                                     # Clear the patternScan, a temporary list, each iteration.
            mse = 0                                                                                 # Reset the variable that holds the mean squared error.
            for columns in range(PATTERN_WIDTH):                                                    # Iterate columns.
                for rows in range(PATTERN_HEIGHT):                                                  # Iterate rows.
                    patternScan.append(gray[scan-rows,PATTERN_X_COORD+columns])                     # Add gray value to the temporary list patternScan[]
            for j in range(len(patternScan)):                                                       # All pixels in the scanned region.
                diff = pattern[j] - patternScan[j]                                                  # Find the difference.
                diffSquared = diff * diff                                                           # Square the differences.
                mse = diffSquared + mse                                                             # Set mse to the sum of all the squared differences.
                # For debugging: print("pattern[j]",j, " ",pattern[j],"patternScan[j]",patternScan[j],"diff: ",diff,"difference Squared: ",diffSquared,"mse: ",mse)
            if (mse < mseLowest):                                                                   # If the mse of any of the iterations of vert are the smallest yet seen
                mseLowest = mse                                                                     # Update the lowest mse.
                ycoordinate = scan                                                                  # Set the new y coordinate to the region that was most similar.
        for horz in range(totalHorzShift):                                                          # Scan horizontally for greater than the maximum horizontal shift.
            scan = PATTERN_X_COORD + horz                                                           # Shift to the right. 
            patternScan.clear()                                                                     # Clear the temporary list.
            xmse = 0                                                                                # Reset the variable that holds the mean squared error. 
            for columns in range(PATTERN_WIDTH):                                                    # Iterate columns.
                for rows in range(PATTERN_HEIGHT):                                                  # Iterate rows.
                    patternScan.append(gray[ycoordinate-rows,scan+columns])                         # Add gray value to the temporary list patternScan[]
            for j in range(len(patternScan)):                                                       # All pixels in the scanned region. 
                diff = pattern[j] - patternScan[j]                                                  # Find the difference.
                diffSquared = diff * diff                                                           # Square the differences.
                xmse = diffSquared + xmse                                                           # Set xmse to the sum of all the squared differences.
            if (xmse < xmseLowest):                                                                 # If the mse of any of the iterations of horz are the smallest yet seen
                xmseLowest = xmse                                                                   # Update the lowest mse.
                xshift = horz                                                                       # Set the xshift to the region that was most similar.
        # For debugging: print("y: ",y,"xshift: ",xshift)

        cv2.rectangle(frame,(PATTERN_X_COORD+xshift-1,ycoordinate),(PATTERN_X_COORD+xshift+PATTERN_WIDTH,ycoordinate-PATTERN_HEIGHT),(NO_HUE,FULL_HUE,NO_HUE),1) # Green box for the region most like the original red box.
        mseRawDataList.append(xmseLowest)                                                           # The raw data for what MSE values were seen.
        mseRelativeList.append(xmseLowest/mseRelative)                                              # The lowest MSE value seen should be compared to an arbitrary high mse value.
        yshiftList.append(ycoordinate)                                                              # Add the current ycoordinate to save the vertical shift.
        xshiftList.append(xshift)                                                                   # Add the x shift.
        cv2.imshow('Original', frame)                                                               # Draw the result.
        
        # Uncomment these lines if you want to see different snapshots of the whole video.        
        # if count == 505:
        #     count = 2500
        #     camera.set(cv2.CAP_PROP_POS_FRAMES, 2500)   
        # if count == 2505:
        #     count = 5000
        #     camera.set(cv2.CAP_PROP_POS_FRAMES, 5000)     
        # if count == 5005:   
        #     count = 7500
        #     camera.set(cv2.CAP_PROP_POS_FRAMES, 7500)                                        
        # if count == 7505:
        #     count = 10000
        #     camera.set(cv2.CAP_PROP_POS_FRAMES, 10000)                                              
        
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

filmStablity("Notneeded.jpg","Notneeded.jpg","WetVidJuly20.avi",HIDE,HIDE,SHOW)                     # Call the filmStability function by inputing the correct files.

print(len(mseRawDataList),len(yshiftList))                                                          # Output the lengths of some lists to verify correct computation.

mseRawData = np.array(mseRawDataList)                                                               # Convert python lists to numpy arrays.
mseRelativeData = np.array(mseRelativeList)                                                         
xshiftData = np.array(xshiftList)                                                                   
yshiftData = np.array(yshiftList)                                                                   

fps = 2                                                                                             # The fps of the video being analyzed.
seconds = 60                                                                                        # Seconds in a minute.

time1 = np.arange(len(yshiftList))                                                                  # Numpy array from 0 to the length of the python list.
fig = plt.figure(1)                                                                                 # Initialize a figue.

# Graph formatting:
axcoordinate = fig.add_subplot(311)                                                                 # (number of graphs,row,column)
axcoordinate.plot(time1, yshiftData, 'bo')                                                          # x and y axis for a graph, with blue dots.
ax2 = fig.add_subplot(312)
ax2.plot(time1, xshiftData, 'bo')
ax3 = fig.add_subplot(313)
ax3.plot(time1, mseRelativeData, 'bo')

ticks_xcoordinate = ticker.FuncFormatter(lambda time1, pos: '{0:g}'.format(time1/(seconds*fps)))    # Adjust the x-axis to be the correct time in minutes
axcoordinate.xaxis.set_major_formatter(ticks_xcoordinate)
axcoordinate.set_xlabel("Time in Minutes",fontsize = 24)
axcoordinate.set_ylabel("Vertical Shift in Pixels",fontsize = 24)
axcoordinate.set_title("Averaged Stability of Film Thickness Over Time Single Location",fontsize = 28)

ax2.xaxis.set_major_formatter(ticks_xcoordinate)
ax2.set_xlabel("Time in Minutes",fontsize = 24)
ax2.set_ylabel("Horizontal Shift in Pixels",fontsize = 24)
ax2.set_title("Vertical Shift in Pixels over Time",fontsize = 28)

ax3.xaxis.set_major_formatter(ticks_xcoordinate)
ax3.set_xlabel("Time in Minutes",fontsize = 24)
ax3.set_ylabel("Mean Squared Error Relative to Arbitrary Scan",fontsize = 24)
ax3.set_title("Relative MSE",fontsize = 28)

plt.subplots_adjust(hspace=.7)                                                                      # Widen the gap between the two plots
plt.show()                                                                                          # Show the plot. 'q' will exit from it.

header = "Vertical Shift Upwards in pixels"                                                         # Creates a header that will appear in the first line of a dat file generated.
np.savetxt('yshift.dat',yshiftData,header = header)                                                 # Generates a dat file given certain data.
header = "Horizontal Shift to the Right in Pixels"  
np.savetxt('xshift.dat',xshiftData,header = header)
header = "Minimum MSE Values Found In Scan"
np.savetxt('MinimumMSE.dat',mseRawData,header = header)
header = "Relative MSE Values to an Arbitrary Scan Elsewhere on Device"
np.savetxt('RelativeMSE.dat',mseRelativeData,header = header)
