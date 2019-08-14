# Load text files and graph the data

import numpy as np
import cv2
import pylab as plt
import matplotlib.ticker as ticker

filmDepthData1 = np.loadtxt('FilmDepth325-64.txt')                                                 # Text files to be loaded.
filmDepthData2 = np.loadtxt('FilmDepth167-64.txt')
filmDepthData3 = np.loadtxt('FilmDepth562.txt') 
filmDepthData4 = np.loadtxt('FilmDepth850-64.txt') 

print(filmDepthData1.shape,filmDepthData2.shape,filmDepthData3.shape,filmDepthData4.shape)                               # Print shape of arrays.
np.append(filmDepthData1,0)                                                                         # Append to numpy array.
np.append(filmDepthData2,0)
np.append(filmDepthData3,0)
np.append(filmDepthData4,0)

time = np.arange(len(filmDepthData2))
fps = 2                                                                                             # Fps of video.
seconds = 60

ticks_x1 = ticker.FuncFormatter(lambda time, pos: '{0:g}'.format(time/(seconds*fps)))               # Adjust the x-axis to be the correct time in minutes.


# Middle of Device
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
redline, = ax1.plot(time, filmDepthData1, 'r.')
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Film Thickness Averaged over Time",fontsize = 28)
plt.show()       
# Top of Device
fig2 = plt.figure(1)
ax1 = fig2.add_subplot(211)
blueline, = ax1.plot(time, filmDepthData2, 'b.')
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Film Thickness Averaged over Time",fontsize = 28)
plt.show()    
# Bottom of Device
fig3 = plt.figure(1)
ax1 = fig3.add_subplot(211)
greenline, = ax1.plot(time, filmDepthData3, 'g.')
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Film Thickness Averaged over Time",fontsize = 28)
plt.show()    
# Neck of Device
fig4 = plt.figure(1)
ax1 = fig4.add_subplot(211)
yellowline, = ax1.plot(time, filmDepthData4, 'y.')
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Film Thickness Averaged over Time",fontsize = 28)
plt.show()    