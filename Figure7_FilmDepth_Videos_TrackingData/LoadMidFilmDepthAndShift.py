# Load text files and graph the data

import numpy as np
import cv2
import pylab as plt
import matplotlib.ticker as ticker

filmDepthData1 = np.loadtxt('FilmDepth325-64.txt')                                                 # Text files to be loaded.
xshiftData = np.loadtxt('xshift500to12450.txt')

print(filmDepthData1.shape,xshiftData.shape)                               # Print shape of arrays.
np.append(filmDepthData1,0)                                                                         # Append to numpy array.
np.append(xshiftData,0)

time = np.arange(len(filmDepthData1))
fps = 2                                                                                             # Fps of video.
seconds = 60

xaxis = np.arange(len(xshiftData))
ticks_x1 = ticker.FuncFormatter(lambda time, pos: '{0:g}'.format(time/(seconds*fps)))               # Adjust the x-axis to be the correct time in minutes.


# Middle of Device
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
redline, = ax1.plot(time, filmDepthData1, 'r.')
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Film Thickness Averaged over Time",fontsize = 28)

# Shift Data
ax2 = fig.add_subplot(212)
ax2.plot(xaxis,xshiftData,'k.')
ax2.xaxis.set_major_formatter(ticks_x1)
ax2.set_xlabel("Time in Minutes",fontsize = 24)
ax2.set_ylabel("Horizontal Shift in Pixels",fontsize = 24)
ax2.set_title("Horizontal Shift over Time",fontsize = 24)

plt.subplots_adjust(hspace=.7)          # widen the gap between the two plots
plt.show()       



                                