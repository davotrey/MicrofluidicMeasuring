# Load text files and graph the data

import numpy as np
import cv2
import pylab as plt
import matplotlib.ticker as ticker

filmDepthData1 = np.loadtxt('FilmDepth167-64Avg.txt')                                                  # Text files to be loaded.
filmDepthData2 = np.loadtxt('FilmDepth325-64Avg.txt') 
filmDepthData3 = np.loadtxt('FilmDepth850-64Avg.txt') 

print(filmDepthData1.shape,filmDepthData2.shape,filmDepthData3.shape)                               # Print shape of arrays.
np.append(filmDepthData1,0)                                                                         # Append to numpy array.
np.append(filmDepthData2,0)
np.append(filmDepthData3,0)
print(filmDepthData1.shape,filmDepthData2.shape,filmDepthData3.shape)

time = np.arange(len(filmDepthData1))
fps = 2                                                                                             # Fps of video.
seconds = 60
scaleTime = (len(filmDepthData1))/(seconds*fps)                                                     # Time in minutes of averaged values.

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
redline, = ax1.plot(time, filmDepthData2, 'b.')
blueline, = ax1.plot(time, filmDepthData1, 'r.')
greenline, = ax1.plot(time, filmDepthData3, 'g.')

ticks_x1 = ticker.FuncFormatter(lambda time, pos: '{0:g}'.format(time/(seconds*fps)))               # Adjust the x-axis to be the correct time in minutes.
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Averaged Stability of Film Thickness Over Time Multiple Locations",fontsize = 28)
plt.subplots_adjust(hspace=.7)                                                                      # Widen the gap between the two plots.

plt.show()                                    