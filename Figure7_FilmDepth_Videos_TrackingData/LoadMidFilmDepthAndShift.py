# Load text files and graph the data

import numpy as np
import cv2
import pylab as plt
import matplotlib.pyplot as ticker

filmDepthData1 = np.loadtxt('FilmDepth325-64.txt')                                                 # Text files to be loaded.
xshiftData = np.loadtxt('xshift500to12450.txt')

print(filmDepthData1.shape,xshiftData.shape)                               # Print shape of arrays.
np.append(filmDepthData1,0)                                                                         # Append to numpy array.
np.append(xshiftData,0)

time = np.arange(len(filmDepthData1))                                                               # Create an array of the same length as data taken.
fps = 2                                                                                             # Fps of video.
seconds = 60                                                                                        # Seconds in a minute.        
tenMinutes = 10                                                                                     # Minutes in one tick.
tickInterval = fps * seconds * tenMinutes                                                           # Length of one tick interval.
totalMin = 110                                                                                      # Lenght of video analyzed.

# Middle of Device
fig = plt.figure(1)                                                                                 # Initialize a figure.
ax1 = fig.add_subplot(211)                                                                          # Create a figure with half height to fit better.
redline, = ax1.plot(time, filmDepthData1, 'r.')                                                     # Plot the film depth with red dots over time.
ticker.xticks(np.arange(0,len(filmDepthData1),step=tickInterval),np.arange(0,totalMin,step=tenMinutes)) # Reaarange the ticks to show ten minute intervals.
for tick in ax1.xaxis.get_major_ticks():                                                            # Change the fontsize of ticks and labels.
    tick.label.set_fontsize(18) 
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(18) 
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)

# Shift Data
ax2 = fig.add_subplot(212)
ax2.plot(time,xshiftData,'k.')
ticker.xticks(np.arange(0,len(filmDepthData1),step=tickInterval),np.arange(0,totalMin,step=tenMinutes)) # Reaarange the ticks to show ten minute intervals.
for tick in ax2.xaxis.get_major_ticks():                                                            # Change the fontsize of ticks and labels.
    tick.label.set_fontsize(18) 
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(18) 
ax2.set_xlabel("Time in Minutes",fontsize = 24)
ax2.set_ylabel("Horizontal Shift (pixels)",fontsize = 24)

plt.subplots_adjust(hspace=.7)          # widen the gap between the two plots
plt.show()       



                                