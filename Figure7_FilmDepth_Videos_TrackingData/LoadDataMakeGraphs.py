# Load text files and graph the data

import numpy as np
import cv2
import pylab as plt
import matplotlib.pyplot as ticker

filmDepthData1 = np.loadtxt('FilmDepth325-64.txt')                                                  # Text files to be loaded.
filmDepthData2 = np.loadtxt('FilmDepth167-64.txt')
filmDepthData3 = np.loadtxt('FilmDepth562.txt') 
filmDepthData4 = np.loadtxt('FilmDepth850-64.txt') 

print(filmDepthData1.shape,filmDepthData2.shape,filmDepthData3.shape,filmDepthData4.shape)          # Print shape of arrays.
np.append(filmDepthData1,0)                                                                         # Append to numpy array.
np.append(filmDepthData2,0)
np.append(filmDepthData3,0)
np.append(filmDepthData4,0)

time = np.arange(len(filmDepthData2))                                                               # Create an array of the same length as data taken.
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
    tick.label.set_fontsize(14) 
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
ax1.set_xlabel("Time in Minutes",fontsize = 24)                                             
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
plt.show()                                                                                          # Show plot, press q to exit.
# Top of Device                                                                                     # Same as above.
fig2 = plt.figure(1)
ax1 = fig2.add_subplot(211)
blueline, = ax1.plot(time, filmDepthData2, 'b.')
ticker.xticks(np.arange(0,len(filmDepthData1),step=tickInterval),np.arange(0,totalMin,step=tenMinutes))
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
plt.show()    
# Bottom of Device
fig3 = plt.figure(1)
ax1 = fig3.add_subplot(211)
greenline, = ax1.plot(time, filmDepthData3, 'g.')
ticker.xticks(np.arange(0,len(filmDepthData1),step=tickInterval),np.arange(0,totalMin,step=tenMinutes))
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
plt.show()    
# Neck of Device
fig4 = plt.figure(1)
ax1 = fig4.add_subplot(211)
yellowline, = ax1.plot(time, filmDepthData4, 'y.')
ticker.xticks(np.arange(0,len(filmDepthData1),step=tickInterval),np.arange(0,totalMin,step=tenMinutes))
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
plt.show()    