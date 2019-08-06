import numpy as np
import cv2
#import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.ticker as ticker

filmDepthData1 = np.loadtxt('FilmDepth167-64.txt') 
filmDepthData2 = np.loadtxt('FilmDepth325-64.txt') 
filmDepthData3 = np.loadtxt('FilmDepth850-64.txt') 
# yShiftData = np.loadtxt('yshift500to12450.txt') 
# xShiftData = np.loadtxt('xshift500to12450.txt')
# relativeMSEData = np.loadtxt('RelativeMSE500to12450.txt')
# np.append(yShiftData,0)
# np.append(xShiftData,0)
# np.append(relativeMSEData,0)

print(filmDepthData1.shape,filmDepthData2.shape,filmDepthData3.shape)
np.append(filmDepthData1,0)
np.append(filmDepthData2,0)
np.append(filmDepthData3,0)
print(filmDepthData1.shape,filmDepthData2.shape,filmDepthData3.shape)



time = np.arange(len(filmDepthData1))
fps = 2
seconds = 60
scaleTime = (len(filmDepthData1))/(seconds*fps) # time in minutes of averaged values

fig = plt.figure(1)

#Same graph option

ax1 = fig.add_subplot(111)
redline, = ax1.plot(time, filmDepthData2, 'ro')
# blueline, = ax1.plot(time, filmDepthData1, 'bo')
# greenline, = ax1.plot(time, filmDepthData3, 'go')
# ax2 = fig.add_subplot(412)
# ax2.plot(time, yShiftData, 'bo')
# ax3 = fig.add_subplot(413)
# ax3.plot(time, xShiftData, 'bo')
# ax4 = fig.add_subplot(414)
# ax4.plot(time, relativeMSEData,'bo')

ticks_x1 = ticker.FuncFormatter(lambda time, pos: '{0:g}'.format(time/(seconds*fps))) #adjust the x-axis to be the correct time in minutes
ax1.xaxis.set_major_formatter(ticks_x1)
ax1.set_xlabel("Time in Minutes",fontsize = 24)
ax1.set_ylabel("Film Thickness (um)",fontsize = 24)
ax1.set_title("Averaged Stability of Film Thickness Over Time Multiple Locations",fontsize = 28)

# ax2.xaxis.set_major_formatter(ticks_x1)
# ax2.set_xlabel("Time in Minutes",fontsize = 24)
# ax2.set_ylabel("Shift in Pixels",fontsize = 24)
# ax2.set_title("Vertical Shift in Pixels over Time",fontsize = 28)


# ax3.xaxis.set_major_formatter(ticks_x1)
# ax3.set_xlabel("Time in Minutes",fontsize = 24)
# ax3.set_ylabel("Shift in Pixels",fontsize = 24)
# ax3.set_title("Horizontal Shift in Pixels over Time",fontsize = 28)

# ax4.xaxis.set_major_formatter(ticks_x1)
# ax4.set_xlabel("Time in Minutes",fontsize = 24)
# ax4.set_ylabel("Mean Squared Error",fontsize = 24)
# ax4.set_title("Relative MSE to a Mean Arbitrary Scan",fontsize = 28)

plt.subplots_adjust(hspace=.7)          # widen the gap between the two plots

plt.show()                                    