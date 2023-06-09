
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.animation as anim
import pandas as pd
from scipy import signal
from sklearn.preprocessing import normalize

######################################################################
### @@@ Simple animation of the cross-correlation of 
### @@@ two random time series (the time-domain) 
### Edited 30th May 23, by Thuany 



######################################################################
##### Defining some fixed parameters and cross-correlation function
######################################################################

x = np.arange(-300, 301, 1) ## x-axis of the animation 
lagtime = np.arange(-1*len(x), len(x)+1, 1) ## range for the lag, we dont want it to be too long

## cross-correlating via panda.corr of panda series: dataframe.corr \
## is computing pairwise correlation between panda dataframes (or series in this case).
def crosscorr_panda(ts1, ts2, lagtime=0): return ts1.corr(ts2.shift(lagtime))  

## cross-correlating via numpy dot product of arrays:
def crosscorr_dot(ts1, ts2, lagtime=0): return np.dot(ts1, ts2) 
    

######################################################################
##### Defining waveforms for cross-corr
######################################################################

### Function for the animation:
x0_func, x1_func, xstep_func = 0, 181, 1 ## x-axis of the function i.e., length of the time series (let's say 180 seconds)
x_func = np.arange(x0_func, x1_func, xstep_func) 

def func(x):
    y=[]
    
    for var in x:
        if float(var) <= x0_func or float(var) >=x1_func: y.append(0)
            
        #else: y.append(np.sin(np.radians(var*8)) ) ##### For sine wave
        else: y.append(np.exp(- np.radians(var)**2 / 2) * np.cos(10*np.radians(var)+np.pi/2)) ##### For wave train
            
    return np.array(y)

'''
### Cyclic function for the animation:
#func = lambda x: np.exp(- np.radians(x)**2 / 2) * np.cos(10*np.radians(x)) ##### Wave train
#func = lambda x: signal.square(2* np.radians(x)) ##### Square wave
#func = lambda x: np.sin(np.radians(x)*4) ##### Sine wave


### Gaussian function for the animation:
mu=150; width=10
mu=0; width=10
func = lambda x: np.exp(-np.power(x-mu, 2.0) / (2* np.power(width, 2.))) ##### Gaussian function
'''

######################################################################
##### Calculating the cross-corr between waveforms
##### looping through lag times :)
######################################################################


df1 = pd.Series(func(x)) # Converting the array into panda series (neded if using panda.corr option).
df2 = pd.Series(func(x)) # This is for auto-correlation, change here to cross-correlate df1 with some other time series.
norm = np.dot(func(x), func(x))

xcorr_time = []  
for lag in lagtime:
   
    #### cross-correlating via panda.corr of panda series
    xcorr_time.append(np.nan_to_num(crosscorr_panda(df1, df2, lag)) )# replacing Nan for Zero values

    #### cross-correlating via numpy dot product of arrays
    #xcorr_time.append(np.nan_to_num(crosscorr_dot(func(x), func(x +1*lag ))) /norm)


######################################################################
#####  Figure layout
######################################################################

fig = plt.figure( figsize=(6,6) )
gs = plt.GridSpec(4,4)
ax = [fig.add_subplot(gs[0:1, 0:4]),
        fig.add_subplot(gs[1:2, 0:4]),
        fig.add_subplot(gs[2:4,0:4])]


wave0, = ax[0].plot(x, func(x), label = r"$f(x)$")

wave1, = ax[1].plot(x, func(x), color='goldenrod', label=r"$f(x - ct)$" , animated=True) # wave
dot1, = ax[1].plot([], [], 'o', color='goldenrod')

wave2, = ax[2].plot(lagtime, xcorr_time, color='grey', label = "xcorr", animated=True)
dot2, = ax[2].plot([], [], 'o', color='red')

ax[0].set_title("Cross-correlation in the time domain")

for i in range(0, 2, 1):
    ax[i].set_ylim(-1, 1)
    ax[i].set_ylabel("Amplitude",fontsize = 12,color = 'black')

    ax[i].set_xlim(-1*max(x), max(x))
    ax[i].grid(color = 'grey')

for i in range(0, 3, 1):
    ax[i].set_yticks((-1, 0, 1))
    ax[i].set_ylim(-1, 1)

    ax[i].grid(True)
    ax[i].set_facecolor('black')
    ax[i].patch.set_alpha(0.9)
    ax[i].legend(loc="lower right")


for i in range(2, 3, 1):

    samples=int(1* len(x) -1)
    ax[i].set_xticks(np.arange(-1*(samples), samples+1, samples/4 ))
    
    ax[i].set_xticklabels(np.arange(-1*(samples), samples+1, samples/4 ) )
    ax[i].set_xlim(-1*(samples) -1, samples+1)
    ax[i].set_xlabel('Lag',fontsize = 12,color = 'black')
    ax[i].set_ylabel("Normalised Amplitude",fontsize = 12,color = 'black')
    ax[i].grid(color = 'grey')


###################################
#####  Setting up animation 
###################################

## Shifting function in time: f(x) -> f(x - ct) 
def time_shift(t, b = 1):
    
    y_shift = func(x -b*t + len(x)/2) #
    wave1.set_ydata(y_shift) # this updates the plot
    #dot1.set_data(t - len(x)/2+x1_func, 0 )
    
    dt=int(t+len(x)/2) 
    dot2.set_data(t - len(x)/2, xcorr_time[dt] )
    wave2.set_data(lagtime[:dt], xcorr_time[:dt])

   

    return(wave1, wave2, dot2, dot1)

K=1 # any factor
xx_ani = anim.FuncAnimation(fig, time_shift, frames = len(x), fargs = (K,), interval = 15, blit = True)
plt.show()

## Saving animatiotn into a gif file
xx_ani.save('/Users/thuanycostadelima/Desktop/xx_anime.gif')
