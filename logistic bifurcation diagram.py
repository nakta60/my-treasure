import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]
startval = 2
endval = 4
xvals = np.array([[],[]])
n_iter = 1000
n_plot = 100

def logistic(xk,r):
    return r*xk*(1-xk)

for r in np.arange(startval,endval,0.00025):
    x = 0.5
    for i in range(n_iter):
        x = logistic(x,r)
        if i == n_iter-n_plot:
            xss = x
        if i > n_iter-n_plot:
            xvals = np.append(xvals,np.array([[r],[x]]),axis=1)
            if np.abs(x-xss) < 0.001:
                break
plt.plot(xvals[0,:],xvals[1,:],'.',ms=0.1,color='k')
plt.show()