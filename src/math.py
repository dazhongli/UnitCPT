import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def linearfit(x, y, yerr):
    """Linear fit of x and y with uncertainty and plots results."""
    '''
    Reference: https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    '''
    
    import numpy as np
    import scipy.stats as stats
    
    x, y = np.asarray(x), np.asarray(y)
    n = y.size
    p, cov = np.polyfit(x, y, 1, w=1/yerr, cov=True)  # coefficients and covariance matrix
    yfit = np.polyval(p, x)                           # evaluate the polynomial at x
    perr = np.sqrt(np.diag(cov))     # standard-deviation estimates for each coefficient
    R2 = np.corrcoef(x, y)[0, 1]**2  # coefficient of determination between x and y
    resid = y - yfit
    chi2red = np.sum((resid/yerr)**2)/(n - 2)  # Chi-square reduced
    s_err = np.sqrt(np.sum(resid**2)/(n - 2))  # standard deviation of the error (residuals)
    
    # Confidence interval for the linear fit:
    t = stats.t.ppf(0.975, n - 2)
    ci = t * s_err * np.sqrt(    1/n + (x - np.mean(x))**2/np.sum((x-np.mean(x))**2))
    # Prediction interval for the linear fit:
    pi = t * s_err * np.sqrt(1 + 1/n + (x - np.mean(x))**2/np.sum((x-np.mean(x))**2))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.fill_between(x, yfit+pi, yfit-pi, color=[1, 0, 0, 0.1], edgecolor='')
    plt.fill_between(x, yfit+ci, yfit-ci, color=[1, 0, 0, 0.15], edgecolor='')
    plt.errorbar(x, y, yerr=yerr, fmt = 'bo', ecolor='b', capsize=0)
    plt.plot(x, yfit, 'r', linewidth=3, color=[1, 0, 0, .8])
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.title('$y = %.2f \pm %.2f + (%.2f \pm %.2f)x \; [R^2=%.2f,\, \chi^2_{red}=%.1f]$'
              %(p[1], perr[1], p[0], perr[0], R2, chi2red), fontsize=20, color=[0, 0, 0])  
    plt.xlim((0, n+1))
    plt.show()

def simple_average(x, y, )