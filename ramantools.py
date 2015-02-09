
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
import os

def ramanfit(filename,
             cntr=(470, 560),
             amp1=(20, 20),
             amp2=(20, 20),
             std1=(10, 5),
             std2=(10, 5),
             drange=None,
             output=True,
             step=4):

    # step = 1: Fittings the baseline
    # step = 2: Choosing initial guess for peaks
    # step = 3: Evaluate the fit
    # step = 4: View and save the final figure

    # This unpacks the data from the text file.
    S, I = np.loadtxt(filename, usecols=(0, 1), unpack=True)

    if drange == None:
        drange = [min(S), max(S)]

    # Define the low and high regions for baseline sampling
    dx = 80.
    low = drange[0] + dx
    high = drange[1] - dx

    # Seperate the data points to be used for fitting the baseline
    xbl = np.append(S[(S < low)], S[(S > high)])
    ybl = np.append(I[(S < low)], I[(S > high)])

    # Fits a line to the base line points
    blpars = np.polyfit(xbl, ybl, 1)
    blfit = np.poly1d(blpars)


    if step != 1 and step != 2 and step != 3 and step != 4:
        print 'Set step = 1, 2, 3, or 4 to continue'

    # Step 1: Choose low and high values for a satisfactory baseline
    if step == 1:
        plt.figure()
        plt.plot(S, I, label='data')
        plt.plot(S, blfit(S), 'r-', lw=2, label='base line')
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Intensity (counts)')
        plt.legend(loc='best')
        plt.show()
        print 'When you are satisfied with the fit of the base line, set step = 2'
        exit()

    # Subtracts the baseline from the intensities
    I -= blfit(S)

    # Gaussians will only be fit the the data not used for the baseline
    nS = S[(S > low) & (S < high)]
    nI = I[(S > low) & (S < high)]

    # These are functions which define the types of fit which you could implement
    # Currently, the code only utilizes Gaussians
    # ----------------------------------------------------------------------
    def gaussian(x, pars):
        A = pars[0]    # amplitude
        mu = pars[1]   # means
        sig = pars[2]  # std dev
        return A * np.exp((-(x - mu)**2.) / ((2*sig)**2.))

    def sum_gaussian(x, *p):
        g1 = gaussian(x, [p[2], p[0], p[6]])
        g2 = gaussian(x, [p[3], p[0], p[7]])
        g3 = gaussian(x, [p[4], p[1], p[8]])
        g4 = gaussian(x, [p[5], p[1], p[9]])
        return g1 + g2 + g3 + g4
    # ----------------------------------------------------------------------

    # These are initial guesses of the tuning parameters for the Gaussian fits.
    #    Peak #: 1    2
    parguess = (cntr[0], cntr[1],       # Peak center
                amp1[0], amp1[1],       # Amplitude of peak 1
                amp2[0], amp2[1],       # Amplitude of peak 2
                std1[0], std1[1],       # Standard deviation of peak 1
                std2[0], std2[1])       # Standard deviation of peak 2

    # Step 2: Fitting the curves to the data
    if step == 2:
        plt.figure()
        plt.plot(nS, nI, 'b-', label='Data')
        plt.plot(S, sum_gaussian(S, *parguess), 'g--', lw=3, label='Initial guess')
        plt.xlim(drange[0], drange[1])
        plt.ylim(0, max(nI) + 2)
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Intensity (counts)')
        plt.legend(loc='best')
        plt.show()
        print 'Once the initial guess looks reasonable, set step = 3'
        exit()

    # This is a multivaraible curve fitting program which attempts to optimize the fitting parameters
    popt, pcov = curve_fit(sum_gaussian, S, I, parguess)

    peak1 = gaussian(S, [popt[2], popt[0], popt[6]]) + gaussian(S, [popt[3], popt[0], popt[7]])
    peak2 = gaussian(S, [popt[4], popt[1], popt[8]]) + gaussian(S, [popt[5], popt[1], popt[9]])

    # Step 3: Evaluate the fit
    if step == 3:
        plt.figure()
        plt.plot(nS, nI, 'b-', label='Data')
        plt.plot(S, sum_gaussian(S, *popt), 'r-', lw=3, label='Final Fit')
        plt.plot(S, peak1, 'm-', lw=3, label='Fit for peak 1')
        plt.plot(S, gaussian(S, [popt[4], popt[1], popt[8]]) + gaussian(S, [popt[5], popt[1], popt[9]]), 'c-', lw=3, label='Fit for peak 2')
        plt.xlim(low, high)
        plt.ylim(0, max(nI) + 2)
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Intensity (counts)')
        plt.legend(loc='best')
        plt.show()
        print 'When you are satisfied with the peak fit, set step = 3'
        print 'else, return to step 2 and choose new fitting parameters'
        exit()

    # Step 4: A summary of the resulting fit
    if step == 4:
        ypeak1 = popt[2] + popt[3] + blfit(popt[0])
        ypeak2 = popt[4] + popt[5] + blfit(popt[1])

        area1 = -np.trapz(S, peak1)
        area2 = -np.trapz(S, peak2)

        savefile = filename.rstrip('txt')
    
        plt.figure()
        plt.plot(S, I + blfit(S), label='data')
        plt.plot(S, sum_gaussian(S, *popt) + blfit(S), 'r-', lw=3, label='fit')

        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Intensity (counts)')
        plt.savefig(savefile + 'png')
        plt.show()

        print 'These are the diagnols of a 10x10 matrix of the covarience of the 10 fitting parameters.'
        print np.diag(pcov)
        print 'The diagonals of this array are representative of the error in each of the 10 fitting parameters.'
        print 'One standard deviation of this fitting error is defined as the square root of this covarience.'
        print 'I do not display fitting errors for area as I am not currently sure how to'
        print 'propogate error through numerical integration.'
        print 'The error reported in the table of results shown below is representative of one standard deviation.'
        print 

        perr = np.sqrt(np.diag(pcov))
 
        pk1err = np.sqrt(perr[2]**2. + perr[3]**2 + 2 * pcov[2][3])
        pk2err = np.sqrt(perr[4]**2. + perr[5]**2 + 2 * pcov[4][5])

        print 'Results'
        print '======='
        print 'Mean = {0:1.1f} $\pm$ {1:1.2f}'.format(popt[0], perr[0])
        print 'Mean = {0:1.1f} $\pm$ {1:1.2f}'.format(popt[1], perr[1])

        print 'Height = {0:1.1f} $\pm$ {1:1.2f}'.format(ypeak1, pk1err)
        print 'Height = {0:1.1f} $\pm$ {1:1.2f}'.format(ypeak2, pk2err)

        print 'Area = {0:1.1f}'.format(area1)
        print 'Area = {0:1.1f}'.format(area2)

        if output:
            savefile = savefile + 'fit'

            f = 'Initial guess parameters:\n'
            f += '=========================\n'
            f += '                      Peak 1, Peak 2\n'
            f += 'Peak center =         {0:1.1f}, {1:1.2f}\n'.format(cntr[0], cntr[1])
            f += 'Amplitude fit 1 =     {0:1.1f}, {1:1.2f}\n'.format(amp1[0], amp1[1])
            f += 'Amplitude fit 2 =     {0:1.1f}, {1:1.2f}\n'.format(amp2[0], amp2[1])
            f += 'Standard dev. fit 1 = {0:1.1f}, {1:1.1f}\n'.format(std1[0], std1[1])
            f += 'Standard dev. fit 2 = {0:1.1f}, {1:1.1f}\n'.format(std2[0], std2[1])

            f += '\nFitted parameters:\n'
            f += '==================\n'
            f += '                      Peak 1, Peak 2\n'
            f += 'Peak center =         {0:1.1f}, {1:1.2f}\n'.format(popt[0], popt[1])
            f += 'Amplitude fit 1 =     {0:1.1f}, {1:1.2f}\n'.format(popt[2], popt[3])
            f += 'Amplitude fit 2 =     {0:1.1f}, {1:1.2f}\n'.format(popt[4], popt[5])
            f += 'Standard dev. fit 1 = {0:1.1f}, {1:1.1f}\n'.format(popt[6], popt[7])
            f += 'Standard dev. fit 2 = {0:1.1f}, {1:1.1f}\n'.format(popt[8], popt[9])

            f += '\nCalculation output:\n'
            f += '======================\n'
            f += 'Mean peak 1 =         {0:1.1f} +/- {1:1.2f}\n'.format(popt[0], perr[0])
            f += 'Mean peak 2 =         {0:1.1f} +/- {1:1.2f}\n'.format(popt[1], perr[1])
            f += 'Height peak 1 =       {0:1.1f} +/- {1:1.2f}\n'.format(ypeak1, pk1err)
            f += 'Height peak 2 =       {0:1.1f} +/- {1:1.2f}\n'.format(ypeak2, pk2err)
            f += 'Area peak 1 =         {0:1.1f}\n'.format(area1)
            f += 'Area peak 2 =         {0:1.1f}\n'.format(area2)

            fl = open(savefile, 'w')
            fl.write(f)
            fl.close()
