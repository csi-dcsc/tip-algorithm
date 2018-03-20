##----------------------------------------------------------------------------------------------------------------------
#  TIP Algorithm 31-07-2017
# (c) Dean Wilding, Oleg Soloviev 2016-17
# Distributed under the GNU Lesser General Public License v3.0
#
#   tip.py contains the Python class library for using the TIP algorithm.
#   Usage is shown below.
##----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import tip

# tipclass = tip.tip(path,nIter,nFrames,compression,interpolation,psfSize,saveBasis)
#
#   ------
#   Inputs
#   ------
#   path [str] - the relative or absolute path to a folder containing image files (*.png,*.jpg, *.tif)
#   nIter [int] - number of iterations to run the algorithn (default is 10)
#   nFrames [int] - number of frames from the directory to load in alphanumeric order
#   compression [int] - the relative size of the subregion to use to determine PSFs (i.e. 2 = quarter the image area)
#   interpolation [float] - change the relative size of the pixels in the images
#   psfSize [int] - the a priori size of the PSF
#   saveBasis [bool] - whether you wish to save the basis functions
#   ------
#   Returns
#   ------
#   o [ndarray float] - the output object scaled to requested size
#   h [ndarray float] - the array containing the calculated PSFs (nFrames x Image X x Image Y)

# Initialize the class
tipclass = tip.tip('inputs/mountain/',10,4,1,1.0,15,False)

# Run the deconvolution
o, h = tipclass.deconvolve()

# Display the results
plt.imshow(o,cmap='gray',interpolation='none')
plt.title('TIP Object')
plt.show()