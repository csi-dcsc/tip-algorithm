
##----------------------------------------------------------------------------------------##
##
## TIP Algorithm
## Dean Wilding (C) 2016 GNU GLP
## Control for Scientific Imaging and Instrumentation
## Delft Center for Systems and Control
## Technische Universiteit Delft
##
##----------------------------------------------------------------------------------------##
## Python 2.7 Version
##----------------------------------------------------------------------------------------##

##----------------------------------------------------------------------------------------##
## Dependencies
##----------------------------------------------------------------------------------------##

This version of the TIP algorithm requires the following Python packages to be installed:

- Numpy
- Scipy
- Pillow
- Matplotlib (for plotting only)

##----------------------------------------------------------------------------------------##
## How to use
##----------------------------------------------------------------------------------------##

Take the "example.py" file as a template for running the algorthim, as the "tip.py" file contains the class that has the function "deconvolve", which runs the TIP algorithm.

Essentially, two lines are required:

tip = tip.tip(path, nIter, nFrames, ROI) 

This initializes the class and performs start up operations, such as defining the constraints and processing the images ready for the algorithm.  It requires that one specifies the "path" to the images (in separate files .png, .tif, etc.), the number of iterations, "nIter", the number of frames from the path to use, "nFrames", and additionally there is the option to specify a region of interest, "ROI" as a tuple.

The second important line runs the algorithm producing to outputs, the object "o_tip" and the set of PSFs "h_tip":

o_tip, h_tip = tip.deconvolve() 

##----------------------------------------------------------------------------------------##
## Examples
##----------------------------------------------------------------------------------------##

In the "inputs" folder two images sets are included.  One is a mountain scene and the other is of the Lenna image.  The object for the generation of these sets are included.

##----------------------------------------------------------------------------------------##
## Set Contraints
##----------------------------------------------------------------------------------------##

Depending on the image set that you choose to input, there is some minima a priori information that is required to produce a good result from the dataset.  One should estimate the size of the aperture in Fourier space.  This can be estimated by looking at the image spectra.  The second important parameters is the lower bound on the PSF, this is very helpful in thresholding the algorithm when noise levels are high, however, the higher this value is make the more point-like the PSF will become.  Typical range of 0.0 to 0.2 appear to work on most image sets.