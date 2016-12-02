import tip
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil

##--------------------------------------------------------------------------------------------------------------------##
## TIP Example Code
##--------------------------------------------------------------------------------------------------------------------##

## Directory and Settings
path = "inputs/lenna/"                              # Path to image
nIter = 10                                          # Number of TIP iterations
nFrames = 4                                         # Number of frames to use

ROI = (0,0,512,512)                                 # Region of Interest ( x , y , w , h )
tip = tip.tip(path, nIter, nFrames, ROI)            # Initialize the class

## H Set Constraints
tip.psf_lb = 0.10                                   # PSF Threshold Lower Bound
tip.psf_ub = 1.00                                   # PSF Threshold Upper Bound (do not change)
tip.aperture = 0.45                                 # Estimate of the Aperture Size
tip.padding_ratio = 1.0                             # Ratio of space the images take (1.0 = full frame)

## Load actual object (only for simulations)
tip.object = np.array(pil.open('inputs/lenna/object.tif'))

## Run Deconvolution
o_tip, h_tip = tip.deconvolve()                     # Run algorithm

## Print to Screen (this will only work for 4 images and object files)
plt.subplot(3,4,1)
plt.imshow(tip.i[0],cmap='gray',interpolation='none')
plt.title('Image 1')
plt.axis('off')
plt.subplot(3,4,2)
plt.imshow(tip.i[1],cmap='gray',interpolation='none')
plt.title('Image 2')
plt.axis('off')
plt.subplot(3,4,3)
plt.imshow(tip.i[2],cmap='gray',interpolation='none')
plt.title('Image 3')
plt.axis('off')
plt.subplot(3,4,4)
plt.imshow(tip.i[3],cmap='gray',interpolation='none')
plt.title('Image 4')
plt.axis('off')
plt.subplot(3,4,5)
plt.imshow(h_tip[0,ROI[2]/2-16:ROI[2]/2+16,ROI[3]/2-16:ROI[3]/2+16],cmap='gray',interpolation='none')
plt.title('PSF 1')
plt.axis('off')
plt.subplot(3,4,6)
plt.imshow(h_tip[1,ROI[2]/2-16:ROI[2]/2+16,ROI[3]/2-16:ROI[3]/2+16],cmap='gray',interpolation='none')
plt.title('PSF 2')
plt.axis('off')
plt.subplot(3,4,7)
plt.imshow(h_tip[2,ROI[2]/2-16:ROI[2]/2+16,ROI[3]/2-16:ROI[3]/2+16],cmap='gray',interpolation='none')
plt.title('PSF 3')
plt.axis('off')
plt.subplot(3,4,8)
plt.imshow(h_tip[3,ROI[2]/2-16:ROI[2]/2+16,ROI[3]/2-16:ROI[3]/2+16],cmap='gray',interpolation='none')
plt.title('PSF 4')
plt.axis('off')
plt.subplot(3,4,9)
plt.imshow(np.log(np.abs(tip.ift(o_tip))+tip.eps),cmap='jet',interpolation='none')
plt.title('Spectral Amp.')
plt.axis('off')
plt.subplot(3,4,10)
plt.imshow(np.angle(tip.ift(o_tip)),cmap='jet',interpolation='none')
plt.title('Spectral Phase')
plt.axis('off')
plt.subplot(3,4,11)
plt.imshow(o_tip,cmap='gray',interpolation='none')
plt.title('TIP Object')
plt.axis('off')
plt.subplot(3,4,12)
plt.imshow(tip.object,cmap='gray',interpolation='none')
plt.title('Real Object')
plt.axis('off')
plt.show()
plt.close()

## Print Results to File
for i in range(0,nFrames):
    pil.fromarray(h_tip[i]).save('outputs/mountain/p_{}.tif'.format(i))
pil.fromarray(o_tip).save('outputs/mountain/obj_tip.tif')
