import numpy as np
import scipy.fftpack
import os, os.path
import PIL.Image as pil
import scipy.misc
import scipy.ndimage
import scipy.signal
import time

class tip():

    def __init__(self, path, nIter, nFrames, compression, scale, psf_size, save_basis):

        # Load the files from the directory
        files = os.listdir(path)
        test = np.array(pil.open(path + files[0]))
        N, M = test.shape

        print "Starting TIP Algorithm --- loading from directory '{}' ".format(path)
        print "PSF Support Size = {}".format(psf_size)
        print "Number of Frames = {}".format(nFrames)
        print "Number of Frames = {}".format(nIter)
        print "Processing Compression = {}".format(compression)
        print "Interpolation = {}".format(scale)
        if save_basis == True:
            if os.path.isdir('./base/') == False:
                os.mkdir('./base/')
            print "Basis files will be saved to ./base/"

        # Expected image dimensions
        self.N = int(N * scale)          # Grid Size
        self.M = int(M * scale)          # Grid Size
        self.D = nFrames                 # Number of Frames to load
        self.compression = compression   # The reduction in the size used for the calculation
        self.scale = scale               # Down/Up-scaling for the deconvolution
        self.nIter = nIter               # Number of iterations desired
        self.psf_size = psf_size         # a priori PSF size
        self.save_basis = save_basis     # Boolean for saving the basis

        if self.N % 2 != 0:
            self.N += 1
        if self.M % 2 != 0:
            self.M += 1

        # Load the images from the directory into memory / scale
        self.i = np.zeros((self.D,self.N,self.M))
        for d in range(0,self.D):
            self.i[d] = scipy.misc.imresize(np.array(pil.open(path+files[d])).astype('float'),(self.N,self.M))

        print "Found {} images ({}x{})".format(self.D,self.N,self.M)

        # Calculate the new array size
        self.N = int(float(self.N) / self.compression)
        self.M = int(float(self.M) / self.compression)

        if self.N % 2 != 0:
            self.N += 1
        if self.M % 2 != 0:
            self.M += 1

        # Check if it has already been created
        if os.path.isfile('base/A_{}x{}_{}.npy'.format(self.N, self.M, self.psf_size)):
            print "Basis file exists.  Loading from file."
            self.A = np.load('base/A_{}x{}_{}.npy'.format(self.N, self.M, self.psf_size))
            self.Ainv = np.load('base/Ainv_{}x{}_{}.npy'.format(self.N, self.M, self.psf_size))
            print "Complete."
        else:
            print "Basis file does not exists.  Creating file."
            # Create aperture grid
            x = np.linspace(-1.0, 1.0, self.N)
            y = np.linspace(-1.0, 1.0, self.M)
            X, Y = np.meshgrid(y, x)
            RHO = np.sqrt(X ** 2 + Y ** 2)

            mask = (RHO < (float(self.psf_size) / (self.N))).astype('uint8')
            self.psf_points = int(np.sum(mask))

            self.A = np.zeros((self.M * self.N, self.psf_points), dtype='complex')
            for i in range(0, self.psf_points):
                f = np.zeros((self.N, self.M))
                px, py = np.unravel_index(np.argmax(mask), mask.shape)
                f[px, py] = 1.0
                mask[px, py] = 0.0
                F = self.ift(f)
                self.A[:, i] = F.reshape(self.M * self.N)

            self.Ainv = np.linalg.pinv(self.A)

            if self.save_basis == True:
                print 'Saving file. '
                np.save('base/A_{}x{}_{}'.format(self.N, self.M, self.psf_size), self.A)
                np.save('base/Ainv_{}x{}_{}'.format(self.N, self.M, self.psf_size), self.Ainv)

            print "Complete."

        print "Start up complete."

        return

    def ft(self, a): # Fourier Tranform Definition with scaling and shifting
        A = scipy.fftpack.fftshift(scipy.fftpack.fft2(scipy.fftpack.fftshift(a)))
        r = np.sum(np.abs(a) ** 2) / np.sum(np.abs(A) ** 2)
        return r * A

    def ift(self, a): # Inverse Fourier Tranform Definitions with scaling and shifting
        A = scipy.fftpack.ifftshift(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(a)))
        r = np.sum(np.abs(a) ** 2) / np.sum(np.abs(A) ** 2)
        return r * A

    def ls(self, H, F): # Least-squares solution
        D, N, M = H.shape
        A = np.zeros((N, M), dtype='complex128')
        B = np.zeros((N, M), dtype='complex128')
        for d in range(0, D):
            A += H[d].conj() * F[d]
            B += np.abs(H[d]) ** 2
        B = B + 1e-15
        G = self.divide(A, B)
        return G

    def divide(self, A, B):  # Safe division
        if type(A) == float:
            out = np.zeros(B.shape, dtype='complex128')
            out[np.abs(B) > np.max(np.abs(B))*1e-11] = A[np.abs(B) > np.max(np.abs(B))*1e-11] / B[np.abs(B) > np.max(np.abs(B))*1e-11]
        else:
            out = np.zeros(A.shape, dtype='complex128')
            out[np.abs(B) > np.max(np.abs(B))*1e-11] = A[np.abs(B) > np.max(np.abs(B))*1e-11] / B[np.abs(B) > np.max(np.abs(B))*1e-11]
        return out

    def deconvolve(self): # Main deconvolution command

        # Record starting time
        t1 = time.time()

        # Create a working copy cropped to the compression size
        D,N,M = self.i.shape
        i = np.copy(self.i[:,N/2-self.N/2:N/2+self.N/2,M/2-self.M/2:N/2+self.M/2])

        # Compute the starting OTFs
        # Generate starting OTFs - blind
        H = np.ones((self.D, self.N, self.M), dtype='complex')
        I = np.zeros((self.D, self.N, self.M), dtype='complex')
        for d in range(0, self.D):
            I[d] = self.ift(i[d])

        # Main algorithm loop
        for k in range(0, self.nIter):
            # Calculate object spectrum via least-squares (P_1)
            O = self.ls(H, I)
            # Renormalise the object
            o = np.real(self.ft(O))
            o[o<0] = 0
            o /= np.sum(o)
            O = self.ift(o)
            for d in range(0, self.D):
                # Tangential Projection (P_3)
                H[d] = self.divide(I[d], O)
                # Project to OTF set (P_4)
                alpha = np.real(self.Ainv.dot(H[d].reshape(self.N*self.M)))
                alpha[alpha<0] = 0
                # Renormalise the PSF
                alpha /= np.sum(alpha)
                H[d] = self.A.dot(alpha).reshape(self.N,self.M)
            print "Iteration Number:",k+1,"/",self.nIter

        # Make PSFs and Full Size Image Spectrum
        h = np.zeros((self.D, N, M), dtype='float')
        H2 = np.zeros((self.D, N, M), dtype='complex')
        I2 = np.zeros((self.D, N, M), dtype='complex')
        for d in range(0, self.D):
            h[d,N/2-self.N/2:N/2+self.N/2,M/2-self.M/2:N/2+self.M/2] = np.real(self.ft(H[d]))
            h[d,h[d]<0] = 0
            H2[d] = self.ift(h[d])
            I2[d] = self.ift(self.i[d])

        # Create final output
        O = self.ls(H2,I2)
        o = np.real(self.ft(O))
        o[o<0] = 0

        # Record finishing time
        t2 = time.time()
        print "TIP Complete. Time elapsed: {}".format(t2-t1),"seconds."

        return o, h

