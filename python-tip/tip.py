import numpy as np
import scipy.fftpack
import os
import PIL.Image as pil
import scipy.misc
import scipy.ndimage
import scipy.signal

class tip():

    def __init__(self, path, nIter, nFrames, ROI):

        # Load the files from the directory
        files = os.listdir(path)
        test = np.array(pil.open(path + files[0]))
        N, M = test.shape

        # Expected image dimensions
        self.N = N
        self.M = M
        self.D = nFrames
        self.ROI = ROI  # ( x , y , w , h )

        # Number of iterations desired
        self.nIter = nIter

        self.i = np.zeros((self.D,N,M))
        for d in range(0,self.D):
            self.i[d] = np.array(pil.open(path+files[d])).astype('float')

        # Apply ROI
        self.i = self.i[:,ROI[0]:ROI[0]+ROI[2],ROI[1]:ROI[1]+ROI[3]]
        self.N = ROI[2]
        self.M = ROI[3]

        # Imaging tiling if the sizes are not the same (helps with F.T. issues)
        if self.N > self.M:
            self.ratio = self.N / self.M
            self.N_o = np.copy(self.N)
            self.M_o = np.copy(self.M)
            i = np.zeros((self.D, N, N))
            for d in range(0, self.D):
                for k in range(0,int(self.ratio)):
                    i[d,:,k*self.M:(k+1)*self.M] = self.i[d]
            self.D, self.N, self.M = i.shape
            self.i = i
            self.cut = True
        elif self.M > self.N:
            self.ratio = self.M / self.N
            self.N_o = np.copy(self.N)
            self.M_o = np.copy(self.M)
            i = np.zeros((self.D, M, M))
            for d in range(0, self.D):
                for k in range(0, int(self.ratio)):
                    i[d,k * self.N:(k + 1) * self.N,:] = self.i[d]
            self.D, self.N, self.M = i.shape
            self.i = i
            self.cut = True
        else:
            self.cut = False

        # Normalization
        for d in range(0, self.D):
            self.i[d] = self.i[d] / np.sum(self.i[d]) * self.N*self.M

        # Space
        px = np.linspace(-1.0, 1.0, self.N)
        py = np.linspace(-1.0 * float(self.M) / float(self.N), 1.0 * float(self.M) / float(self.N), self.M)
        X, Y = np.meshgrid(py, px)
        self.RHO = np.sqrt(X ** 2 + Y ** 2)
        del px, py, X, Y


        ## Default Constraints
        # Bounds on the PSF values
        self.psf_lb = 0.20
        self.psf_ub = 1.00

        # Aperture Estimate
        self.aperture = 0.9

        # Spectral filters
        self.filter = np.ones((self.N, self.M)) * (self.RHO < 2*self.aperture)
        self.filter_psf = np.ones((self.N, self.M)) * (self.RHO < 2*self.aperture)

        # Apodization of Images
        self.apodize = False
        self.apod_distance = 0.75
        self.apod_width = 0.05

        # Padding
        self.padding_ratio = 1.00

        # Printing
        self.printing = False

        # Divisor Limit
        self.eps = 1e-15

        return

    def ft(self, a): # Fourier Tranform Definition with scaling and shifting
        A = scipy.fftpack.fftshift(scipy.fftpack.fft2(scipy.fftpack.fftshift(a)))
        r = np.sum(np.abs(a) ** 2) / np.sum(np.abs(A) ** 2)
        return r * A

    def ift(self, a): # Inverse Fourier Tranform Definitions with scaling and shifting
        A = scipy.fftpack.ifftshift(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(a)))
        r = np.sum(np.abs(a) ** 2) / np.sum(np.abs(A) ** 2)
        return r * A

    def ls(self, H, F, eps=0.0): # Least-squares solution
        D, N, M = H.shape
        A = np.zeros((N, M), dtype='complex128')
        B = np.zeros((N, M), dtype='complex128')
        for d in range(0, D):
            A += H[d].conj() * F[d]
            B += np.abs(H[d]) ** 2
        B = B + eps
        G = self.divide(A, B)
        return G

    def realize(self, F, filter, lb, ub): # Projection Operator
        f = np.real(self.ft(F * filter))
        f = f / f.max()
        f = (f * (f >= lb) * (f <= ub))
        F = self.ift(f)
        return F

    def divide(self, A, B):  # Safe division
        if type(A) == float:
            out = np.zeros(B.shape, dtype='complex128')
            out[B != 0] = A / B[B != 0]
        else:
            out = np.zeros(A.shape, dtype='complex128')
            out[B != 0] = A[B != 0] / B[B != 0]
        return out

    def remake_filters(self):
        self.filter = np.ones((self.N, self.M)) * (self.RHO < 2*self.aperture)
        self.filter_psf = np.ones((self.N, self.M)) * (self.RHO < 2*self.aperture)

    def deconvolve(self):

        i_copy = np.copy(self.i)

        # Add the padding to the image arrays
        N = int(self.N / self.padding_ratio)
        M = int(self.M / self.padding_ratio)

        i = np.zeros((self.D,N,M))
        filter = np.zeros((N, M))
        filter_psf = np.zeros((N, M))

        self.remake_filters()

        for d in range(0,self.D):
            i[d,N/2-self.N/2:N/2+self.N/2,M/2-self.M/2:M/2+self.M/2] = self.i[d]
            filter[N/2-self.N/2:N/2+self.N/2,M/2-self.M/2:M/2+self.M/2] = self.filter
            filter_psf[N/2-self.N/2:N/2+self.N/2,M/2-self.M/2:M/2+self.M/2] = self.filter_psf
        self.i = i

        # Add the padding to the filter arrays
        self.filter = scipy.misc.imresize(self.filter.astype('float'),(N,M),mode='F').astype('float')
        self.filter_psf = scipy.misc.imresize(self.filter_psf.astype('float'),(N,M),mode='F').astype('float')
        self.RHO = scipy.misc.imresize(self.RHO,(N,M),mode='F').astype('float')

        # Retain the inputs for the final deconvolution
        i = np.copy(self.i)

        if self.apodize == True: # For images with bright borders apodization can help
            apod = np.ones((N, M)) * (self.RHO < self.apod_distance)
            apod += np.exp( - (self.RHO-self.apod_distance)** 2 / (2 * self.apod_width ** 2)) * (self.RHO > self.apod_distance)
            for d in range(0, self.D):
                i[d] = i[d] * apod

        # Compute the OTFs and apply the filters
        # Generate starting OTFs - blind
        H = np.ones((self.D, N, M), dtype='complex')
        I = np.zeros((self.D, N, M), dtype='complex')
        for d in range(0, self.D):
            I[d] = self.ift(i[d])
            I[d] = I[d] * self.filter
            H[d] = H[d] * self.filter_psf

        # Initial O
        O = np.ones((N, M), dtype='complex')

        # Main algorithm loop
        for k in range(0, self.nIter):
            O = self.ls(H, I, self.eps)
            for d in range(0, self.D):
                H[d] = self.divide(I[d]*O.conj(),np.abs(O)**2)
                H[d] = self.realize(H[d],self.filter_psf,self.psf_lb,self.psf_ub)

        # Final step: final deconvolution
        O = np.zeros((N,M),dtype='complex')
        O[self.RHO<2*self.aperture] = self.ls(H, I, self.eps)[self.RHO<2*self.aperture]
        o = np.real(self.ft(O))
        O = self.ift(o)
        o = np.real(self.ft(O))
        o[o<0] = 0
        o = o / o.max()

        # Make PSFs:
        h = np.ones((self.D, N, M), dtype='float')
        for d in range(0, self.D):
            h[d] = np.real(self.ft(H[d]))
            h[d] = h[d] * (h[d] > 0.0)

        # If images are padded, it will cut the middle out
        out = o[N/2-self.N/2:N/2+self.N/2,M/2-self.M/2:M/2+self.M/2].astype('float')
        if self.cut == True:
            out = out[0:self.N_o,0:self.M_o].astype('float')

        self.i = np.copy(i_copy)

        return out, h