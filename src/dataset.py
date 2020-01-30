import os
import cv2

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from PIL import Image
import torch
import torch.utils.data as utils
import matplotlib.pyplot as plt
from scipy import ndimage
# from src.transformation import autoaugment
class FAZ_Preprocess:
    def __init__(self,image_name, sigma, spacing, tau):
        super(FAZ_Preprocess, self).__init__()
        image = Image.open(image_name).convert("RGB")
        image = np.array(image)
        image = 255-image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.size = image.shape
        thr = np.percentile(image[(image > 0)], 1)*0.9
        image[(image <= thr)] = thr
        image = image - np.min(image)
        image = image / np.max(image)
        
        self.image =image
        self.sigma = sigma
        self.spacing = spacing
        self.tau = tau
    
    def gaussian(self, image, sigma):
        siz = sigma * 6
        temp = round(siz/self.spacing/2)
        #processing x-axis
        x = [i for i in range(-temp, temp+1)]
        x = np.array(x)
        H = np.exp(-(x**2 / (2*((sigma/self.spacing)**2) )))
        H = H / np.sum(H)
        Hx = H.reshape(len(H),1)
        I = ndimage.filters.convolve(image, Hx, mode='nearest')

        #processing y-axis
        temp = round(siz/self.spacing/2)
        x = [i for i in range(-temp, temp+1)]
        x = np.array(x)
        H = np.exp(-(x**2 / (2*((sigma/self.spacing)**2) )))
        H = H / np.sum(H[:])
        Hy = H.reshape(1,len(H))
        I = ndimage.filters.convolve(I, Hy, mode= 'nearest')
        return I
    
    def gradient2(self, F, option):
        k = self.size[0]
        l = self.size[1]
        D = np.zeros(F.shape)
        if option == "x":
            D[0,:] = F[1,:] - F[0,:]
            D[k-1, :] = F[k-1,:]-F[k-2,:]

            #take center differences on interior points
            D[1:k-2, :] = (F[2:k-1,:]-F[0:k-3,:])/2
        else:
            D[:,0] = F[:,1]-F[:,0]
            D[:,l-1] = F[:,l-1]-F[:, l-2]
            D[:,1:l-2] = (F[:,2:l-1] - F[:,0:l-3])/2
        return D
    
    def Hessian2d(self, image, sigma):
        image = self.gaussian(image, sigma)
    #     image = ndimage.gaussian_filter(image, sigma, mode = 'nearest')
        Dy = self.gradient2(image,"y") 
        Dyy = self.gradient2(Dy, "y")

        Dx = self.gradient2(image, "x")
        Dxx = self.gradient2(Dx, "x")
        Dxy = self.gradient2(Dx, 'y')
        return Dxx, Dyy, Dxy
    
    def eigvalOfhessian2d(self, Dxx, Dyy, Dxy):
        tmp = np.sqrt((Dxx-Dyy)**2 + 4*(Dxy**2))
        #compute eigenvectors of J, v1 and v2
        mu1 = 0.5 * (Dxx+Dyy + tmp)
        mu2 = 0.5 * (Dxx+Dyy - tmp)
        #Sort eigen values by absolute value abs(Lambda1) < abs(Lambda2)
        indices = (np.absolute(mu1) > np.absolute(mu2))
        Lambda1 = mu1
        Lambda1[indices] = mu2[indices]
        
        Lambda2 = mu2
        Lambda2[indices] = mu1[indices]
        return Lambda1, Lambda2
    
    def imageEigenvalues(self, I, sigma):
        hxx, hyy, hxy = self.Hessian2d(I, sigma)
        # hxx, hyy, hxy = self.Hessian2d(I, sigma)
        c = sigma ** 2 
        hxx = -c* hxx
        # hxx = hxx.flatten()
        hyy = -c* hyy
        # hyy = hyy.flatten()
        hxy = -c* hxy
        # hxy = hxy.flatten()

        # # reduce computation by computing vesselness only where needed
        B1 = -(hxx+hyy)
        B2 = hxx * hyy - hxy **2
        T = np.ones(B1.shape)
        T[(B1 < 0)]= 0
        T[(B1==0) & (B2==0)]= 0
        T = T.flatten()
        indeces = np.where(T == 1)[0]
        hxx = hxx.flatten()
        hyy = hyy.flatten()
        hxy = hxy.flatten()
        hxx = hxx[indeces]
        hyy = hyy[indeces]
        hxy = hxy[indeces]
    #     lambda1i, lambda2i = hessian_matrix_eigvals([hxx, hyy, hxy])
        lambda1i, lambda2i = self.eigvalOfhessian2d(hxx, hyy, hxy)
        lambda1 = np.zeros(self.size[0] * self.size[1],)
        lambda2 = np.zeros(self.size[0] * self.size[1],)

        lambda1[indeces] = lambda1i
        lambda2[indeces] = lambda2i

        #removing noise
        lambda1[(np.isinf(lambda1))] = 0
        lambda2[(np.isinf(lambda2))] = 0

        lambda1[(np.absolute(lambda1) < 1e-4)]=0
        lambda1 = lambda1.reshape(self.size)
        
        lambda2[(np.absolute(lambda2) < 1e-4)]=0
        lambda2 = lambda2.reshape(self.size)
        return lambda1, lambda2
    
    def vesselness2d(self):
        for j in range(len(self.sigma)):
            lambda1, lambda2 = self.imageEigenvalues(self.image, self.sigma[j])
            lambda3 = lambda2.copy()
            new_tau = self.tau * np.min(lambda3) 
            lambda3[(lambda3 <0) & (lambda3>= new_tau)] = new_tau
            different = lambda3 - lambda2
            response = ((np.absolute(lambda2)**2) * np.absolute(different)) *27 / ((2*np.absolute(lambda2)+np.absolute(different))**3)
            response[(lambda2 < lambda3/2 )] = 1
            response[(lambda2 >= 0)] = 0

            response[np.where(np.isinf(response))[0]] = 0
            if j == 0 :
                vesselness = response
            else:
                vesselness = np.maximum(vesselness, response)
    #     vesselness = vesselness / np.max(vesselness)
        vesselness[(vesselness < 1e-2)] = 0
#         vesselness = vesselness.reshape(self.size)
        return vesselness
    
######################################################################################    
class RetinalDataset(utils.Dataset):
    def __init__(self, image_folder, size, image_transform, phase = "valid"):
        self.phase = phase
        self.transform = image_transform
        self.size = size
        #source dir
        self.raw = image_folder
        self.mask = image_folder.replace("raw", "mask")
        
        #list of raw image and ground truth
        self.image = os.listdir(image_folder)
        
    def __getitem__(self, index):
        image_name = self.image[index]
        if self.phase == "valid":
            mask_name = image_name
        else:
            mask_name = image_name.replace("tif", "png")
        
        image = FAZ_Preprocess(os.path.join(self.raw, image_name),[0.5,1, 1.5, 2, 2.5],1, 2)
        image = image.vesselness2d()
        image = Image.fromarray(image.astype(np.float32)*255).convert("RGB")
        mask = Image.open(os.path.join(self.mask, mask_name))
        image, mask = self.transform(image, mask, self.size)

        return image, mask
    def __len__(self):
        return len(self.image)
