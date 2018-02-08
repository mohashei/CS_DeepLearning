import nibabel as nib
import numpy as np

def fft3(data):
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.fft.fft(
           np.fft.fftshift(np.fft.fft(data, axis=0),axes=0), axis=1), axes=1), axis=2), axes=2)

class Image(object):
    def __init__(self,path_to_data):
        self.path_to_data = path_to_data
        self.img = nib.load(self.path_to_data)

    def read_shape(self): 
        return self.img.shape

    def produce_mask(self,usfactor,shape,random=1,oned_us=0,spiral=0):
        mask = np.zeros(np.prod(shape), dtype=bool)
        if random:
            rsize = int( np.floor(np.prod(shape)*usfactor) )
            inds = np.random.choice(np.prod(shape), rsize, replace=False)        
            mask[inds] = True
            mask = mask.reshape(shape)
            return mask
        if oned_us:
            #undersample y axis
            skip = np.floor(shape[1]*usfactor)
            mask[:] = True
            mask = mask.reshape((shape[0],shape[1],shape[2]))
            mask[:,1:skip:,:] = 0
            return mask

    def get_k_data(self, dim = 2, usfactor=0.5):
        data = self.img.get_data()
        #image has 4 dimensions, last dimension is not used
        data = data[:,:,:,0]
        shape = self.read_shape()
        #downside is, full acceleration is not possible
        #upside is: more data (128x more!) + faster training
        if dim == 2:
            rsize = int( np.floor(np.prod(shape[0:2])*usfactor) )
            data_k = np.zeros((rsize,shape[2]), dtype=complex)
            for k in range(shape[2]):
                mask = self.produce_mask(usfactor, shape[0:2])
                data_tmp = np.fft.fft2(data[:,:,k])
                data_k[:,k] = data_tmp[mask] 
        else:
            data_k = fft3(data)
            mask = self.produce_mask(usfactor, shape)
            data_k = data_k[mask]
        #return a vector of data values in k space and the undersampled data
        return data_k
    
    def get_x_data(self):
        x = self.img.get_data()
        return x[:,:,:,0]
