# Artan Salihu (artan.salihu@tuwien.ac.at)

import torch
import torch.nn as nn
from einops import rearrange # No need for rearrange for transformations used. But at the bottom of this file (> lines 355), there are some more that might need it.

''' Example on how to try ChannelTransformationModule only: 
# # Create the ChannelTransformationModule
# from ChannelTransformationModule import ChannelTransformationModule 
# aug_module = ChannelTransformationModule('first') 
# H = torch.randn(1, 3, 64, 32) 
# H_out = augmentation_module(H)
# print(H_out)
# print(H_out.size())
'''
class ChannelTransformationModule(nn.Module):
    def __init__(self, view_type, realMax = 0.5, imagMax = 0.5, absMax = 0.5):
        super().__init__()
        self.view_type = view_type
        self.realMax = realMax
        self.imagMax = imagMax
        self.absMax = absMax

        self.transform = self._get_channel_transforms()
        
    def _get_channel_transforms(self):
        """
        Returns a composite function of augmentations for the specified view or type.
        Args:
            view_type (str): The type of transformation function that should apply based on the view selection function. 
                                    Three view selection functions during SSL: 'first', 'second', and 'others' (N_s).
                                    Two view selection functions during training: 'classifier', 'regression'.
            realMax (float): Maximum value of the real part of the channel (the whole dataset).
            imagMax (float): Maximum value of the imaginary part of the channel (the whole dataset).
            absMax (float): Maximum value of the absolute part of the channel (the whole dataset).
        """
        if self.view_type == 'first':
            return nn.Sequential(
                RandomSubcarrierSelection(p=1.0, gamma=0.9),
                RssResize((64,36)),
                RandomSubcarrierFlipping(p=0.4),
                RandomGainOffset(p=0.2, xi_rgo=0.1),
                #RandomFadingComponent(p=0.0, sigma_rfc=[0.5,0.6]),
                #RandomChangeSign(p=0.0),
                NormalizeMaxValue(p=1.0, realMax=self.realMax, imagMax=self.imagMax, absMax=self.absMax),
                GaussianNoise(p=0.2, mu=0, sigma_Q_var=1e-7)
            )

        elif self.view_type == 'second':
            return nn.Sequential(
                RandomSubcarrierSelection(p=1.0, gamma=0.8),
                RssResize((64,36)),
                RandomSubcarrierFlipping(p=0.4),
                RandomGainOffset(p=0.8, xi_rgo=0.1),
                RandomFadingComponent(p=0.1, sigma_rfc=[0.5,0.6]),
                RandomChangeSign(p=0.2),
                NormalizeMaxValue(p=1.0, realMax=self.realMax, imagMax=self.imagMax, absMax=self.absMax),
                GaussianNoise(p=0.2, mu=0, sigma_Q_var=1e-7)
            )
        elif self.view_type == 'others':
            return nn.Sequential(
                RandomSubcarrierSelection(p=1.0, gamma=0.1),
                RssResize((64,16)),
                #RandomSubcarrierFlipping(p=0.0),
                #RandomGainOffset(p=0.0, xi_rgo=0.1),
                #RandomFadingComponent(p=0.0, sigma_rfc=[0.5,0.6]),
                #RandomChangeSign(p=0.0),
                NormalizeMaxValue(p=1.0, realMax=self.realMax, imagMax=self.imagMax, absMax=self.absMax),
                #GaussianNoise(p=0.0, mu=0, sigma_Q_var=1e-7)
            )
        elif self.view_type == 'classifier':
            return nn.Sequential(
                CenterSubcarriers((64,32)),
                NormalizeMaxValue(p=1.0, realMax=self.realMax, imagMax=self.imagMax, absMax=self.absMax),
            )
        elif self.view_type == 'regressor':
            return nn.Sequential(
                CenterSubcarriers((64,32)),
                NormalizeMaxValue(p=1.0, realMax=self.realMax, imagMax=self.imagMax, absMax=self.absMax),
            )            
        else:
            raise ValueError(f"Type of the view selection function is not implemented yet: {self.view_type}")
        
    def forward(self, x):
        return self.transform(x)

### 1. Random Subcarrier Selection (RSS) ####
class RandomSubcarrierSelection(nn.Module):
    def __init__(self, p=0.5, gamma=0.9):
        """
        Random Subcarrier Selection (RSS): Select a group of subcarriers.
        Args:
        - p: float, optional (default=0.5)
            Probability for RSS augmentation.
        - gamma: float, optional (default=0.9)
            The number/fraction of N_c to select from the tensor.
        """

        super(RandomSubcarrierSelection, self).__init__()

        self.p = p
        self.gamma = gamma

    def forward(self, x):
        """
        Apply Random Subcarrier Selection (RSS) augmentation.
        
        Args:
        - x: torch.Tensor
            Shape (B, 3, N_r, N_c).
            
        Returns:
        - torch.Tensor
        """

        if torch.rand(1) < self.p:
            # Get the 90% of N_c, floored down to the nearest integer
            gamma = int(self.gamma * x.size(3))

            # Randomly select index of N_c for RSS aug. at
            index_sub = int((x.size(3) - gamma) * torch.rand(1).item())
            
            # Select the tensor based on 3rd dim. (N_c)
            x = x[:, :, :, index_sub:(index_sub + gamma)]
            return x

        else:
            return x


### 2. RSS Resizing ####
class RssResize(nn.Module):
    """
    RssResize: Resizing the input channel ("should" be similar to Pytorch transforms.Resize()).

    Args:
    - size: the pre-defined size/shape of the output (h, w). 
    - type_resize (str): resizing can be 'nearest', 'bilinear', 'bicubic', 'area'. 

    Shape:
    - Input: [B, channel_parts, N_r, N_c]
    - Output: [B, channel_parts, N_r, N_c]
    """

    def __init__(self, size, type_resize='bilinear'):

        super(RssResize, self).__init__()

        self.size = size
        self.type_resize = type_resize

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, mode=self.type_resize)

### 3. Random Subcarrier Flipping (RFC) ###
class RandomSubcarrierFlipping(nn.Module):
    """Random Subcarrier Flipping (RSF): Module to do flipping along the subcarriers.

    Args:
        p (float): Probability of occurance (defaults to 0.5).

    """

    def __init__(self, p=0.5):
        super(RandomSubcarrierFlipping, self).__init__()
        self.p = p
        
    def forward(self, x):
        """Forward pass for tensor x.

        Args:
            x (torch.Tensor): Input channel (batch of H).

        Returns:
            torch.Tensor

        """
        if torch.rand(1) < self.p:

            # Flip along the subcarriers
            return torch.flip(x, [-1])

        else:
            # No flipping
            return x


### 4. Random Gain Offset (RGO) ###
class RandomGainOffset(nn.Module):
    """Random Gain Offset (RGO): Randomly scale channel coeffiecients by an offset.

    Args:
        p (float): Probability of applying it (Default: 0.5)
        xi_rgo (float): xi_rgo factor. The xi_rgo scale 
        factor is sampled uniformly from [1-xi_rgo, 1+xi_rgo]. Default value is 0.1
    """

    def __init__(self, p=0.5, xi_rgo=0.1):

        super(RandomGainOffset, self).__init__()

        self.p = p

        self.xi_rgo = xi_rgo

    def forward(self, x):

        if torch.rand(1) < self.p:

            # Sample xi_rgo from a uniform distribution
            xi = 1 + (torch.rand(1) - 0.5) * 2 * self.xi_rgo
            #print(xi)

            # Scale by xi_rgo of the input tensor
            x = x * xi
        return x

### 5. Random Change Sign (CS) ###

class RandomChangeSign(torch.nn.Module):
    def __init__(self, p=1.0):
        """ Random Change Sign (RCS): Mirror the Signal (LP to HP) to all channel coefficients.    
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super(RandomChangeSign, self).__init__()
        self.p = p
        self.a = -1

    def forward(self, x):
        """
        Args:
            sample (Tensor): Tensor to change the sign of the values.

        Returns:
            Tensor: Changed signs randomly.
        """
        if torch.rand(1) < self.p:
          return torch.mul(x, self.a)
        return x


# 6. Random Fading Component
class RandomFadingComponent(torch.nn.Module):
    def __init__(self, p=1.0, sigma_rfc=[0.5,0.9]):
        """ Random Fading Component (RFC): Add a random component for Reileigh distribution.     

        Args:
          p (float): probability of transformation to happen. Default value is 0.5
          sigma_rfc (float): lower and upper bounds of scale values for Rayleigh Fading.
            sigma_rfc = 0.5**2
            a = a1/(sigma_rfc)*np.exp(-(a1**2)/(2*sigma_rfc))
        """

        super(RandomFadingComponent,self).__init__()
        
        self.p = p
        self.sigma_rfc = torch.FloatTensor(1).uniform_(sigma_rfc[0],sigma_rfc[1])**2

    def forward(self, x):
        """
        Args:
            sample (Tensor): Tensor channel sample subcarriers.

        Returns:
            Tensor: Subset of subcaerriers.
        """
        if torch.rand(1) <= self.p:
          C = x.index_select(-3,torch.tensor([2],dtype=torch.int32))

          C = C/(self.sigma_rfc)*torch.exp(-(C**2)/(2*self.sigma_rfc))

          x = torch.cat([x.index_select(-3,torch.tensor([0,1],dtype=torch.int32)),C], dim=-3)

          return x

        else:
            # No RFC component
            return x

    def string(self):
        """
        Nothing!
        """
        return f'Default'

### 7. Post Normalization 
class NormalizeMaxValue(torch.nn.Module):
    def __init__(self, p=1.0, realMax=0.1, imagMax=0.1, absMax=0.1):
        """ NormalizeMaxValue: Normalize channel by max values in re, imag and abs parts.
        Args:
          p (float): probability of transformation to happen. Default value is 1.0
          realMax (float): Use statistics of you data. Maximum value in the real part of the channel.
          imagMax (float): Use statistics of you data. Maximum value in the imag part of the channel.
          absMax (float): Use statistics of you data. Maximum value in the abs part of the channel.
        """
        super(NormalizeMaxValue, self).__init__()

        self.p = p
        self.realMax = torch.tensor([realMax])
        self.imagMax = torch.tensor([imagMax]) 
        self.absMax = torch.tensor([absMax]) 

    def forward(self, x):
        """
        Args:
            sample (Tensor): Tensor sample

        Returns:
            Tensor: Subset of subcaerriers.
        """
        if torch.rand(1) <= self.p:
          a = x.index_select(-3,torch.tensor([0],dtype=torch.int32))/self.realMax
          b = x.index_select(-3,torch.tensor([1],dtype=torch.int32))/self.imagMax
          c = x.index_select(-3,torch.tensor([2],dtype=torch.int32))/self.absMax

          x = torch.cat([a,b,c], dim=-3)

          return x
        return x

### 8. Add Gaussian Noise
class GaussianNoise(nn.Module):
    """Add Gaussian noise to H

    Args:
        p (float): probability of transformation to happen. Default value is 0.0
        mu (float, optional): Mean of the Gaussian distribution (default = 0)
        sigma_Q_var (float, optional): var of the Gaussian distribution (default = 1)
    """
    def __init__(self, p=0.0, mu=0, sigma_Q_var=1):
        super(GaussianNoise, self).__init__()
        self.p = p
        self.mu = mu
        self.sigma_Q_var = sigma_Q_var

    def forward(self, x):
        if torch.rand(1) < self.p:
            noise = torch.randn_like(x) * self.sigma_Q_var + self.mu
            return x + noise
        else:
            return x

### 9. Get center-subcarriers.
class CenterSubcarriers(nn.Module):
    """Slice the channel in the "center".
    """
    def __init__(self, size):
        super(CenterSubcarriers, self).__init__()
        self.size = size

    def forward(self, x):
        B, C, N_r, N_c = x.shape
        index_N_r = (N_r - self.size[0]) // 2
        index_N_c = (N_c - self.size[1]) // 2
        return x[:, :, index_N_r:(index_N_r + self.size[0]), index_N_c:(index_N_c + self.size[1])]

# Numpy to Tensor

class NumpyToTensor(torch.nn.Module):
    def __init__(self, p=1.0):
        """ From Numpy to Tensor
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) <= self.p:
          x = torch.from_numpy(x)
          x = rearrange(x,'a s c -> c a s')
          return x
        return x     


# Tensor to Numpy
class TensorToNumpy(torch.nn.Module):
    def __init__(self, p=1.0):
        """ From Numpy to Tensor
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) <= self.p:
          x = rearrange(x, 'c a s -> a s c')
          x = x.numpy()
          return x
        return x


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##################### And, below a bunch of other simple augmentations that we have investigated at some point in time. ##################
##########################################################################################################################################
####################################### Note: Double check before using any of the below implementations. ###############################
##########################################################################################################################################

class Complemenraty_Filter(torch.nn.Module):
    def __init__(self, p=0.5):
        """ Complemenraty the Signal (1-H(z)). It changes only the magnitude part of the signal.    
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p
        self.complimentary = torch.tensor([1.])

    def forward(self, x):
        if torch.rand(1) < self.p:
          C = self.complimentary.unsqueeze(-2) - x[:,2,...].unsqueeze(-3)
          return torch.cat([x[:,0:2,...],C], dim=-3) #C
        return x   

class Sine_Filter(torch.nn.Module):
    def __init__(self, p=0.5):
        """ Sine the Signal (sin(H(z)). It changes only the magnitude part of the signal.    
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
          C = torch.sin(x[:,2,...]).unsqueeze(-3)
          return torch.cat([x[:,0:2,...],C], dim=-3) #C
        return x 


class Cosine_Filter(torch.nn.Module):
    def __init__(self, p=0.5):
        """ Cosine the Signal (cos(H(z)). It changes only the magnitude part of the signal.    
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
          C = torch.cos(x[:,2,...]).unsqueeze(-3)
          return torch.cat([x[:,0:2,...],C], dim=-3) #C#
        return x     

class TwoD_FFT(torch.nn.Module):
    def __init__(self, p=0.5):
        """ 2D FFT on input tensor (space/antenna and frequency).    
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
          C = x[:,0:2,...]
          A_ReImag = torch.rearrange(C, 'b c a s -> b a s c').contiguous()#W.view(-1,32).permute(1,0).contiguous()
          A_ReImag_complex = torch.view_as_complex(A_ReImag)
          A_fft2 = torch.fft.fft2(A_ReImag_complex)
          A_fft_re_imag = torch.cat([torch.real(A_fft2).unsqueeze(-1),torch.imag(A_fft2).unsqueeze(-1),
                            torch.abs(A_fft2).unsqueeze(-1)], dim=-1)
          return torch.rearrange(A_fft_re_imag,'b a s c -> b c a s')
        return x
        
class Cosine_Sine_Filter(torch.nn.Module):
    def __init__(self):
        """ Cosine or Sine the Signal (cos(H(z)). It changes only the magnitude part of the signal.    
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.random_val = torch.rand(1)

    def forward(self, x):
        if self.random_val < 0.5:
          C = torch.cos(x[:,2,...]).unsqueeze(-3)
          return torch.cat([x[:,0:2,...],C], dim=-3)
        elif self.random_val >= 0.5:
          C = torch.sin(x[:,2,...]).unsqueeze(-3)
          return torch.cat([x[:,0:2,...],C], dim=-3)
        return x

    def string(self):
        """
        Nothing!
        """
        return f'y = {self.a.item()}+x'        

class OnlyReal(torch.nn.Module):
    def __init__(self, p=0.5):
        """ Show Only Real Part, Make IMag part all zeros.    
        Args:
          p (float): probability of adding noise to Abs part. Default value is 0.5
        """
        super().__init__()
        self.p = p
        self.ZeroingM = torch.zeros((1,64,32)).cuda()

    def forward(self, x):
        if torch.rand(1) < self.p:
          Ze = torch.zeros((x.shape[0],1,64,32)).cuda()
          A_Re = x[:,0,...].unsqueeze(-3)
          A_Abs = x[:,2,...].unsqueeze(-3)
          A_Re_Z = torch.cat([A_Re,Ze,A_Abs], dim=-3)
          return A_Re_Z
        return x

class OnlyImag(torch.nn.Module):
    def __init__(self, p=0.5):
        """ Show Only Real Part, Make IMag part all zeros.    
        Args:
          p (float): probability of adding noise to Abs part. Default value is 0.5
        """
        super().__init__()
        self.p = p
        self.ZeroingM = torch.zeros((1,64,32)).cuda()

    def forward(self, x):
        if torch.rand(1) < self.p:
          Ze = torch.zeros((x.shape[0],1,x.shape[-2],x.shape[-1])).cuda()
          A_Re = x[:,1,...].unsqueeze(-3)
          A_Abs = x[:,2,...].unsqueeze(-3)
          A_Re_Z = torch.cat([A_Re,Ze,A_Abs], dim=-3)
          return A_Re_Z
        return x


class ReAbsChans(torch.nn.Module):
    def __init__(self, p=1.0):
        """ Get 2 channels (Re and Abs) 
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) <= self.p:
          tensor = torch.Tensor([0,2]).to(dtype=torch.long)
          #idx = perm[:self.num_sub]
          x = x.index_select(-3,tensor)
          return x
        return x
    

class ImagAbsChans(torch.nn.Module):
    def __init__(self, p=1.0):
        """ Get 2 channels (Imag and Abs) 
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) <= self.p:
          tensor = torch.Tensor([1,2]).to(dtype=torch.long)
          #idx = perm[:self.num_sub]
          x = x.index_select(-3,tensor)
          return x
        return x   

class AbsChans(torch.nn.Module):
    def __init__(self, p=1.0):
        """ Get 2 channels (Imag and Abs) 
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) <= self.p:
          tensor = torch.Tensor([2]).to(dtype=torch.long)
          #idx = perm[:self.num_sub]
          x = x.index_select(-3,tensor)
          return x
        return x

class RealImag(torch.nn.Module):
    def __init__(self, p=1.0):
        """ Get 2 channels (Real and Imag) 
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) <= self.p:
          tensor = torch.Tensor([0,1]).to(dtype=torch.long)
          x = x.index_select(-3,tensor)
          return x
        return x

    def string(self):
        """
        Nothing!
        """
        return f'y = {self.a.item()}+x'  

class TenLogTen(torch.nn.Module):
    def __init__(self, p=1.0):
        """ 10log10
        Args:
          p (float): probability of transformation to happen. Default value is 0.5
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) <= self.p:
          x = 10*torch.log10(x)
          return x
        return x 