# from cgi import print_directory
# from urllib.request import ProxyDigestAuthHandler
# import numpy as np 
# import matplotlib.pyplot as plt
# import cv2
# J = 1j # 复数i

# def My_fft2d(img, shift=True):
#     '''
#     输入：
#         img: 归一化后图像
#         shift: 
#             bool type
#             if 'False': no shifting operation 
    
#     输出：
#         img_fft: img的2d fourier 变换后结果
#     '''
#     img_fft = np.fft.fft2(img)
#     if shift:
#         img_fft = np.fft.fftshift(img_fft)

#     return img_fft

# def My_ifft2d(img_fft, shift=True):
#     '''
#     输入：
#         img_fft: fft 后图像
#         shift: 
#             bool type
#             if 'False': no shifting operation 
    
#     输出：
#         img_rec: fft 反变换后重建图像
#     '''
#     if shift:
#         img_fft = np.fft.fftshift(img_fft)
#     img_rec = np.fft.ifft2(img_fft)
    
#     return img_rec


# img_input = plt.imread("moving.tif")

# T = 1
# a = b = 0.1
# M = img_input.shape[0]
# N = img_input.shape[1]
# u_grid =  np.linspace(0,M-1,M)
# v_grid = np.linspace(0,N-1,N)
# v,u = np.meshgrid(v_grid,u_grid)

# u = u-M/2
# v = v-N/2

# G = My_fft2d(img_input)


# numerator = T * np.sin((np.pi)*(u*a+v*b)) * np.exp(-J* np.pi * (u*a+v*b))
# denominator = np.pi*(u*a+v*b)
# numerator = np.where(numerator ==0, 1e-9,numerator)
# denominator = np.where(denominator ==0,1e-9,denominator)

# H = numerator/denominator

# H_conjugate = np.conjugate(H)
# product = np.real(H*H_conjugate)

# K =10
# F = (1/H)*(product/product+K)*G


import numpy as np 
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

import cv2

# img = plt.imread("donut.tif")
# img1 = radon(img)
# img2 = iradon(img1)

# error = np.sqrt((img-img2))

# fig,ax = plt.subplots(1,3)
# ax[0].imshow(img1,cmap = "gray")
# # ax[1].imshow(img2,cmap = "gray")
# ax[2].imshow(error,cmap = "gray")
# plt.show()
a = np.array([[1,2,3],[2,2,3],[1,5,1]])
b = np.array([[1,2,3],[2,2,3],[1,5,1]])
print(a**2+b**2)
