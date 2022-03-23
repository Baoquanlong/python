import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import cv2
J = 1j # 复数i

def My_fft2d(img, shift=True):
    '''
    输入：
        img: 归一化后图像
        shift: 
            bool type
            if 'False': no shifting operation 
    
    输出：
        img_fft: img的2d fourier 变换后结果
    '''
    img_fft = np.fft.fft2(img)
    if shift:
        img_fft = np.fft.fftshift(img_fft)

    return img_fft

def My_ifft2d(img_fft, shift=True):
    '''
    输入：
        img_fft: fft 后图像
        shift: 
            bool type
            if 'False': no shifting operation 
    
    输出：
        img_rec: fft 反变换后重建图像
    '''
    if shift:
        img_fft = np.fft.fftshift(img_fft)
    img_rec = np.fft.ifft2(img_fft)
    
    return img_rec


def Inverse_filtering(img_input):
    T = 1
    a = b = 0.1
    M = img_input.shape[0]
    N = img_input.shape[1]
    u_grid =  np.linspace(0,M-1,M)
    v_grid = np.linspace(0,N-1,N)

    ticks = np.meshgrid(v_grid,u_grid)
    

    G = My_fft2d(img_input)

    H = (T*np.sin(np.pi*((ticks[1]-M/2)*a + (ticks[0]-N/2)*b))*np.exp(-J*np.pi*((ticks[1]-M/2)*a + (ticks[0]-N/2)*b)))/np.pi*((ticks[1]-M/2)*a + (ticks[0]-N/2)*b) 

    H = np.where(np.isnan(H),1,H)

    #wienner
    K = 1
    H_conjugate = np.conjugate(H)
    product = np.real(H*H_conjugate)



    # F = (1/H)*(product/product+K)* G
    # img_wienner = My_ifft2d(F)

    #最小二乘
    gamma = 0.5
    _p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    P = np.zeros(img_input.shape)
    start = img_input.shape[0]//2-1
    end = start+3
    P[start:end, start:end] = _p
    P = My_fft2d(P)
    P_conjugate = np.conjugate(P)
    # F = (H_conjugate/((product)+gamma*(P*P_conjugate))) * G

    # img_minus = My_ifft2d(F)

    fig, ax = plt.subplots(1,3)

    ax[0].imshow(img_input, cmap = 'gray')
    # ax[1].imshow(np.real(img_wienner), cmap = 'gray')
    # ax[2].imshow(np.real(img_minus), cmap = 'gray')

    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return



img_names = ['uniform_noise.tif',  'gaussian_noise.tif', 'saltpep_noise.tif',
                'moving.tif', 
                'donut.tif']
img_list = []
for img_name in img_names:
    temp_img = plt.imread(img_name)
    img_list.append(temp_img)

# Denoise(img_list[:3])
Inverse_filtering(img_list[-2])