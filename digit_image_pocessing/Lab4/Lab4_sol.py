#%%
from math import gamma
from cv2 import hconcat
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import cv2

#%%
'''
你的辅助函数可以写在这里
'''
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
   

#%%
'''
Q1: 实现对uniform_noise.tif, gaussian_noise.tif 和 saltpep_noise.tif 的去噪
输出比对结果 两行三列
uniform_noise.tif，  gaussian_noise.tif， saltpep_noise.tif
上面去噪后结果

可以自由选择空间或频率域滤波
'''
def Denoise(img_input):
    #img_input[0] 使用均值滤波
    img0 = img_input[0]
    # print(img0.dtype)
    img0 = cv2.blur(img0, (3,3))

    #img_input[1] 使用几何平均滤波， 就是对log（img1）使用均值滤波
    img1 = np.where(img_input[1] == 0, 1e-7, img_input[1])
    img1 = np.log(img1)
    img1 = cv2.blur(img1,(3,3))
    img1 = np.exp(img1)
    
    
    # #img_input[2] 使用中值滤波。
    img2 = img_input[2]
    img2 = cv2.medianBlur(img2,3)


    
    fig, ax = plt.subplots(2,3)
    for i in range(3):
        ax[0][i].imshow(img_input[i], cmap = 'gray')
    ax[1][0].imshow(img0, cmap = 'gray')
    ax[1][1].imshow(img1, cmap = 'gray')
    ax[1][2].imshow(img2, cmap = 'gray')

    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return


#%%
'''
Q2:使用moving.tif, 比较Wienner逆滤波和约束最小二乘逆滤波的结果
输出图像： 一行三列： 原图， Winner逆滤波结果， 约束最小二乘逆滤波结果
公式（5-77）里的H使用了 a = b = 0.1, T = 1 

注意：
   * 针对带平移的Fourier变换， 课本的公式需要使用其平移版本： u -> u-M/2, v -> v-N/2
   * 对有可能出现的nan，比如 sin(x)/x 数值计算x=0 时，需要作特别处理。
     比如 H 含 nan， 可以用 H = np.where(np.isnan(H), 1.0, H) 来方便地修改， 式子里
     使用1.0替换nan是因为 sin(x)/x 在x=0 时的理论值是1.0
   * 关于P(u,v)的获得，可以手动计算其显示表达式，再在数值格点上生成。更方便地，可以参见课本的描述 P253 式（5.90）下面一段，
     把这个3*3 的核嵌入到与原图像同样大小的零矩阵的“中心”， 然后再使用fft2d得到。同上，得到P(u,v)结果后不要忘了做一次平移。
   * 可以自由调试 K 和 gamma 的值，到你觉得结果ok的程度。
'''
def Inverse_filtering(img_input):
    
    G = My_fft2d(img_input)
    T = 1
    a = b = 0.1
    M = img_input.shape[0]
    N = img_input.shape[1]
    u_grid =  np.linspace(0,M-1,M)
    v_grid = np.linspace(0,N-1,N)
    v,u = np.meshgrid(v_grid,u_grid)

    u = u-M/2
    v = v-N/2

    numerator = T * np.sin((np.pi)*(u*a+v*b)) 
    denominator = np.pi*(u*a+v*b)
    numerator = np.where(numerator == 0, 1e-7,numerator)
    denominator = np.where(denominator == 0,1e-7,denominator)

    H = (numerator/denominator) * np.exp(-J* np.pi * (u*a+v*b))


    H_conjugate = np.conjugate(H)
    product =  H * H_conjugate 


    #wienner
    K = 0.5
    F = (product/product+K)*(G/H)
    img_wienner = My_ifft2d(F)
    img_wienner = np.real(img_wienner)

    #最小二乘
    gamma = 0.05
    _p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    P = np.zeros(img_input.shape)
    start = img_input.shape[0]//2-1
    end = start+3
    P[start:end, start:end] = _p
    P = My_fft2d(P)
    P_conjugate = np.conjugate(P)

    product1 = np.real(P*P_conjugate)

    denominator1 = product+gamma*product1
    denominator1 = np.where(denominator1==0,1e-9,denominator1)

    F = (H_conjugate/denominator1) * G

    img_minus = My_ifft2d(F)
    img_minus = np.real(img_minus)

    fig, ax = plt.subplots(1,3)

    ax[0].imshow(img_input, cmap = 'gray')
    ax[1].imshow(img_wienner, cmap = 'gray')
    ax[2].imshow(img_minus, cmap = 'gray')

    plt.show(block=False)
    plt.pause(10)
    plt.close()

    return


#%%
'''
Q3: 使用donut.tif, 利用
from skimage.transform import radon, iradon
导入的radon 正变换和反变换处理图像
输出出其Radon变换图像(左)，重建图像（中）及重建误差（右）
'''
def radon_test(img_input):
    img_radon = radon(img_input)
    img_iradon = iradon(img_radon)
    error = (img_input-img_iradon)**2

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(img_radon, cmap='gray')
    ax[1].imshow(img_iradon, cmap='gray')
    ax[2].imshow(error, cmap='gray')

    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return

questions = [Denoise, Inverse_filtering, radon_test]

#%%
if __name__ == '__main__':
    img_names = ['uniform_noise.tif',  'gaussian_noise.tif', 'saltpep_noise.tif',
                 'moving.tif', 
                 'donut.tif']
    img_list = []
    for img_name in img_names:
        temp_img = plt.imread(img_name)
        img_list.append(temp_img)

    # Denoise(img_list[:3])
    Inverse_filtering(img_list[-2])
    # radon_test(img_list[-1])
   


# %%
