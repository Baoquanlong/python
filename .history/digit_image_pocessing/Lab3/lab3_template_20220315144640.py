'''
For use in Lab3

* 在写fft和ifft的时候，可以在空间上使用（-1）*（x+y）改变图像，也可以在频率域上做shift，使用numpy的fft模组
* 可以自由选择做不做padding
* 注意在读入图像后，进行运算前最好使用img/255做归一化
* 注意ifft后的数据类型是复数，显示前要转为实数
* 生成核函数的时候可以使用np.linspace()和np.meshgrid()先生成网格，再统一计算网格上的值
* 建议调试的时候画出kernel 函数的图像确认
* 交的作业不允许使用其他的import, 自己自由探索的时候可以随意。多对比不同包的处理效果。

'''
#%%
import numpy as np
import matplotlib.pyplot as plt

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

#Q1, 使用translated_rectangle.tif, 一行两列，左：原图， 右：Fourier 谱图
def question1(image_input):
    img = image_input/np.max(image_input)
    img_fft = My_fft2d(img)
    img_fft = np.log(abs(img_fft))


    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image_input , cmap='gray') # 原图
    plt.subplot(1,2,2)

    plt.imshow(img_fft , cmap='gray') # 谱图，注意做个log变换

    plt.show(block=False)
    plt.pause(3)
    plt.close()
    return

#Q2, 使用characters_test_pattern.tif, 一行三列，：ilpf, blpf, glpf
def question2(image_input):

    img = plt.imread(r"digit_image_pocessing\Lab3\translated_rectangle.tif")
    h,w = img.shape
    center = h/2,w/2
    u_grid = np.linspace(0,h-1,h)
    v_grid = np.linspace(0,w-1,w)
    grid = np.meshgrid(u_grid,v_grid)
    Distance = np.sqrt((grid[1]-center[0])**2 + (grid[0]-center[1])**2)            # 表示各个点到中心点的距离
    
    #image after fourier converter 
    img = My_fft2d(image_input)



    #ideal kernel
    D0 = 10
    iH = np.where(Distance < D0, 1, 0)
    img1 = img * iH 
    img1 = My_ifft2d(img1)



    #BLPF 
    D0 = 10
    n = 1
    bH = 1/(1+(Distance/D0)**2*n)
    img2 = img * bH 
    img2 = My_ifft2d(img2)

    #GLPF
    D0 = 40
    gH = np.exp(-Distance**2/(2*D0**2))
    img3 = img * gH 
    img3= My_ifft2d(img3)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img1 , cmap='gray') # ilpf
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap='gray') # blpf
    plt.subplot(1,3,3)
    plt.imshow(img3 , cmap='gray') # glpf

    plt.show(block=False)
    plt.pause(10)
    plt.close()

#Q3, 使用characters_test_pattern.tif, 一行三列，：ihpf, bhpf, ghpf
# def question3(image_input):

#     plt.figure()
#     plt.subplot(1,3,1)
#     plt.imshow( , cmap='gray') # ihpf
#     plt.subplot(1,3,2)
#     plt.imshow( , cmap='gray') #  bhpf
#     plt.subplot(1,3,3)
#     plt.imshow( , cmap='gray') # ghpf

#     plt.show(block=False)
#     plt.pause(3)
#     plt.close()

# #Q4, 使用blurry_moon.tif, 一行两列：原图，增强后图
# def question4(image_input):

#     plt.figure()
#     plt.subplot(1,2,1)
#     plt.imshow( , cmap='gray') # 原图
#     plt.subplot(1,2,2)
#     plt.imshow( , cmap='gray') # 增强后图
    
#     plt.show(block=False)
#     plt.pause(3)
#     plt.close()

# #Q5，使用cassini.tif, 一行两列：原图，处理后图
# def question5(image_input):

#     plt.figure()
#     plt.subplot(1,2,1)
#     plt.imshow( , cmap='gray') # 原图
#     plt.subplot(1,2,2)
#     plt.imshow( , cmap='gray') # 处理后图
    
#     plt.show(block=False)
#     plt.pause(3)
#     plt.close()

# questions = [question1, question2, question3, question4, question5]


if __name__ == '__main__':
    # image_input_list = [...]
    # question_count = len(questions)
    # for index in range(0, question_count):

    ## 自己的
    img = plt.imread(r"digit_image_pocessing\Lab3\translated_rectangle.tif")
    question2(img)    



    pass