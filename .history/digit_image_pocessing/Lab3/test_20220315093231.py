#%%
import numpy as np
import matplotlib.pyplot as plt


img = plt.imread(r"digit_image_pocessing\Lab3\translated_rectangle.tif")
h,w = img.shape

center = h/2,w/2

u_grid = np.linspace(0,h-1,h)
v_grid = np.linspace(0,w-1,w)

grid = np.meshgrid(u_grid,v_grid)


print(grid)             # grid[0]为列标，grid[1] 为行标

Distance = np.sqrt((grid[1]-center[0])**2 + (grid[0]-center[1])**2)            # 表示各个点到中心点的距离

H = np.where(Distance < 100, 1, 0)

# 画出H看一下
plt.subplot(111)
plt.imshow(H, cmap ="gray")









# %%
