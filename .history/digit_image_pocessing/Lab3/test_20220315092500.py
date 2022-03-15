#%%
import numpy as np
import matplotlib.pyplot as plt


img = plt.imread(r"digit_image_pocessing\Lab3\translated_rectangle.tif")
h,w = img.shape

center = h/2,w/2

u_grid = np.linspace(0,h-1,h)
v_grid = np.linspace(0,w-1,w)

grid = np.meshgrid(u_grid,v_grid)


print(grid)







# %%
