import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
print(matplotlib.__version__)
np.set_printoptions(threshold=np.nan)

# Read in the image samples
img_org = cv2.imread('Lina_s_o.tif', 0)
img_1_1 = cv2.imread('Lena_2_10.tif', 0)
print(img_org.shape)
print(img_1_1.shape)

f1 = np.fft.fft2(img_org)
f2 = np.fft.fft2(img_1_1)

# Calculate phase correlation of image
CPS = np.multiply(f1, np.conj(f2))
NCPS = np.divide(CPS, np.abs(CPS))
d = np.fft.ifft2(NCPS)
poc = np.abs(d)
poc_shift = np.fft.fftshift(poc)
print(np.amax(poc_shift))   # Find the value of highest peak
print(np.unravel_index(poc_shift.argmax(), poc_shift.shape))   # Find the coordinate of the highest peak

# Plotting the result
fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.arange(128)
y = np.arange(128)
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, poc_shift, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_title('surface plot')
fig.show()
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
plt.savefig('figure.png')

