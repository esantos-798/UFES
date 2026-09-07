import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import ndimage, misc
from PIL import Image

# Read images
IL = cv2.imread('left.ppm') # left image
IR = cv2.imread('right.ppm')  # right image
gray1 = cv2.cvtColor(IL, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(IR, cv2.COLOR_BGR2GRAY)

print(IL.shape)


# intrinsic parameter matrix
fm = 403.657593 # Focal distantce in pixels
cx = 161.644318 # Principal point - x-coordinate (pixels) 
cy = 124.202080 # Principal point - y-coordinate (pixels) 
bl = 119.929 # baseline (mm)
# for the right camera    
right_k = np.array([[ fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

# for the left camera
left_k = np.array([[fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

# Extrinsic parameters
# Translation between cameras
T = np.array([-bl, 0, 0]) 
# Rotation
R = np.array([[ 1,0,0],[ 0,1,0],[0,0,1]])

print('Intrinsic Paramenters')
print('Left_K:\n', left_k)
print('Right_K:\n', right_k)

print('Extrinsic Paramenters')
print('R:\n', R)
print('T:\n', T)

fm = left_k[0,0]
cx = left_k[0,2]
cy = left_k[1,2]

im_l = np.array(Image.open('left.ppm').convert('L'),'f')
im_r = np.array(Image.open('right.ppm').convert('L'),'f')
steps = 60
start = 0
wid = 2.5
def plane_sweep_gauss(im_l,im_r,start,steps,wid):
    """ Find disparity image using normalized cross-correlation
    with Gaussian weighted neigborhoods. """
    m,n = im_l.shape

    # arrays to hold the different sums
    mean_l = np.zeros((m,n))
    mean_r = np.zeros((m,n))
    s = np.zeros((m,n))
    s_l = np.zeros((m,n))
    s_r = np.zeros((m,n))

    # array to hold depth planes
    dmaps = np.zeros((m,n,steps))

    # compute mean
    ndimage.gaussian_filter(im_l,wid,0,mean_l)
    ndimage.gaussian_filter(im_r,wid,0,mean_r)

    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r

    # try different disparities
    for displ in range(steps):
      # move left image to the right, compute sums
      ndimage.gaussian_filter(norm_l*np.roll(norm_r,displ+start),wid,0,s)  # sum nominator
      ndimage.gaussian_filter(norm_l*norm_l,wid,0,s_l)
      ndimage.gaussian_filter(np.roll(norm_r,displ+start)*np.roll(norm_r,displ+start),wid,0,s_r) # sum denominator

      # store ncc scores
      dmaps[:,:,displ] = s/np.sqrt(s_l*s_r)

      # pick best depth for each pixel
      best_map = np.argmax(dmaps,axis=2)+ start


    return best_map

best_map = plane_sweep_gauss(im_l,im_r,start,steps,wid)
print(best_map)

m, n = best_map.shape

Z = fm * bl / best_map

u, v = np.meshgrid(np.arange(n), np.arange(m))

X = (u - cx) * Z / fm
Y = (v - cy) * Z / fm

mask = (best_map > 0) & (Z < 2000)

X = X[mask]
Y = Y[mask]
Z = Z[mask]

plt.figure()
plt.imshow(best_map, cmap='jet')
plt.colorbar(label='Disparidade (pixels)')
plt.title(f'Mapa de Disparidade: steps={steps}, start={start}, wid={wid}')
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

colors = cv2.cvtColor(IL, cv2.COLOR_BGR2RGB)/255.0
colors = colors[mask]

assert len(X) == len(Y) == len(Z) == len(colors)
ax.scatter(
    X,
    Y,
    Z,
    c=colors,
    s=1
)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("Reconstrução 3D")

plt.show()