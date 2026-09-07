import numpy as np
import cv2 as cv
import scipy.io
import sys
from matplotlib import pyplot as plt

#Estimate Fundamental Matrix
def estimate_fundamental (ptsl, ptsr):


    # Fundamental Matrix Estimation
    # A_nx9.Fs_9x1 = 0

    # Assemble matrix A
    # Calculate matrix Ai for each corresponding pair of points using the Kronecker Product


    # Stack the equation for each pair of matchings



    # Compute the SVD of matrix A

    # Pick the singular vector corresponding to the smallest singular valeu
    # Define the first estimation for the Fundamental matrix


    # Compute the SVD of the first estimation of the Fundamental Matrix

    # Substitute the diagonal matrix to project the first estimation into the
    # Fundamental Matrix Space/Domain
    # Compute the new Fundamental Matrix belonging to the right domain



    return F/F[2,2]


def estimate_normalized_fundamental (ptsl, ptsr):

    ptsl_n, Tl = normalize(ptsl)
    ptsr_n, Tr = normalize(ptsr)
    # Estimate essential matrix
    F = estimate_fundamental (ptsl_n, ptsr_n)
    # reverse normalization
    F = np.dot(Tr.T,np.dot(F,Tl))

    return F/F[2,2]

# Normalize the points
def normalize(pts):

    # to be sure that the last line is all of ones - homogeneous coordinates
    pts = pts / pts[2]

    npoints = pts.shape[1] # number of columns
    # Calculate the Centroid
    #centx = np.sum (pts[0,:])/npoints
    #centy = np.sum (pts[1,:])/npoints
    centroid = np.mean(pts[:2],axis=1)
    # Calculation of the  mean distance in relation to the centroid
    dist_med = sum(np.sqrt((pts[0,:] - centroid[0])**2 + (pts[1,:] - centroid[1])**2))/npoints
    # Scale to make the mean distance equal to  sqrt(2)
    esc = np.sqrt(2)/dist_med
    esc = np.sqrt(2)/np.std(pts[:2])
    # Normalization matrix
    T = np.array([[esc, 0, -esc*centroid[0]],[0, esc, -esc*centroid[1]],[ 0, 0, 1]])

    #Normalized points
    pts_norm = np.dot(T,pts)

    return pts_norm,T


def plot_epipolar_line(im,F,x,xx):

  # Plot the epipolar line F*x in an image. F is the fundamental matrix,
  # x a point in the other image and xx a point in the image where the
  # epipolar line will be plotted.

  m,n = im.shape[:2]
  line = np.dot(F,x)

  # epipolar line parameter and values
  t = np.linspace(0,n,100)
  lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

  # take only line points inside the image
  ndx = (lt>=0) & (lt<m)
  plt.plot(t[ndx],lt[ndx],linewidth=1)
  plt.plot(xx[0],xx[1],'o')

