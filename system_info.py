
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import cv2
import scipy.special
import scipy.ndimage as nd
import copy


edge_length = 208
curvature = 0.07179
rmax = 18.3
rmin = 14.1
rstar = (rmax+rmin)/2
#suppose the resolution of the TEM is 0.2 nm
resolution = 0.2
frame_sz = int(208 * 0.7/resolution)
thickness = 6.78

Rmax = int(rmax/resolution)
Rmin = int(rmin/resolution)
dthick = int(thickness/resolution)

Ravg = (Rmax+Rmin)//2
hface = int(math.sqrt(Rmax**2-Ravg**2))
mask_sz = int(2*hface+1)
sigma = int(0.24/resolution)
apr_r_Rmin = int(np.round(math.sqrt(Ravg ** 2 - Rmin ** 2)))

'''edge_st = int(edge_length * curvature/resolution * math.sqrt(3))
cx = edge_st
cy = int(edge_length*curvature/resolution)
print('starting point of prism is at pixel number', edge_length * 0.1/resolution)'''
Rc = int(curvature*edge_length/resolution)
Pedge = int(edge_length/resolution)
Pedge_h = int(edge_length/resolution*math.sqrt(3)/2)
o1r,o1c = Pedge_h-1-Rc,int(math.sqrt(3)*Rc)
df = pd.read_excel('data/automated position classify1.xlsx')


