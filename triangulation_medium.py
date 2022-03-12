from operator import truediv
import math
from itertools import product
import numpy as np
from scipy.linalg import lstsq, schur
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

ax = plt.figure().gca(projection='3d')

def readFile(file):
    v = []
    with open(file, 'r') as vertices:
        for line in vertices:
            v.append(list(map(float, line.split())))
    return v

def scatter(ax, file):
    v = readFile(file)
    for p in v: ax.scatter(*p, color='black', s=10)

def triangulate(ax, file):
    v = np.array(readFile(file))
    tri = Delaunay(v[:,:2])
    ax.plot_trisurf(
        v[:,0], v[:,1], v[:,2],
        triangles = tri.simplices,
        cmap=cm.inferno
    )

def drawSphere(ax, k=30):
  
    # sphere parametrization
    T = np.linspace(0, 2 * np.pi, k)
    X = np.zeros((k,k))
    Y = np.zeros((k,k))
    Z = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            X[i,j] = np.cos(T[i]) * np.sin(T[j])
            Y[i,j] = np.sin(T[i]) * np.sin(T[j])
            Z[i,j] = np.cos(T[j])

    # triangulate the points in [0,2pi] x [0,2pi]
    [T1, T2] = np.meshgrid(T, T)
    tri = Delaunay(np.array([T1.flatten(), T2.flatten()]).T)
    
    # plot the sphere
    ax.plot_trisurf(
        X.flatten(), Y.flatten(), Z.flatten(),
        triangles=tri.simplices, cmap=cm.inferno,
        lightsource = LightSource()
    )

def triangulateEllipsoid(ax, A, k=30):
    # sphere parametrization
    T = np.linspace(0, 2 * np.pi, k)
    X = np.zeros((k,k))
    Y = np.zeros((k,k))
    Z = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            X[i,j] = np.cos(T[i]) * np.sin(T[j])
            Y[i,j] = np.sin(T[i]) * np.sin(T[j])
            Z[i,j] = np.cos(T[j])

    # map sphere to elipsoid
    U1 = np.zeros((k,k))
    U2 = np.zeros((k,k))
    U3 = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            xyz = np.array([X[i,j], Y[i,j], Z[i,j]])
            [U1[i,j], U2[i,j], U3[i,j]] = A @ xyz

    # triangulate the elipsoid
    [T1, T2] = np.meshgrid(T, T)
    tri = Delaunay(np.array([T1.flatten(), T2.flatten()]).T)
    
    # plot the elipsoid
    ax.plot_trisurf(
        U1.flatten(), U2.flatten(), U3.flatten(),
        triangles=tri.simplices, cmap=cm.inferno,
        lightsource = LightSource()
    )


A = np.array([
    [-0.01289453, -0.02087514,  0.04109751],
    [-0.00261222, -0.01984956, -0.15409974],
    [-0.00431062,  0.07447336, -0.0295528 ]
])

triangulateEllipsoid(ax, A)
plt.show()
        
    
