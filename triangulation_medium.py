import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
from matplotlib import cm
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
            
def triangulateFace(ax, file, cmap=cm.magma):
    v = np.array(readFile(file))
    tri = Delaunay(v[:,:2])
    ax.plot_trisurf(
        v[:,0], v[:,1], v[:,2],
        triangles = tri.simplices,
        cmap=cmap
    )

def triangulateSphere(ax, k=30, cmap=cm.magma):
  
    # sphere parametrization
    U = np.linspace(0, np.pi, k)
    V = np.linspace(0, 2 * np.pi, k)
    S1 = np.zeros((k,k))
    S2 = np.zeros((k,k))
    S3 = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            S1[i,j] = np.cos(U[i]) * np.cos(V[j])
            S2[i,j] = np.sin(U[i]) * np.cos(V[j])
            S3[i,j] = np.sin(V[j])

    # triangulate the points in [0,2pi] x [0,2pi]
    [T1, T2] = np.meshgrid(U, V)
    tri = Delaunay(np.array([T1.flatten(), T2.flatten()]).T)
    
    # plot the sphere
    ax.plot_trisurf(
        S1.flatten(), S2.flatten(), S3.flatten(),
        triangles=tri.simplices,
        cmap=cmap
    )

def triangulateEllipsoid(ax, A, k=30,cmap=cm.magma):
    
    # sphere parametrization
    U = np.linspace(0, np.pi, k)
    V = np.linspace(0, 2 * np.pi, k)
    S1 = np.zeros((k,k))
    S2 = np.zeros((k,k))
    S3 = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            S1[i,j] = np.cos(U[i]) * np.cos(V[j])
            S2[i,j] = np.sin(U[i]) * np.cos(V[j])
            S3[i,j] = np.sin(V[j])

    # map sphere to elipsoid
    E1 = np.zeros((k,k))
    E2 = np.zeros((k,k))
    E3 = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            xyz = np.array([S1[i,j], S2[i,j], S3[i,j]])
            [E1[i,j], E2[i,j], E3[i,j]] = A @ xyz

    # triangulate the elipsoid
    [T1, T2] = np.meshgrid(U, V)
    tri = Delaunay(np.array([T1.flatten(), T2.flatten()]).T)
    
    # plot the elipsoid
    ax.plot_trisurf(
        E1.flatten(), E2.flatten(), E3.flatten(),
        triangles=tri.simplices, cmap=cmap
    )


A = np.array([
    [-0.01289453, -0.02087514,  0.04109751],
    [-0.00261222, -0.01984956, -0.15409974],
    [-0.00431062,  0.07447336, -0.0295528 ]
])

triangulateEllipsoid(ax, A)
plt.show()

