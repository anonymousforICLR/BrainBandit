# @Start date : 2023/8/14
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import sympy  as sp
import seaborn as sns
from tqdm import tqdm
def func(x,gamma,w,I,b,n,k):
    def sig(x, n, k):
        return 1 / (1 + np.exp(n * (k - x)))
    xy =x
    return [-gamma[0]*xy[0]-sig(xy[1],n,k)*w[0]+I[0]+b[0],
            -gamma[1]*xy[1]-sig(xy[0],n,k)*w[1]+I[1]+b[1]]

def Energy(x,y):
    sx = np.exp(n*k)+np.exp(n*x)
    sy = np.exp(n*k)+np.exp(n*y)
    f1 = gamma[0]*( x*np.exp(n*x) / sx - np.log(sx)/n) + gamma[1]*(y*np.exp(n*y) / sy - np.log(sy)/n)
    f2 = -(I[0]+b[0])*np.exp(n*x) /sx -(I[1]+b[1])*np.exp(n*y) /sy
    f3 = 0.5*(w[0]+w[1])*(np.exp(n*x)/sx)*(np.exp(n*y)/sy)
    return f1+f2+f3

def calculate_escape(I:list = [2.,0], gamma:list = [0.5,0.5], w:list=[4.,4.], b:list = [5., 5.], n:float = 1., k:float=7.,noise:list=[1.,1.]):

    dim = 2
    p_list = []

    x , y = sp.symbols('x y')
    root1 = fsolve(func, x0=np.array([20, 0]), args=(gamma, w, I, b,n,k))
    root2 = fsolve(func, x0=np.array([0, 20]), args=(gamma, w, I, b,n,k))
    sx = sp.exp(n*k)+sp.exp(n*x)
    sy = sp.exp(n*k)+sp.exp(n*y)
    f1 = gamma[0]*( x*sp.exp(n*x) / sx - sp.log(sx)/n) + gamma[1]*(y*sp.exp(n*y) / sy - sp.log(sy)/n)
    f2 = -(I[0]+b[0])*sp.exp(n*x) /sx -(I[1]+b[1])*sp.exp(n*y) /sy
    f3 = 0.5*(w[0]+w[1])*(sp.exp(n*x)/sx)*(sp.exp(n*y)/sy)
    f = f1+f2+f3
    fx = sp.diff(f,x)
    fxx = sp.diff(fx,x)
    fy = sp.diff(f,y)
    fyy = sp.diff(fy,y)
    sigma = np.array(noise)
    tr = fxx * sigma[0] + fyy * sigma[1]
    p1 = tr.subs([(x,root1[0]),(y,root1[1])])
    p2 = tr.subs([(x, root2[0]), (y, root2[1])])
    P1 = p2 / (p1+p2)
    P2 = p1 / (p1+p2)
    return P1, P2

if __name__ == '__main__':
    dim = 2
    gamma = np.ones(dim) * 0.5
    w = np.ones(dim) * 4.
    I = [2, 0]
    b = np.ones(dim) * 5.0
    n = 1.0
    k = 7.
    p_list = []

    x , y = sp.symbols('x y')
    root1 = fsolve(func, x0=np.array([20, 0]), args=(gamma, w, I, b,n,k))
    root2 = fsolve(func, x0=np.array([0, 20]), args=(gamma, w, I, b,n,k))
    sx = sp.exp(n*k)+sp.exp(n*x)
    sy = sp.exp(n*k)+sp.exp(n*y)
    f1 = gamma[0]*( x*sp.exp(n*x) / sx - sp.log(sx)/n) + gamma[1]*(y*sp.exp(n*y) / sy - sp.log(sy)/n)
    f2 = -(I[0]+b[0])*sp.exp(n*x) /sx -(I[1]+b[1])*sp.exp(n*y) /sy
    f3 = 0.5*(w[0]+w[1])*(sp.exp(n*x)/sx)*(sp.exp(n*y)/sy)
    f = f1+f2+f3
    fx = sp.diff(f,x)
    fxx = sp.diff(fx,x)
    fy = sp.diff(f,y)
    fyy = sp.diff(fy,y)
    sigma = np.array([1.,1.])
    tr = fxx * sigma[0] + fyy * sigma[1]
    p1 = tr.subs([(x,root1[0]),(y,root1[1])])
    p2 = tr.subs([(x, root2[0]), (y, root2[1])])
    P1 = p2 / (p1+p2)
    P2 = p1 / (p1+p2)
    # print(P1)

    print(P2)