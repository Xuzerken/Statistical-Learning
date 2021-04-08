import numpy as np 
import matplotlib.pyplot as plt 

t0_1 = np.array([1,1])
t0_2 = np.array([0.1,0.1])

def da(a, b):
    return -9*(a**(4/3))*(a+0.5*b-1)
def db(a, b):
    return -9*0.5*(b**(4/3))*(a+0.5*b-1)
def err(da, db):
    return da**2 + db**2

def gd(t0):
    delta = 0.0025
    x = np.arange(0, 2, delta)
    y = np.arange(0, 2, delta)
    X, Y = np.meshgrid(x, y)
    Z = (X**(2/3))*((X/t0[0])**(1/3)-3/2) + (Y**(2/3))*((Y/t0[1])**(1/3)-3/2)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=100)
    ax.plot(1-0.5*y ,y)
    ax.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("$\\theta^{(1)}$")
    plt.ylabel("$\\theta^{(2)}$")

    a = t0[0]
    b = t0[1]
    path_a = []
    path_a.append(a)
    path_b = []
    path_b.append(b)
    epsilon = 1e-8
    rate = 1e-2
    maxiter = 10000
    for i in range(maxiter):
        delta_a = da(a,b)
        delta_b = db(a,b)
        if(err(delta_a, delta_b) < epsilon):
             break
        a = a + rate*delta_a
        b = b + rate*delta_b
        path_a.append(a)
        path_b.append(b)
    ax.plot(a, b, marker='x', color='r')
    ax.plot(path_a, path_b, 'r')
    plt.savefig("./regularization/images/gd/theta0_1="+str(t0[0])+"theta0_2="+str(t0[1])+".png")

gd(t0_1)
gd(t0_2)