import numpy as np
from numba import njit
import sys
from multiprocessing import Process

@njit
def A2B(lattice,L,T,h):
    N = L**2
    m = lattice.sum()
    count = 0
    J = 1

    while m < 0.8*N:
        count += 1
        y = np.random.randint(0,L)
        x = np.random.randint(0,L)

        b = 0
        for i,j in ((-1,0),(1,0),(0,1),(0,-1)):
            b += lattice[(y+j)%L,(x+i)%L]
        
        #E = -J*b*lattice[y,x] - h*lattice[y,x]
        #E1 = -E
        #dE = E1 - E = -2*E
        dE = 2*(J*b+h)*lattice[y,x]
        if dE <= 0 or np.random.random() < np.exp(-dE/T):
            #if lattice[y,x] == 1:
            #    m -= 2
            #else:
            #    m += 2
            m -= 2*lattice[y,x]
            lattice[y,x] *= -1

    return count
 
@njit
def eq(lattice,L,T,h):
    N = L**2
    count = 0
    J = 1

    for count in range(100*N):
        y = np.random.randint(0,L)
        x = np.random.randint(0,L)

        b = 0
        for i,j in ((-1,0),(1,0),(0,1),(0,-1)):
            b += lattice[(y+j)%L,(x+i)%L]
        
        dE = 2*(J*b+h)*lattice[y,x]
        if dE <= 0 or np.random.random() < np.exp(-dE/T):
            lattice[y,x] *= -1

    return lattice

def mc(L,T,h,index):
    # A: low m
    # B: high m
    N = L**2
    list_A2B = []

    for i in range(1000):
        m = N
        # Equilibrate
        while m > -0.8*N:
            lattice = np.full((L,L),-1,dtype=np.int8)
            lattice = eq(lattice,L,T,h)
            m = lattice.sum()
        count = A2B(lattice,L,T,h)
        print(index,i,count,flush=True)
        list_A2B.append(count)

    #np.save("list_A2B_brute%i.npy"%index,list_A2B)

if __name__ == "__main__":
    L = int(sys.argv[1])
    T = float(sys.argv[2])
    h = float(sys.argv[3])
    nodes = int(sys.argv[4])

    jobs = [Process(target=mc,args=(L,T,h,i)) for i in range(nodes)]
    for job in jobs: job.start()
    for job in jobs: job.join()

