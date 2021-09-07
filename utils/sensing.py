from math import sqrt
import numpy as np
import scipy.sparse as sp

def haar(N, mul1, mul2):
    print(N)
    if N == 2:
        return sp.csr_matrix(np.array([[1, 1], [1, -1]]))
    
    res = haar(N // 2, mul1, mul2)
    upper = sp.kron(res, mul1)
    lower = sqrt(N // 2) * sp.kron(sp.eye(N // 2), mul2)
    
    return sp.vstack([upper, lower])

def Haar(n):
    mul1 = sp.csr_matrix(np.array([1, 1]))
    mul2 = sp.csr_matrix(np.array([1, -1]))
    return 1 / sqrt(n) * haar(n, mul1, mul2)

def H_k(x, k):
    ind = np.argpartition(x, -k)[-k:]
    ind_other = list(set(range(len(x))) - set(ind))
    x[ind_other] = 0
    return x

def support(x):
    return np.where(x>1e-6)[0]

def threshold(x_1, x_0, phi, c=0.01):
    return (1-c) * np.linalg.norm(x_1-x_0, 2) ** 2 \
                 / np.linalg.norm(np.dot(phi, x_1-x_0), 2) ** 2

def cost(x_0, y_star, phi):
    return np.linalg.norm(y_star-np.dot(phi, x_0))

def compressive_sensing(args, adj):
    # initialize and pad the input (symmetric)
    n = adj.shape[0] ** 2
    n_pad = int(2 ** np.ceil(np.log2(n)))

    D = np.matrix.flatten(adj.A)
    n_pad_all = n_pad - n
    n_pad_left = n_pad_all // 2
    n_pad_right = n_pad_all - n_pad_left
    D = np.insert(D, 0, np.zeros(n_pad_left))
    D = np.insert(D, -1, np.zeros(n_pad_right))

    S = int(args.S)
    k = int(S * np.log(n_pad / S))

    # generate the sensing matrix (~ symmetric Bernoulli distr)
    phi = np.random.binomial(1, 1/2, k*n_pad)
    phi[np.where(phi==0)[0]] = -1
    # phi = phi.astype(np.float32) * coeff / sqrt(k)
    # coeff = 1
    # phi *= 
    phi = phi.reshape(k, n_pad)
    print('generating Phi done!')

    # generate the representation matrix
    haar_N = 1 / sqrt(n_pad) * Haar(n_pad)
    print('generating Haar matrix done!')

    # add noise in the latent space
    print(phi.shape, D.shape)
    y = np.dot(phi, D)
    print(y.shape)
    noise = np.random.laplace(0, 2*k/args.epsilon, size=k)
    y_star = y + noise
    print('adding noise done!')

    # recover
    x_0 = np.zeros_like(D, dtype=np.float32)
    ini = np.dot(phi.T, y)
    support_0 = np.argpartition(ini, -S)[-S:]
    cost_0 = cost(x_0, y_star, phi)

    for i in range(10000):
        g = np.dot(phi.T, y_star - np.dot(phi, x_0))
        
        g_tau = g[support_0]
        phi_tau = phi[:, support_0]
        mu = np.dot(g_tau.T, g_tau) / np.dot(np.dot(np.dot(g_tau.T, phi_tau.T), phi_tau), g_tau)
        
        x_1 = H_k(x_0 + mu * g, S)

        support_1 = support(x_1)
        
        if (support_1 == support_0).all():
            x_0 = x_1
        else:
            lim = threshold(x_1, x_0, phi, args.c)
            if mu <= lim:
                x_0 = x_1
                support_0 = support_1
            else:
                while mu > lim:
                    mu /= 2 # k * (1-c)
                    x_1 = H_k(x_0 + mu * g, S)
                    lim = threshold(x_1, x_0, phi, args.c)
                x_0 = x_1
                support_0 = support(x_1)
            
        cost_1 = cost(x_1, y_star, phi)
        if cost_0 - cost_1 < 1e-6:
            break
        cost_0 = cost_1
        print(f'{i}: {cost_0 : .4f}')

    # reconstruct
    ret = haar_N.dot(x_0)
    A = sp.csr_matrix(ret[n_pad_left : n_pad - n_pad_right].reshape(1000, 1000))
    A = sp.triu(A, k=1)
    A += A.T
    return A