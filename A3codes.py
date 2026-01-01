# A3codes.py
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import numpy.linalg as la
from A3helpers import synClsExperiments as _syn
from A3helpers import generateData, augmentX


#Q1(a)
def minMulDev(X, Y):
    n, d = X.shape
    k = Y.shape[1]
    
    def objective(w_flat):
        W = w_flat.reshape(d, k)
        scores = X @ W
        
        #Calculate loss
        #Tried direct exp first but had overflow issues

        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)  
        sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
        log_probs = scores - max_scores - np.log(sum_exp)
        
        loss = -np.sum(Y * log_probs) / n
            
        return loss
    
    w0 = np.random.randn(d * k) * 0.01  #small random init #works better than 0

    result = minimize(objective, w0, method='L-BFGS-B')
    
    
    return result.x.reshape(d, k)


#Q1(b)
def classify(Xtest, W):
    scores = Xtest @ W   #(m * k)

    #Get indx of the max for each row
    class_idx = np.argmax(scores, axis=1)

    #Create onehot
    m = Xtest.shape[0]
    k = W.shape[1]
    Yhat = np.zeros((m, k))
    Yhat[np.arange(m), class_idx] = 1

    return Yhat


#Q1(c)
def calculateAcc(Yhat, Y):
    pred_labels = np.argmax(Yhat, axis=1)
    true_labels = np.argmax(Y, axis=1)
    acc = np.mean(pred_labels == true_labels)
    return acc


#Q2(a)
def PCA(X, k):
    n, d = X.shape
    
    #Center data
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    
    #matrix
    C = X_centered.T @ X_centered
    
    #Eigen
    eigenvalues, eigenvectors = eigh(C)
    
    #Sort
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    #Take top
    U = eigenvectors[:, sorted_indices[:k]].T
    
    return U #, mu  <-Error


#Q2(b)
def projPCA(Xtest, mu, U):
    #Center test
    X_centered = Xtest - mu
    #Proj
    Xproj = X_centered @ U.T
    return Xproj


#Q2(c)
def kernelPCA(X, k, kernel_func):
    n, d = X.shape
    X = X.astype(float)
    
    #matrix
    K = kernel_func(X, X)
    
    #Center
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    #eigen
    eigenvalues, eigenvectors = eigh(K_centered)
    
    #Sort&take top k
    sorted_idx = np.argsort(eigenvalues)[::-1][:k]
    top_eigenvalues = eigenvalues[sorted_idx]
    top_eigenvectors = eigenvectors[:, sorted_idx]  # n * k

    A = top_eigenvectors.T  # k * n
    for i in range(k):
        A[i] = A[i] / np.sqrt(top_eigenvalues[i] * n)
    
    return A


#Q2(d)
def projkernelPCA(Xtest, Xtrain, kernel_func, A):
    m = Xtest.shape[0]
    n = Xtrain.shape[0]
    
    Xtrain = Xtrain.astype(float)
    Xtest = Xtest.astype(float)
    
   
    K_te_tr = kernel_func(Xtest, Xtrain)  # m * n
    K_tr_tr = kernel_func(Xtrain, Xtrain)  # n * n
 
    ones_m_n = np.ones((m, n)) / n
    ones_n_n = np.ones((n, n)) / n
    
    K_te_tr_centered = (K_te_tr - ones_m_n @ K_tr_tr - 
                        K_te_tr @ ones_n_n + ones_m_n @ K_tr_tr @ ones_n_n)
    

    Xproj = K_te_tr_centered @ A.T
    
    return Xproj


#Q2(e)
def synClsExperimentsPCA():
    n_runs = 100
    n_train = 128
    n_test = 1000
    dim_list = [1, 2]
    gen_model_list = [1, 2]
    
    train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    
    #group number
    np.random.seed(78)

    for r in range(n_runs):
        for i, k in enumerate(dim_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = generateData(n=n_test, gen_model=gen_model)

                U = PCA(Xtrain, k)

                mu = np.mean(Xtrain, axis=0)

                Xtrain_proj = projPCA(Xtrain, mu, U)
                Xtest_proj = projPCA(Xtest, mu, U)

                Xtrain_proj = augmentX(Xtrain_proj)
                Xtest_proj = augmentX(Xtest_proj)

                W = minMulDev(Xtrain_proj, Ytrain)

                Yhat_train = classify(Xtrain_proj, W)
                train_acc[i, j, r] = calculateAcc(Yhat_train, Ytrain)

                Yhat_test = classify(Xtest_proj, W)
                test_acc[i, j, r] = calculateAcc(Yhat_test, Ytest)

    avg_train = np.mean(train_acc, axis=2)
    avg_test = np.mean(test_acc, axis=2)

    return avg_train, avg_test


# Q3(a)
def kmeans(X, k, max_iter=1000):
    n, d = X.shape
    assert max_iter > 0 and k < n
    
    #randomly selecting k points from X
    U = X[np.random.choice(n, k, replace=False)]
    
    for _ in range(max_iter):
        #distances (n * k)
        D = cdist(X, U, metric='sqeuclidean')
        
        #build Y (n * k)
        closest = np.argmin(D, axis=1)
        Y = np.zeros((n, k))
        Y[np.arange(n), closest] = 1
        
        #Update
        old_U = U.copy()
        U = np.linalg.pinv(Y) @ X
        
    
        if np.allclose(old_U, U):
            break
    
    obj_val = (0.5 / n) * np.sum(D.min(axis=1))
    
    return Y, U, obj_val

#Q3(b)
def repeatKmeans(X, k, n_runs=100):
    best_obj_val = float('inf')
    best_Y = None
    best_U = None
    
    for _ in range(n_runs):
        Y, U, obj_val = kmeans(X, k)
        
        #Check run
        if obj_val < best_obj_val:
            best_obj_val = obj_val
            best_Y = Y
            best_U = U
    
    return best_Y, best_U, best_obj_val

#Q3(c)
def chooseK(X, k_candidates=[2,3,4,5,6,7,8,9]):
    obj_val_list = []
    
    for k in k_candidates:
        _, _, obj_val = repeatKmeans(X, k)
        obj_val_list.append(obj_val)
    
    return obj_val_list


#Q3(d)
def kernelKmeans(X, kernel_func, k, init_Y, max_iter=1000):
    n, d = X.shape
    X = X.astype(float)
    
    #kernel matrix
    K = kernel_func(X, X)
    
    Y = init_Y.copy()
    
    for iteration in range(max_iter):
        Y_plus = np.linalg.pinv(Y)  # k * n
        
        diag_K = np.diag(K).reshape(-1, 1)  # n * 1
        diag_Yplus = np.diag(Y_plus @ K @ Y_plus.T).reshape(1, -1)  # 1 * k
        
        D = (diag_K @ np.ones((1, k)) + 
             np.ones((n, 1)) @ diag_Yplus - 
             2 * K @ Y_plus.T)
        
        #Update
        old_Y = Y.copy()
        closest = np.argmin(D, axis=1)
        Y = np.eye(k)[closest]
        
        if np.allclose(old_Y, Y):
            break
    
    obj_val = 0.5 * np.sum(np.min(D, axis=1)) / n
    return Y, obj_val