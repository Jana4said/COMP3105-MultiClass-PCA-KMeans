# COMP 3105 Assignment 3
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
from matplotlib import pyplot as plt

import GanaSaid.A3codes as A3codes
from A3helpers import augmentX, gaussKernel, plotModel, generateData, plotPoints
import numpy as np

def _plotCls():

	n = 100

	# Generate data
	Xtrain, Ytrain = generateData(n=n, gen_model=1)
	Xtrain = augmentX(Xtrain)

	# Learn and plot results
	W = A3codes.minMulDev(Xtrain, Ytrain)
	print(f"Train accuaracy {A3codes.calculateAcc(Ytrain, A3codes.classify(Xtrain, W))}")

	plotModel(Xtrain, Ytrain, W, A3codes.classify)

	return


def _testPCA():
	train_acc, test_acc = A3codes.synClsExperimentsPCA()
	print("Train accuracy: \n", train_acc)
	print("Test accuracy: \n", test_acc)
	return


def _plotKmeans():

	n = 100
	k = 3

	Xtrain, _ = generateData(n, gen_model=2)

	Y, U, obj_val = A3codes.kmeans(Xtrain, k)
	plotPoints(Xtrain, Y)
	plt.legend()
	plt.show()

	return


def _plotKernelKmeans():
	Xtrain, _ = generateData(n=100, gen_model=3)
	kernel_func = lambda X1, X2: gaussKernel(X1, X2, 0.25)
	n = Xtrain.shape[0]
	init_Y = np.eye(2)[np.random.choice(2, n)] 

	Y, obj_val = A3codes.kernelKmeans(Xtrain, kernel_func, 2, init_Y)
	plotPoints(Xtrain, Y)
	plt.legend()
	plt.show()
	return
# Create a temporary test script or add to A3testbed.py:

def _testQ1d():
    from A3helpers import synClsExperiments
    train_acc, test_acc = synClsExperiments(A3codes.minMulDev, 
                                           A3codes.classify, 
                                           A3codes.calculateAcc)
    
    print("=== Q1d Results ===")
    print("Training Accuracies:")
    print("n\tModel 1\tModel 2")
    n_values = [16, 32, 64, 128]
    for i, n in enumerate(n_values):
        print(f"{n}\t{train_acc[i, 0]:.3f}\t{train_acc[i, 1]:.3f}")
    
    print("\nTest Accuracies:")
    print("n\tModel 1\tModel 2")
    for i, n in enumerate(n_values):
        print(f"{n}\t{test_acc[i, 0]:.3f}\t{test_acc[i, 1]:.3f}")



if __name__ == "__main__":
	_testQ1d() 
	_plotCls()
	_testPCA()
	_plotKmeans()
	_plotKernelKmeans()
