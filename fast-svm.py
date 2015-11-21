#!/usr/bin/env python
#summary: implementation of Fast training of Support Vector Machines with Gaussian kernel -- paper by Matteo Fischetti
#code author: Daniel Lemes Gribel -- dgribel@inf.puc-rio.br

import numpy as np
import random
import time
import math
from scipy.spatial import distance

def euclidean_distance(a, b):
	return distance.euclidean(a, b)

def linear_kernel(a, b):
	product = 0.0
	for i in range(0, len(a)):
		product = product + a[i]*b[i]
	
	return product

def gaussian_kernel(a, b, gamma):
	d = euclidean_distance(a, b)
	x = gamma*d*d
	x = -1.0*x

	return math.exp(x)

def loo1_error_estimate(z, p):
	s = 0
	for j in range(0, p):
		s = s + z[j];
	
	s = (1.0*s)/p
	
	return s

def miss_classification(y, label):
	cont = 0
	for i in range(0, len(label)):
		if y[i] != label[i]:
			cont = cont+1

	return (1.0*cont)/len(label)

def loo1(x, y, gamma, epslon):
	p = len(x)
	z = np.zeros(p)
	sum = np.zeros(p)

	for j in range(0, p):
		for i in range(0, p):
			if i != j:
				sum[j] = sum[j] + y[i]*gaussian_kernel(x[j], x[i], gamma)

		if y[j]*sum[j] >= epslon:
			z[j] = 0
		else:
			z[j] = 1

	return loo1_error_estimate(z, p)

def fast_leave_one_out(x, y, gamma, alpha):
	p = len(x)
	v = np.zeros(p)
	n_ok = 0

	for j in range(0, p):
		if y[j] == -1:
			n_ok = n_ok + 1

		for i in range(0, p):
			if i != j:
				v[j] = v[j] + (alpha * y[i] * gaussian_kernel(x[j], x[i], gamma))

	sigma = sort_index(v)
	beta_0 = float("-inf")
	n_ok_best = n_ok

	for k in range(0, p):
		j = sigma[p-k-1]

		if y[j] == -1:
			n_ok = n_ok - 1
		else:
			n_ok = n_ok + 1
			if n_ok > n_ok_best:
				n_ok_best = n_ok
				beta_0 = -1.0 * v[j]

	print 'loo2 =', (1.0*(p - n_ok_best))/p

	return beta_0

def normalize_feature(train_set, f, avg, std):
	for i in range(0, len(train_set)):
		if std != 0:
			train_set[i][f] = (train_set[i][f]-avg)/std
		else:
			train_set[i][f] = 0

	return train_set[:,f]

def svm_prediction(test, x, y, gamma, alpha, beta_0):
	test_size = len(test)
	train_size = len(x)

	ytest = np.zeros(test_size)
	sum = np.zeros(test_size)

	for t in range(0, test_size):
		for i in range(0, train_size):
			g = gaussian_kernel(test[t], x[i], gamma)
			sum[t] = sum[t] + ((alpha * y[i] * g))

		if sum[t] >= (-1.0*beta_0):
			ytest[t] = 1
		else:
			ytest[t] = -1

	return ytest

def sort_index(my_list):
	s = [i[0] for i in sorted(enumerate(my_list), key=lambda x:x[1])]
	return s

# calculate loo1 (error estimation)
def run_loo1(x, y, gamma, epslon):
	s = loo1(x, y, gamma, epslon)
	print 'loo1 =', s

	return s

# return beta_0 value
def run_loo2(x, y, gamma, alpha):
	beta_0 = fast_leave_one_out(x, y, gamma, alpha)
	print 'beta_0 =', beta_0

	return beta_0

def optimize_gamma(x, y, alpha):
	gamma_l = 0.01
	gamma_r = 1
	gamma_c = (1.0*(gamma_l + gamma_r))/2.0

	while((gamma_r - gamma_l) > 0.01):
		if abs(gamma_l-gamma_c) <= abs(gamma_r-gamma_c):
			gamma_r = gamma_c
		else:
			gamma_l = gamma_c

		gamma_c = (1.0*(gamma_l + gamma_r))/2.0
		
	return gamma_c

def demo():
	#random.seed(1607)
	mydata = np.genfromtxt('dataset/train12.csv', delimiter = ",")
	
	N = len(mydata) # number of instances
	NUM_ATTRIBUTES = 10
	NUM_POINTS = 4320
	NUM_TEST = int(0.3*NUM_POINTS)

	test_r = random.sample(range(0, NUM_POINTS), NUM_TEST)

	k_test = 0
	k_train = 0

	test_set = np.zeros((NUM_TEST, NUM_ATTRIBUTES+1))
	train_set = np.zeros((NUM_POINTS-NUM_TEST, NUM_ATTRIBUTES+1))

	for i in range(0, N):
		if i in test_r:
			test_set[k_test] = mydata[i]
			k_test = k_test+1
		else:
			train_set[k_train] = mydata[i]
			k_train = k_train+1

	#train_set = mydata[0:483]
	#test_set = mydata[483:]

	label_train = np.zeros(len(train_set))
	label_test = np.zeros(len(test_set))

	for i in range(0, len(train_set)):
		label_train[i] = int(train_set[i][NUM_ATTRIBUTES])
		if label_train[i] == 2:
			label_train[i] = -1

	for i in range(0, len(test_set)):
		label_test[i] = int(test_set[i][NUM_ATTRIBUTES])
		if label_test[i] == 2:
			label_test[i] = -1

	train_set = train_set[:, :-1]
	test_set = test_set[:, :-1]

	avg_train = np.zeros(NUM_ATTRIBUTES)
	std_train = np.zeros(NUM_ATTRIBUTES)

	for i in range(0, NUM_ATTRIBUTES):
		avg_train[i] = np.average(train_set[:,i])
		std_train[i] = np.std(train_set[:,i])

	for i in range(0, NUM_ATTRIBUTES):
		train_set[:,i] = normalize_feature(train_set, i, avg_train[i], std_train[i])
		test_set[:,i] = normalize_feature(test_set, i, avg_train[i], std_train[i])
	
	epslon = pow(10, -5)
	alpha = 1
	beta_0 = 0

	#while(gamma < 0.25):
	#	gamma = gamma + 0.005
	#	z = loo1(train_set, label_train, gamma, epslon)
	#	s = loo1_error_estimate(z, len(train_set))
	#	print gamma,s

	gamma = optimize_gamma(train_set, label_train, alpha)
	print 'gamma =', gamma

	#s = run_loo1(train_set, label_train, gamma, epslon)
	y1 = svm_prediction(test_set, train_set, label_train, gamma, alpha, beta_0)
	m1 = miss_classification(y1, label_test)
	print 'mis1 =', m1

	#beta_0 = run_loo2(train_set, label_train, gamma, alpha)
	#y2 = svm_prediction(test_set, train_set, label_train, gamma, alpha, beta_0)
	#m2 = miss_classification(y2, label_test)
	#print 'mis2 =', m2

	print 'time:', time.time() - start

if __name__ == "__main__":
	start = time.time()
	demo()