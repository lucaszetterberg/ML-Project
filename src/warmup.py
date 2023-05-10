import matplotlib.pyplot as plt
import numpy as np
import pylab as pb
from math import pi
from scipy.spatial.distance import cdist
import statsmodels.api as sm
import math


## Om man väljer punkt som inte finns, ger det udda resultat på fördelning över W

###########Task 1.1#######################################

mu = [0,0]
sigma = 0.2
alpha = 1/sigma


data_t = [-2, -1.995, -1.99, -1, -1.005, -1.5, -1.125]
data_x = [-1, -0.99, -0.98, 1, 0.99, 0, 0.75]
t = -0.5
x = 1
w0 = 0.5
w1 = -1.5
## visulaize prior distribution over W

def g(x, w0, w1):
    return w0* x + w1

K = np.identity(2) * alpha**-1

prior = np.random.multivariate_normal(mu, K, 10000)

pb.scatter(prior[:,0], prior[:,1], alpha = 0.2)

###########Task 1.2#######################################

## visulaize posterior distribution over W

def likelihood(t, x, w0, w1):
    return np.exp(-0.5 *((t - g(x, w0, w1))/sigma)**2)


def posterior(data_t, data_x, w0, w1, prior):
    posterior = likelihood(data_t[0], data_x[0], w0, w1) * prior
    for i in range(len(data_t)):
        if i > 0:
            posterior *= likelihood(data_t[i], data_x[i], w0, w1)
    return posterior

##likelihood = likelihood(-0.5, 1, w0, w1)

#likelihood = np.exp(-0.5 *((t - g(x, 0.5, -1.5))/sigma)**2)

posterior = posterior(data_t, data_x, w0, w1, prior) #likelihood(data_t[0], data_x[0], w0, w1) * prior

pb.figure()
pb.scatter(posterior[:,0], posterior[:,1], alpha = 0.2)
pb.xlabel('x')
pb.ylabel('y')
pb.title('Posterior distribution over W')


###########Task 1.3#######################################

## take 5 random samples from the posterior distribution
five_samples = []
for i in range(5):
    random = np.random.randint(0,10000)
    five_samples.append(posterior[random])

w0 = []
w1 = []
for i in five_samples:
    w0.append(i[0])
    w1.append(i[1])

## print(five_samples) 
#print(w0[0])
#print(w1[0]) 
x_axis = np.linspace(-1,1, 200)

y = []
plt.figure()
for i in range(5): 
    y = w0[i]*x_axis + w1[i]
    plt.plot(x_axis, y, label= f"w0 = {w0[i]}, w1 = {w1[i]}")



plt.legend(loc='upper center')
plt.show()


###########Task 2#######################################


###########Task 2.1#######################################


x1 = np.linspace(-1,1, 1000)
x2 = np.linspace(-2,2, 1000)

w = [1, 2]

t = w[0]*x1 + w[1]*x2

beta = 1/alpha
## calculate the log likelihood for the data set


def log_likelihood(t, x1, x2, w0, w1, beta):
    return -(beta/2 * sum(t - w0*x1 + w1*x2)) + len(x1)/2 * math.log(beta) - len(x1)/2 * math.log(2*pi)


print(log_likelihood(t, x1, x2, 1, 2, beta))

#-2070.2310797016953


## function that calculate the sum of the error


sum_error_list = []

def sum_error(w_test):
    return sum((t - w_test[0]*x1 + w_test[1]*x2)**2)

for i in range(len(prior)):
    sum_error_list.append(sum_error(prior[i]))
    

print(len(sum_error_list))
print(min(sum_error_list))

print(sum_error_list.index(min(sum_error_list)))
##969

print(prior[sum_error_list.index(min(sum_error_list))])



pb.figure()
#pb.scatter(post_new[:,0], post_new[:,1], alpha = 0.2)
pb.xlabel('x')
pb.ylabel('y')
pb.title('Posterior distribution over W')

pb.show()

#xi = [x1, x2]

#W = [w0, w1]

#ti = x1 * w0 + x2 * w1 + epsilon


