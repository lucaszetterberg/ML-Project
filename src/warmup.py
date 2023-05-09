import matplotlib.pyplot as plt
import numpy as np
import pylab as pb
from math import pi
from scipy.spatial.distance import cdist


## Om man väljer punkt som inte finns, ger det udda resultat på fördelning över W

###########Task 1.1#######################################

mu = [0,0]
sigma = 0.2
alpha = 1/sigma


t = -1.25
x = 0.5
## visulaize prior distribution over W

def g(x, w0, w1):
    return w0* x + w1

K = np.eye(2) * alpha**-1

prior = np.random.multivariate_normal(mu, K, 10000)

pb.scatter(prior[:,0], prior[:,1], alpha = 0.2)

###########Task 1.2#######################################

## visulaize posterior distribution over W


likelihood = np.exp(-0.5 *((t - g(x, 0.5, -1.5))/sigma)**2)
posterior = likelihood * prior
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
print(w0[0])
print(w1[0]) 
x_axis = np.linspace(-1,1, 200)

y = []
plt.figure()
for i in range(5): 
    y = w0[i]*x_axis + w1[i]
    plt.plot(x_axis, y, label= f"w0 = {w0[i]}, w1 = {w1[i]}")



plt.legend(loc='upper center')
plt.show()
pb.show()

###########Task 1.4#######################################

## repeat 1.2 and 1.3 by addiing additional data points up to 7 data points



