import random
import numpy as np

## Multi-Linear Extension


## Key values
n = 5 # number of elements in set N
S_size = 2 # size of

N = [(i+1) for i in range(n)] # set N
S = random.sample(N, S_size) # subset S
print(S)

# define a function that returns all possible subsets of s
def powerset(s):
    x = len(s)
    power_set = []
    for i in range(1 << x):
       power_set.append([s[j] for j in range(x) if (i & (1 << j))])

    return power_set

## get all possible subsets / powerset of S
ps_S = powerset(S)
print(ps_S)
print(len(ps_S))

## get a random sample set from the power set of S
# def get_s_input(S):
#     ps_S = powerset(S)
#     return random.sample(ps_S, 1)[0]
# print(get_s_input(S))


## define a pseudo-function f: 2^S -> R

# The following f is computing the sum of all subsets,
# which means that the maximum of F will be reached when all xi are 1
def f(S):
    sum = 0
    ps_S = powerset(S)
    for list in ps_S:
        sum += np.sum(list)
    return sum

# print(f(S))

## function F: [0,1]^n -> R
## function f: 2^S -> R

## define an initial vector of x
# x = [np.random.choice([0,1], n)][0]
x = np.random.uniform(0,1, n)
print("x: ", x)


def F(x, f, N):
    # print("power set of N")
    ps_N = powerset(N)
    # print('ps_N: ', ps_N)

    sum = 0
    for S in ps_N:
        if S == []:
            # print(len(N))
            # print('x in the loop: ', len(x))
            x_not_S = [1 - x[i - 1] for i in N]
            sum += f(S) * np.prod(x_not_S)
        else:
            x_S = [x[i - 1] for i in S]
            not_S = list(set(N) - set(S))
            # print(x_S)
            # print(not_S)
            x_S_na = [x[j - 1] for j in not_S]
            sum += f(S) * np.prod(x_S) * np.prod(x_S_na)

    return sum
print('Sum: ', F(x, f, N))

import copy
def get_gradient_F(F, x, f, N, i):


    x_without_i = copy.deepcopy(x)
    x_without_i[i] = 0.0

    x_with_i = copy.deepcopy(x)
    x_with_i[i] = 1.0

    # print('x with xi: ', x_with_i)
    # print('x without xi: ', x_without_i)

    df_dxi = F(x_with_i, f, N) - F(x_without_i, f, N)
    return df_dxi

i = 3
print('gradient: ', get_gradient_F(F, x, f, N, i))

## Optimize the F function by first-order gradient ascent

x_init = copy.deepcopy(x)
sum_init = F(x, f, N)
print('x_init: ', x_init)
print('sum_init: ', sum_init)

# stepsize for gradient ascent
alpha = 0.01

def gradient_ascent(x_init, F, x, f, N, alpha):

    # key values to be used
    sum_update = 0
    iter = 0
    sum_temp = copy.deepcopy(sum_init)

    # start updating the parameters x with iterative gradients
    while np.abs(sum_temp - sum_update) > 10 ** (-2):
        iter += 1
        sum_temp = F(x, f, N)

        for i in range(n):
            grad_i = get_gradient_F(F, x, f, N, i)
            x[i] = np.minimum(x[i] + alpha * grad_i, 1.0)

        sum_update = F(x, f, N)

        # print('x updated: ', x)
        # print('sum updated: ', sum_update)

    return iter, sum_update, x


iter, sum_update, x_optimal = gradient_ascent(x_init, F, x, f, N, alpha)
print('Iterations: ', iter)
print('Initial F: ', sum_init)
print('Initial x: ', x_init)
print('Final F: ', sum_update)
print('Final x: ', x)


#
# # Run the function
# if __name__ == '__main__':
#     x_optimal = get_x()
#     print(x_optimial)