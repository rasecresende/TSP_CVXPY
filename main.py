import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# setting variables
A = np.array([[0,5,14,5,11,30,5,8,3],
              [5,0,14,3,13,30,8,4,5],
              [14,13,0,16,6,19,17,12,11],
              [5,3,17,0,15,33,7,7,7],
              [11,12,6,14,0,28,15,15,8],
              [30,30,20,33,29,0,34,26,27],
              [6,9,18,6,15,34,0,13,7],
              [9,5,11,8,16,26,12,0,7],
              [3,4,11,6,9,27,7,7,0]])


X = cp.Variable(A.shape, boolean = True)

array_ones = np.ones((len(A),1))

u = cp.Variable(len(A)) # auxiliary variable

#objective function
objective = cp.Minimize(cp.sum(cp.multiply(A,X)))


#constraints

constraints = [
               #constraints so that traveller must enter and leave exactly once
               X @ array_ones == array_ones,
               X.T @ array_ones == array_ones,
               cp.diag(X) == 0,
               u[0] == 1
]


# I just copied those constraints (no idea what they mean)
# means that the salesman only passes through each city once

for i in range(len(A)):
  constraints.append(u[i] >= 0)

#subtour elimination, this is the MTZ approach
#enforces that there is only a single tour covering all cities
for i in range(1, len(A)):
  for j in range(1, len(A)):
    if i != j:
      constraints.append(u[i] - u[j] + 1 <= (len(A) - 1) * (1 - X[i, j]))

# it forbids the number of selected arcs within the Matrix
# to be equal to or larger than the number of nodes in the Matrix


# Solving the problem
prob = cp.Problem(objective, constraints)
prob.solve()

print(prob.status)
print(prob.value)
print(X.value)
