# McCall Model - (Based  on QuantEcon Lectures)
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from quantecon.distributions import BetaBinomial

N = 50
# Grid for wages:
wmin, wmax = 10, 60
w = np.linspace(wmin, wmax, N+1)
# Use Beta-binomial distriution with parametera a, b:
a, b = 200, 100
q = BetaBinomial(N, a, b).pdf()

# Plot of pdf of wage outcomes:
#fig, ax = plt.subplots()
#ax.plot(w, q, '-o')
#ax.set_xlabel('Wages')
#ax.set_ylabel('Probabilities')
#plt.show()

# Other parameters:
rho = 0.99
c = 25
maxit = 200
tol = 1e-7

# Value Function Iteration:
def findv(w, rho, c, q):

    # Initial guess for value function:
    v0 = w/(1-rho)

    for it in range(maxit):
        v = np.maximum(w/(1-rho), c + rho * np.sum(q * v0))
        diff = np.max(abs(v - v0))
        if diff < tol:
            #print('Convergence OK')
            break
        v0 = v

    return v0

# Function to compute the reservation wage:
def res_wage(v, rho, c, q):
    return (1-rho)*(c + rho*np.sum(q*v))


# Comparative Statics: (change beta and c)
grid_size = 25
R = np.empty((grid_size, grid_size))

cs = np.linspace(10, 30, grid_size)
rhos = np.linspace(0.9, 0.99, grid_size)

for i, c in enumerate(cs):
    for j, b in enumerate(rhos):
        v = findv(w, b, c, q)
        R[i,j] = res_wage(v, b, c, q)

# Plot comparative statics:
fig, ax = plt.subplots()
cs1 = ax.contourf(cs, rhos, R.T, alpha = 0.8)
#ctr1 = ax.contour(cs, rhos, R.T)

#plt.clabel(ctr1, inline = 1, fontsize = 12)
plt.colorbar(cs1, ax = ax)

ax.set_title('Comparative Statics - Reservation Wages')
ax.set_xlabel('Unemployment Benefits - c')
ax.set_ylabel('Discount Rate')
plt.show()
