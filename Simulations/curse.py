import numpy as np 
import matplotlib.pyplot as plt
s1 = []
for n in range(1, 6):
    ans = 0
    for x1 in range(1, n+1):
        for x2 in range(1, n+1):
            for x3 in range(1, n+1):
                for x4 in range(1, n+1):
                    for x5 in range(1, n+1):
                        for x6 in range(1, n+1):
                            for x7 in range(1, n+1):
                                for x8 in range(1, n+1):
                                    for x9 in range(1, n+1):
                                        for x10 in range(1, n+1):
                                            ans += x1**2+x2**2+x3**2+x4**2+x5**2+x6**2+x7**2+x8**2+x9**2+x10**2
    ans = ans/((n+1)**2)
    s1.append(ans/(n**10))
s2 = []
for i in range(1,6):
    ans = 0
    simulate = np.random.rand(i**10, 10)
    for j in range(i**10):
        ans += np.dot(simulate[j],simulate[j])
    s2.append(ans/(i**10))

plt.plot([x**10 for x in range(1,6)], np.array(s1)-10/3, 'b', label='uniform')
plt.plot([x**10 for x in range(1,6)], np.array(s2)-10/3, 'r', label='stochastic')
plt.grid()
plt.xlabel('number of samples')
plt.ylabel('error')
plt.legend(loc='best')
plt.savefig("images/curse of dim/uniform vs stochastic.png")


