import math

b = 2

a = [[3, 2, 0], [2, 0, 3], [0, 3, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
n = sum([sum([a[i][j] for j in range(0, len(a[i]))]) for i in range(0, len(a))])
s = len(a)
k = len(a[0])
print a, n, s, k

h_s = -sum([sum([a[i][j] for j in range(0, k)])/float(n) *
            math.log(sum([a[i][j] for j in range(0, k)])/float(n), b)
            for i in range(0, s)])

h_s_k = -sum([sum([a[i][j]/float(n) * math.log(float(a[i][j])/sum([a[ii][j] for ii in range(0, s)]), b)
                   if a[i][j] > 0 else 0
                   for i in range(0, s) for j in range(0, k)])])

h_k = -sum([sum([a[i][j] for i in range(0, s)])/float(n) *
            math.log(sum([a[i][j] for i in range(0, s)])/float(n), b)
            for j in range(0, k)])

h_k_s = -sum([sum([a[i][j]/float(n) * math.log(float(a[i][j])/sum([a[i][jj] for jj in range(0, k)]), b)
                   if a[i][j] > 0 else 0
                   for j in range(0, k) for i in range(0, s)])])

h = 1 - h_s_k / h_s
c = 1 - h_k_s / h_k
v = (2 * h * c) / (h + c)

print h_s, h_s_k, h_k, h_k_s, h, c, v
