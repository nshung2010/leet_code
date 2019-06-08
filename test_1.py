import numpy as np
a = [7, 3, 2, 4, 6]
b = [4, 2, 5, 6]
for _ in a+b:
    temp = max(a, b)
    print(temp)
#s = [max(a, b).pop(0) for _ in a+b]
# print(s)
a = np.array([[1, 2, 3], [2, 3, 4]])
print(a)
cal = np.sum(a, axis=0)
print(cal)
print(100*a/cal)
