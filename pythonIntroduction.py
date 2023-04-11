import numpy as np

# Data structures
# list
myList = ['hello', 123, 3.14]
myList[1] = True
print(myList)
print('last value: ', myList[-1])
print()

# loops
for i in myList:
    print(i)

for i in range(10):
    print(i)

print()

# conditions
# if
weather = "bad"
if weather == "good":
    print(True)
else:
    print(False)

# while
a = 0
while a < 100:
    a += 1
    if a > 50:
        break
    print(a)

print()


# functions
def square_me(x):
    return x ** 2


print(square_me(10))

# numpy
x_value = np.ones(10)
print(x_value)
print()

# functional programming
x_value = range(10)
y = [square_me(i) for i in x_value]
print(y)
print()

# numpy
x = [1, 2, 3, 1]
x = np.array(x)
print(type(x))
print(x)

w = np.array([0.5] * 4)
print(w)

a = np.dot(x, w)
print(a)

x = np.ones(10)
print(x)
x = np.zeros((2, 10))
print(x)
