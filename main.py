import matplotlib.pyplot as plt
import numpy as np
import math

def p(x):
    return math.cos(x)

def q(x):
    return math.sin(x) + 1

def f(x):
    return (2 + math.sin(x) + math.cos(x)) * math.exp(x) / 2

def DerivativeY(x, y, z):
    return z

def DerivativeZ(x, y, z):
    return f(x) - q(x)*y - p(x)*z

def NextY(x, y, z, step):
    return y + step * DerivativeY(x, y, z)

def NextZ(x, y, z, step):
    return z + step * DerivativeZ(x, y, z)

def EulerMethod(min, max, step, x0, y0, z0):
    currentPoint = (x0, y0, z0)
    massResult = []
    massResult.append(currentPoint)

    for x in np.arange(min+step, max, step):
        currentPoint = (x, NextY(x, currentPoint[1], currentPoint[2], step), NextZ(x, currentPoint[1], currentPoint[2], step))
        massResult.append(currentPoint)

    massX = []
    massY = []
    massZ = []

    for i in range(len(massResult)):
        massX.append(massResult[i][0])
        massY.append(massResult[i][1])
        massZ.append(massResult[i][2])

    return massX, massY, massZ

def DerivativeVector(x, vec):
    return np.array([vec[1], f(x) - q(x)*vec[0] - vec[1]*p(x)])

def NextRK(x, vec, step, massResult):
    k1 = DerivativeVector(x, vec)
    k2 = DerivativeVector(x + step/2, vec + step*k1/2)
    k3 = DerivativeVector(x + step/2, vec + step*k2/2)
    k4 = DerivativeVector(x + step, vec + step*k3)

    return vec + step/6 * (k1 + 2*k2 + 2*k3 + k4)

def NextAdams(x, vec, step, massResult):
    if abs(x) < step/4:
        return NextRK(x, vec, step, massResult)

    if abs(x - step) < step/4:
        return NextRK(x, vec, step, massResult)

    z = int(x/step + (0.5 if x/step > 0 else -0.5))
    return massResult[z][1] + step * (23 / 12 * DerivativeVector(x, massResult[z][1]) - 4 / 3 * DerivativeVector(x - step, massResult[z - 1][1]) +
                                     5 / 12 * DerivativeVector(x - 2 * step, massResult[z - 2][1]))

def Solve(Func, x0, vec0, step, min, max):
    currentPoint = (x0, vec0)
    massResult = []
    massResult.append(currentPoint)

    for x in np.arange(min + step, max, step):
        currentPoint = (x, Func(x-step, currentPoint[1], step, massResult))
        massResult.append(currentPoint)

    massX = []
    massY = []

    for i in range(len(massResult)):
        massX.append(massResult[i][0])
        massY.append(massResult[i][1][0])

    return massX, massY

def AmendmentRunge(x0, vec0, min, max, step, p, Func):
    massX, massY = Solve(Func, x0, vec0, step, min, max)
    massX, massY2 = Solve(Func, x0, vec0, step/2, min, max)

    massResult = []
    for i in range(len(massY)):
        massResult.append(2**p / (2**p - 1) * massY2[2*i] - massY[i] / (2**p - 1))
        massResult.append(massY2[i*2 + 1])

    return massX, massResult

def LogError(min, max, Func):
    step = 0.1
    massErr = []
    massStep = []
    while step > 1e-4:
        calculatedX, calculatedY = Solve(Func, min, np.array([1.5, 0.5]), step, min, max)
        massTrueY = [math.cos(i) + math.exp(i) / 2 for i in calculatedX]

        maxErr = 0
        for i in range(len(massTrueY)):
            if maxErr < abs(calculatedY[i] - massTrueY[i]):
                maxErr = abs(calculatedY[i] - massTrueY[i])

        massStep.append(math.log(step))
        massErr.append(math.log(maxErr))
        step = step / 10

    return massStep, massErr

def LogErrorAR(x0, vec0, min, max, p, Func):
    step = 0.1
    massErr = []
    massStep = []
    while step > 1e-4:
        calculatedX, calculatedY = AmendmentRunge(x0, vec0, min, max, step, p, Func)
        massTrueY = [math.cos(i) + math.exp(i) / 2 for i in calculatedX]

        maxErr = 0
        for i in range(len(massTrueY)):
            if maxErr < abs(calculatedY[i] - massTrueY[i]):
                maxErr = abs(calculatedY[i] - massTrueY[i])

        massStep.append(math.log(step))
        massErr.append(math.log(maxErr))
        step = step / 10

    return massStep, massErr

step = 0.0001
min = 0
max = 1
massXEM, massYEM, massZEM = EulerMethod(min, max, step, 0, 1.5, 0.5)
massXRK, massYRK = Solve(NextRK, min, np.array([1.5, 0.5]), step, min, max)
massXAdams, massYAdams = Solve(NextAdams, min, np.array([1.5, 0.5]), step, min, max)
massARX, massARY = AmendmentRunge(0, np.array([1.5, 0.5]), min, max, step, 4, NextRK)
massStep, massErr = LogError(min, max, NextRK)
massARStep, massARErr = LogErrorAR(0, np.array([1.5, 0.5]), min, max, 4, NextRK)
plt.title("19 задание")  # заголовок
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
# plt.plot(massXEM, massYEM, label='решение для у по методу Эйлера')
plt.plot(massXRK, massYRK, label='решение по методу Рунге-Кутта')
# plt.plot(massXAdams, massYAdams, label='решение по методу Адамса 3его порядка')
plt.plot(massXEM, [math.cos(i) + math.exp(i)/2 for i in massXEM], label='истинное решение')
plt.plot(massARX, massARY, label='поправка Рунге для метода Рунге-Кутта')
# plt.plot(massARStep, massARErr, label='функция логарифма ошибки')
# plt.plot(massStep, massErr, label='функция логарифма ошибки')
plt.legend()
plt.show()
