
import numpy as np

def simplex(A, b, base, nbase, currentIter=0):
    maxIter = len(b) + 1

    if currentIter >= maxIter:
        raise Exception("Max Iteration Reached")

    currentIter += 1

    cBT = np.array([variables[i] for i in base])
    cNT = np.array([variables[i] for i in nbase])

    baseMatrix = np.array([A[:, i] for i in base]).T
    nBaseMatrix = np.array([A[:, i] for i in nbase]).T

    baseInversed = np.linalg.inv(
        np.array([A[:, i] for i in base]).T
    )

    print("\n\n\n")
    print("Iteration:", currentIter)

    print("cBT:", cBT)
    print("cNT:", cNT)

    # Step 1: Calculate the current solution x
    x = baseInversed @ b
    print("Step 1 - Current Solution x:", x)

    # Step 2

    # Step 2.1: Calculate lambdaT
    lambdaT = cBT @ baseInversed
    print("Step 2.1 - LambdaT:", lambdaT)

    # Step 2.2: Calculate cN
    cN = np.array([cNT[i] - (lambdaT @ nBaseMatrix[:, i]) for i in range(len(nbase))])
    print("Step 2.2 - cN:", cN)

    # Step 2.3: Find the index of the minimum value of cN
    cNK = np.min(cN)
    k = np.argmin(cN)
    print("Step 2.3 - Minimum Value of cN:", cNK)
    print("Step 2.3 - Index k:", k)

    # Step 3: Check for optimality
    if np.all(cNK >= 0):
        print("Optimal Solution Found:")
        for i in range(len(base)):
            print("x", base[i] + 1, " = ", x[i])
        for i in range(len(nbase)):
            print("x", nbase[i] + 1, " = ", 0)
        return "Optimal"

    # Step 4: Calculate y
    y = baseInversed @ nBaseMatrix[:, k]
    print("Step 4 - Vector y:", y)

    # Step 5: Check for unboundedness
    if np.all(y <= 0):
        raise Exception("Unbounded")

    resultados = []

    for i in range(len(y)):
        if y[i] <= 0:
            resultados.append(np.inf)
        else:
            resultados.append(x[i] / y[i])

    print("Step 5 - Resultados:", resultados)
    print("Step 5 - Minimum Ratio:", min(resultados))

    E = resultados.index(min(resultados))

    print("Step 5 - Minimum Ratio Test:", E)

    # Step 6: Pivot the elements
    NbaseElement = nbase[k]
    BaseElement = base[E]

    np.put(base, [E], [NbaseElement])
    np.put(nbase, [k], [BaseElement])

    print("Step 6 - Pivot Elements:")
    print("Base:", base)
    print("Non-Base:", nbase)

    return simplex(A, b, base, nbase, currentIter)

A = np.array([
    [1, 2, -2, 1, 0, 0],
    [2, 0, -2, 0, 1, 0],
    [2, -1, 2, 0, 0, 1],
])

b = np.array([
    [4], 
    [6], 
    [2]
])

Base = np.array([3, 4, 5])

NBase = np.array([0, 1, 2])

variables = np.array([1, -2, 1, 0, 0, 0])

optimal = simplex(A, b, Base, NBase)
print(optimal)