import numpy as np
import matplotlib.pyplot as plt

def coefficients(xp, fp, dp):
    n = len(xp)
    if not (len(fp) == n and len(dp) == n):
        raise ValueError("Todos los arreglos deben tener la misma cantidad de datos")

    z = []
    for i in range(n):
        z.append(xp[i])
        z.append(xp[i])

    Q = []
    for i in range(2 * n):
        Q.append([0.0] * (2 * n))
    
    for i in range(n):
        Q[2 * i][0] = fp[i]
        Q[2 * i + 1][0] = fp[i]

    for i in range(n):
        Q[2 * i][1] = dp[i]
        if i < n - 1:
            Q[2 * i + 1][1] = (Q[2 * i + 2][0] - Q[2 * i + 1][0]) / (z[2 * i + 2] - z[2 * i + 1])

    for k in range(2, 2 * n):
        for i in range(2 * n - k):
            Q[i][k] = (Q[i + 1][k - 1] - Q[i][k - 1]) / (z[i + k] - z[i])
    
    return Q, z

def getPolynoms(xp, fp, dp):
    Q, z = coefficients(xp, fp, dp)
    
    coefficientsP = []
    for i in range(len(xp) - 1):
        b0 = Q[2 * i][0]
        b1 = Q[2 * i][1]
        b2 = Q[2 * i][2]
        b3 = Q[2 * i][3]
        
        coefficientsP.append((b0, b1, b2, b3))

        print(f"P{i}(x) en [{xp[i]:.2f}, {xp[i+1]:.2f}]:")
        b1Str = f"{b1:.4f}(x - {xp[i]:.4f})"
        b2Str = f"{b2:.4f}(x - {xp[i]:.4f})^2"
        b3Str = f"{b3:.4f}(x - {xp[i]:.4f})^2(x - {xp[i+1]:.4f})"

        print(f"P{i}(x) = {b0:.4f} + {b1Str} + {b2Str} + {b3Str}\n")
        
    return coefficientsP

def evaluatePolynoms(valX, xi, xiAux, coefficients):
    b0, b1, b2, b3 = coefficients
    term1 = b0
    term2 = b1 * (valX - xi)
    term3 = b2 * (valX - xi)**2
    term4 = b3 * (valX - xi)**2 * (valX - xiAux)
    return term1 + term2 + term3 + term4

def inputs():
    xp = []
    fp = []
    dp = []
    numPuntos = 0

    while numPuntos < 10:
        try:
            xInp = input(f"Ingresa el valor de x para el punto {numPuntos + 1} (o '.' para terminar): ")
            if xInp == '.':
                break
            xVal = float(xInp)

            fInp = input(f"Ingresa el valor de f(x) para el punto {numPuntos + 1}: ")
            fVal = float(fInp)

            dInp = input(f"Ingresa el valor de f'(x) (derivada) para el punto {numPuntos + 1}: ")
            dVal = float(dInp)

            xp.append(xVal)
            fp.append(fVal)
            dp.append(dVal)
            numPuntos += 1
            print("\n\n")

        except ValueError:
            print("Entrada invalida")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue

    if len(xp) < 2:
        print("Minimo 2 puntos")
        return

    try:
        pCoefficients = getPolynoms(xp, fp, dp)
    except ValueError as e:
        print(f"Error: {e}")
        return

    plt.figure(figsize=(10, 6))
 
    plt.plot(xp, fp, 'o', color='red', label='Puntos')

    for i in range(len(xp) - 1):
        xIni = xp[i]
        xEnd = xp[i+1]
        xInterval = np.linspace(xIni, xEnd, 100) 
        yInterval = [evaluatePolynoms(x, xIni, xEnd, pCoefficients[i]) for x in xInterval]
        
        plt.plot(xInterval, yInterval, label=f'P{i}(x) en [{xIni:.2f}, {xEnd:.2f}]', linewidth=2)

    plt.title('Polinomios de Hermite')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    plt.show()

if __name__ == "__main__":
    inputs()