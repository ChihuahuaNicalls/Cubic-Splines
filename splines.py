import numpy as np
import matplotlib.pyplot as plt

def coefficientsSplines(xp, fp, dp):
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


def getPolynomsSplines(xp, fp, dp):
    Q, z = coefficientsSplines(xp, fp, dp)
    
    coefficientsP = []
    for i in range(len(xp) - 1):
        b0 = Q[2 * i][0]
        b1 = Q[2 * i][1]
        b2 = Q[2 * i][2]
        b3 = Q[2 * i][3]
        
        coefficientsP.append((b0, b1, b2, b3))

        print(f"---")
        print(f"P{i}(x) en [{xp[i]:.4f}, {xp[i+1]:.4f}]:")
        
        b1Str = f"{b1:.4f}(x - {xp[i]:.4f})"
        b2Str = f"{b2:.4f}(x - {xp[i]:.4f})^2"
        b3Str = f"{b3:.4f}(x - {xp[i]:.4f})^2(x - {xp[i+1]:.4f})"
        print(f"P{i}(x) = {b0:.4f} + {b1Str} + {b2Str} + {b3Str}")

        A = b3
        B = b2 - 2 * b3 * xp[i] - b3 * xp[i+1]
        C = b1 - 2 * b2 * xp[i] + b3 * xp[i]**2 + 2 * b3 * xp[i] * xp[i+1]
        D = b0 - b1 * xp[i] + b2 * xp[i]**2 - b3 * xp[i]**2 * xp[i+1]

        simplified_poly_parts = []
        if abs(A) > 1e-9:
            simplified_poly_parts.append(f"{A:.4f}x^3")
        if abs(B) > 1e-9:
            simplified_poly_parts.append(f"{B:+.4f}x^2")
        if abs(C) > 1e-9:
            simplified_poly_parts.append(f"{C:+.4f}x")
        if abs(D) > 1e-9:
            simplified_poly_parts.append(f"{D:+.4f}")
        
        simplified_poly_str = "".join(simplified_poly_parts)
        if simplified_poly_str.startswith("+"):
            simplified_poly_str = simplified_poly_str[1:]
        elif not simplified_poly_str:
            simplified_poly_str = "0.0000"

        print(f"P{i}(x) (Simplificado) = {simplified_poly_str}\n")
        
    return coefficientsP

def getPolynomHermite(xp, fp, dp):
    n = len(xp)
    if not (len(fp) == n and len(dp) == n):
        raise ValueError("Todos los arreglos deben tener la misma cantidad de datos")

    z = np.empty(2 * n)
    z[0::2] = xp
    z[1::2] = xp

    Q = np.zeros((2 * n, 2 * n))

    Q[0::2, 0] = fp
    Q[1::2, 0] = fp

    for i in range(n):
        Q[2 * i, 1] = dp[i]
        if i < n - 1:
            Q[2 * i + 1, 1] = (Q[2 * i + 2, 0] - Q[2 * i + 1, 0]) / (z[2 * i + 2] - z[2 * i + 1])

    for k in range(2, 2 * n):
        for i in range(2 * n - k):
            Q[i, k] = (Q[i + 1, k - 1] - Q[i, k - 1]) / (z[i + k] - z[i])
            
    coeffHermite = [Q[i, i] for i in range(2 * n)]
    
    def evaluateHermite(x_val):
        poly_sum = 0.0
        for i in range(2 * n):
            term = coeffHermite[i]
            for j in range(i):
                term *= (x_val - z[j])
            poly_sum += term
        return poly_sum
    
    return evaluateHermite, coeffHermite, z

def printHermite(coeffHermite, z_nodes):
    n = len(coeffHermite)
    
    poly_terms = []
    for i in range(n):
        term = f"{coeffHermite[i]:.4f}"
        if i > 0:
            for j in range(i):
                term += f"*(x - {z_nodes[j]:.4f})"
        poly_terms.append(term)
    poly_str_extended = " + ".join(poly_terms)
    poly_str_extended = poly_str_extended.replace("+ -", "- ")
    print(f"P(x) (Extendido) = {poly_str_extended}")

    hermite_poly_np = np.poly1d([0.0])
    
    for i in range(n):
        current_term_poly = np.poly1d([coeffHermite[i]])
        
        for j in range(i):
            factor = np.poly1d([1, -z_nodes[j]])
            current_term_poly = np.poly1d(np.polymul(current_term_poly.coeffs, factor.coeffs))
            
        hermite_poly_np = np.poly1d(np.polyadd(hermite_poly_np.coeffs, current_term_poly.coeffs))

    coeffs = hermite_poly_np.coeffs
    simplified_poly_parts = []
    for i, coeff in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        if abs(coeff) > 1e-9:
            term_str = f"{coeff:+.4f}"
            if power > 1:
                term_str += f"x^{power}"
            elif power == 1:
                term_str += "x"
            simplified_poly_parts.append(term_str)
            
    simplified_poly_str = "".join(simplified_poly_parts)
    if simplified_poly_str.startswith("+"):
        simplified_poly_str = simplified_poly_str[1:]
    elif not simplified_poly_str:
        simplified_poly_str = "0.0000"

    print(f"P(x) (Simplificado) = {simplified_poly_str}\n")


def evaluatePolynomsSplines(valX, xi, xiAux, coefficients):
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
            print("Entrada invalida. Por favor, ingresa un número.")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue

    if len(xp) < 2:
        print("Mínimo 2 puntos son necesarios para la interpolación.")
        return

    try:
        print("Calculando Splines Cúbicas...")
        pCoefficientsSplines = getPolynomsSplines(xp, fp, dp)
        print("\nCalculando Polinomio de Hermite...")
        hermite_poly_func, hermite_coeffs, z_nodes = getPolynomHermite(xp, fp, dp)
        printHermite(hermite_coeffs, z_nodes)

    except ValueError as e:
        print(f"Error: {e}")
        return

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].plot(xp, fp, 'o', color='red', label='Puntos Originales')

    for i in range(len(xp) - 1):
        xIni = xp[i]
        xEnd = xp[i+1]
        xInterval = np.linspace(xIni, xEnd, 100)
        yInterval = [evaluatePolynomsSplines(x, xIni, xEnd, pCoefficientsSplines[i]) for x in xInterval]
        
        axs[0].plot(xInterval, yInterval, label=f'Spline P{i}(x) en [{xIni:.2f}, {xEnd:.2f}]', linestyle='--', linewidth=2)

    axs[0].set_title('Interpolación con Splines Cúbicas')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('f(x)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    axs[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    axs[1].plot(xp, fp, 'o', color='red', label='Puntos Originales')
    x_hermite = np.linspace(min(xp) - 0.5, max(xp) + 0.5, 500)
    y_hermite = [hermite_poly_func(x) for x in x_hermite]
    axs[1].plot(x_hermite, y_hermite, label='Polinomio de Hermite', color='blue', linewidth=2)

    axs[1].set_title('Interpolación con Polinomio de Hermite')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('f(x)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    axs[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inputs()