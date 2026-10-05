import json
import math
import cmath

def matrE(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
def mul(A, B):
    n, p, m = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)]
            for i in range(n)]
def printmatr(name, M):
    print(f"{name}:")
    for row in M:
        print("  [" + ", ".join(f"{v:.6g}" for v in row) + "]")
def printvectr(name, v):
    print(f"{name}: [" + ", ".join(f"{x:.6g}" for x in v) + "]")
def equalmatr(A, B, eps=1e-8):
    return all(abs(A[i][j] - B[i][j]) < eps
               for i in range(len(A)) for j in range(len(A[0])))
def lu_decompose(A):
    n = len(A)
    U = [row[:] for row in A]
    L = matrE(n)
    P = matrE(n)
    swaps = 0
    for k in range(n):
        oporelemnt = max(range(k, n), key=lambda i: abs(U[i][k]))
        if abs(U[oporelemnt][k]) < 1e-15:
            raise ValueError("Матрица вырождена")
        if oporelemnt != k:
            U[k], U[oporelemnt] = U[oporelemnt], U[k]
            P[k], P[oporelemnt] = P[oporelemnt], P[k]
            for j in range(k):
                L[k][j], L[oporelemnt][j] = L[oporelemnt][j], L[k][j]
            swaps += 1
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
    return P, L, U, swaps
def lu_solve(P, L, U, b):
    n = len(L)
    pb = [sum(P[i][j] * b[j] for j in range(n)) for i in range(n)]
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(L[i][j] * y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x
def lu_inverse(P, L, U):
    n = len(L)
    inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        col = lu_solve(P, L, U, e)
        for i in range(n):
            inv[i][j] = col[i]
    return inv
def lu_det(U, swaps):
    det = (-1.0) ** swaps
    for i in range(len(U)):
        det *= U[i][i]
    return det
def lab1_1(data):
    print("1.1 LU")
    A, b = data["A"], data["b"]
    printmatr("Матрица A", A)
    printvectr("Вектор b", b)
    P, L, U, swaps = lu_decompose(A)
    print()
    printmatr("Матрица L", L)
    printmatr("Матрица U", U)
    printmatr("L * U", mul(L, U))
    x = lu_solve(P, L, U, b)
    print()
    printvectr("Решение x", x)

    inv = lu_inverse(P, L, U)
    printmatr("Обратная матрица A^-1", inv)

    print(f"Определитель det(A) = {lu_det(U, swaps):.10g}")

    print()
    prod = mul(A, inv)
    printmatr("Проверка A * A^-1", prod)
    print("  A * A^-1 ≈ E :", equalmatr(prod, matrE(len(A))))

    lu = mul(L, U)
    pa = mul(P, A)
    print("  L * U ≈ P * A :", equalmatr(lu, pa))
def progonka(lower, diag, upper, b):
    n = len(diag)
    alpha = [0.0] * n
    beta = [0.0] * n
    alpha[0] = -upper[0] / diag[0] if n > 1 else 0.0
    beta[0] = b[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] + lower[i - 1] * alpha[i - 1]
        alpha[i] = -upper[i] / denom if i < n - 1 else 0.0
        beta[i] = (b[i] - lower[i - 1] * beta[i - 1]) / denom
    x = [0.0] * n
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return alpha, beta, x


def task_1_2(data):
    print("1.2 прогонки")
    printvectr("Поддиагональ lower", data["lower"])
    printvectr("Главная диагональ diag", data["diag"])
    printvectr("Наддиагональ upper", data["upper"])
    printvectr("Правая часть b", data["rhs"])
    alpha, beta, x = progonka(data["lower"], data["diag"],data["upper"], data["rhs"])
    print()
    printvectr("Прогоночные коэффициенты alpha", alpha)
    printvectr("Прогоночные коэффициенты beta", beta)
    printvectr("Решение x", x)
def iteration(A, b, eps, max_iter=10000):
    n = len(A)
    x = [0.0] * n
    for k in range(1, max_iter + 1):
        xn = [x[i] + (b[i] - sum(A[i][j] * x[j] for j in range(n))) / A[i][i]
              for i in range(n)]
        if max(abs(xn[i] - x[i]) for i in range(n)) < eps:
            return xn, k
        x = xn
    return x, max_iter
def zaydel(A, b, eps, max_iter=10000):
    n = len(A)
    x = [0.0] * n
    for k in range(1, max_iter + 1):
        x_old = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        if max(abs(x[i] - x_old[i]) for i in range(n)) < eps:
            return x, k
    return x, max_iter
def task_1_3(data):
    print("1.3  итерации и метод Зейделя")
    A, b, eps = data["A"], data["b"], data["eps"]
    printmatr("Матрица A", A)
    printvectr("Вектор b", b)
    print(f"Заданная точность eps = {eps}")
    P, L, U, _ = lu_decompose(A)
    exact = lu_solve(P, L, U, b)
    printvectr("Эталонное решение", exact)
    x_si, k_si = iteration(A, b, eps)
    x_se, k_se = zaydel(A, b, eps)
    print()
    printvectr("Простые итерации: x", x_si)
    print(f"  количество итераций: {k_si}")
    printvectr("Зейдель: x", x_se)
    print(f"  количество итераций: {k_se}")
    print()
    if k_si < k_se:
        ratio = k_se / k_si
        print(f"Метод простых итераций сошёлся быстрее в {ratio:.2f} раз ")
    elif k_se < k_si:
        ratio = k_si / k_se
        print(f"Метод Зейделя сошёлся быстрее в {ratio:.2f} раз ")
    else:
        print("Оба метода сошлись за одинаковое число итераций.")
def jacobi(A, eps, max_iter=1000):
    n = len(A)
    A = [row[:] for row in A]
    V = matrE(n)
    iterations = 0
    for sweep in range(max_iter):
        iterations = sweep + 1
        flag = math.sqrt(sum(A[i][j] ** 2 for i in range(n)for j in range(n) if i != j))
        if flag < eps:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-15:
                    continue
                tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
                if tau >= 0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                app, aqq, apq = A[p][p], A[q][q], A[p][q]
                A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
                A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
                A[p][q] = A[q][p] = 0.0
                for i in range(n):
                    if i != p and i != q:
                        aip, aiq = A[i][p], A[i][q]
                        A[i][p] = A[p][i] = c * aip - s * aiq
                        A[i][q] = A[q][i] = s * aip + c * aiq
                for i in range(n):
                    vip, viq = V[i][p], V[i][q]
                    V[i][p] = c * vip - s * viq
                    V[i][q] = s * vip + c * viq
    vectorlambda = [A[i][i] for i in range(n)]
    return vectorlambda, V, iterations
def task_1_4(data):
    print("1.4  Метод вращений (Якоби)")
    A, eps = data["A"], data["eps"]
    printmatr("Матрица A", A)
    print(f"Заданная точность eps = {eps}")
    vectorlambda, V, iters = jacobi(A, eps)
    n = len(vectorlambda)
    Lambda = [[vectorlambda[i] if i == j else 0.0 for j in range(n)]for i in range(n)]
    print()
    printvectr("Собственные значения", vectorlambda)
    printmatr("Матрица собственных векторов V", V)
    printmatr("Диагональная матрица Lambda", Lambda)
    AV = mul(A, V)
    VLambda = mul(V, Lambda)
    print()
    print("  A * V ≈ V * Lambda :", equalmatr(AV, VLambda))
    print(f"  число проходов: {iters}")
def qr_householder(A):
    m, n = len(A), len(A[0])
    R = [row[:] for row in A]
    Q = matrE(m)
    for k in range(min(m, n)):
        x = [R[i][k] for i in range(k, m)]
        norm = math.sqrt(sum(v * v for v in x))
        if norm < 1e-15:
            continue
        alpha = -math.copysign(norm, x[0]) if x[0] != 0 else -norm
        v = x[:]
        v[0] -= alpha
        vnorm2 = sum(vi * vi for vi in v)
        if vnorm2 < 1e-15:
            continue
        for j in range(k, n):
            s = sum(v[i] * R[k + i][j] for i in range(len(v)))
            coeff = 2.0 * s / vnorm2
            for i in range(len(v)):
                R[k + i][j] -= coeff * v[i]
        for i in range(m):
            s = sum(Q[i][k + j] * v[j] for j in range(len(v)))
            coeff = 2.0 * s / vnorm2
            for j in range(len(v)):
                Q[i][k + j] -= coeff * v[j]
    return Q, R
def qr_complex(A):
    m, n = len(A), len(A[0])
    Q = [[0j] * n for _ in range(m)]
    R = [[0j] * n for _ in range(n)]
    for j in range(n):
        v = [complex(A[i][j]) for i in range(m)]
        for i in range(j):
            dot = sum(Q[k][i].conjugate() * v[k] for k in range(m))
            R[i][j] = dot
            for k in range(m):
                v[k] -= dot * Q[k][i]
        norm = math.sqrt(sum(abs(v[k]) ** 2 for k in range(m)))
        if norm < 1e-15:
            for cand in range(m):
                w = [0j] * m
                w[cand] = 1.0 + 0j
                for i in range(j):
                    d = sum(Q[k][i].conjugate() * w[k] for k in range(m))
                    for k in range(m):
                        w[k] -= d * Q[k][i]
                nw = math.sqrt(sum(abs(w[k]) ** 2 for k in range(m)))
                if nw > 1e-12:
                    for k in range(m):
                        Q[k][j] = w[k] / nw
                    break
        else:
            R[j][j] = norm
            for k in range(m):
                Q[k][j] = v[k] / norm
    return Q, R
def qr_eigenvalues(A, eps=1e-10, max_iter=1000):
    n = len(A)
    A = [[complex(A[i][j]) for j in range(n)] for i in range(n)]
    vectorlambda = []
    m = n
    it = 0
    while m > 0:
        if m == 1:
            vectorlambda.append(A[0][0])
            break
        if m == 2:
            a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
            tr, dt = a + d, a * d - b * c
            disc = cmath.sqrt(tr * tr - 4 * dt)
            vectorlambda += [(tr + disc) / 2, (tr - disc) / 2]
            break
        if abs(A[m - 1][m - 2]) < eps:
            vectorlambda.append(A[m - 1][m - 1])
            m -= 1
            A = [row[:m] for row in A[:m]]
            continue
        if it > max_iter:
            i = 0
            while i < m:
                if i < m - 1 and abs(A[i + 1][i]) > eps:
                    a, b = A[i][i], A[i][i + 1]
                    c, d = A[i + 1][i], A[i + 1][i + 1]
                    tr, dt = a + d, a * d - b * c
                    disc = cmath.sqrt(tr * tr - 4 * dt)
                    vectorlambda += [(tr + disc) / 2, (tr - disc) / 2]
                    i += 2
                else:
                    vectorlambda.append(A[i][i])
                    i += 1
            break

        a, b = A[m - 2][m - 2], A[m - 2][m - 1]
        c, d = A[m - 1][m - 2], A[m - 1][m - 1]
        tr, dt = a + d, a * d - b * c
        disc = cmath.sqrt(tr * tr - 4 * dt)
        mu1, mu2 = (tr + disc) / 2, (tr - disc) / 2
        mu = mu1 if abs(mu1 - d) < abs(mu2 - d) else mu2
        S = [[A[i][j] - (mu if i == j else 0) for j in range(m)]
             for i in range(m)]
        Q, R = qr_complex(S)
        A = mul(R, Q)
        for i in range(m):
            A[i][i] += mu
        it += 1
    return vectorlambda, it
def task_1_5(data):
    print("1.5  QR-разложение и QR-алгоритм")
    A, eps = data["A"], data["eps"]
    printmatr("Исходная матрица A", A)
    Q, R = qr_householder(A)
    print()
    printmatr("Матрица Q", Q)
    printmatr("Матрица R", R)
    printmatr("Q * R", mul(Q, R))
    print("  Q * R ≈ A :", equalmatr(mul(Q, R), A))
    vectorlambda, iters = qr_eigenvalues(A, eps)
    print("\nСобственные значения, найденные QR-алгоритмом:")
    for val in vectorlambda:
        if isinstance(val, complex):
            print(f"  {val.real:.6g}{val.imag:+.6g}j")
        else:
            print(f"  {val:.6g}")
    print(f"Число QR-итераций: {iters}")
    print("\n" + "-" * 70)
    print("Пример матрицы с комплексно-сопряжёнными собственными значениями")
    A_c = [[0.0, -1.0], [1.0, 0.0]]
    printmatr("A_complex", A_c)
    vectorlambda_c, _ = qr_eigenvalues(A_c, eps)
    print("Собственные значения (QR-алгоритм):")
    for val in vectorlambda_c:print(f"  {val.real:.6g}{val.imag:+.6g}j")

with open("input.json", encoding="utf-8") as f:
    data = json.load(f)
if "lu" in data:
    lab1_1(data["lu"])
if "tridiag" in data:
    task_1_2(data["tridiag"])
if "iterative" in data:
    task_1_3(data["iterative"])
if "jacobi" in data:
    task_1_4(data["jacobi"])
if "qr" in data:
    task_1_5(data["qr"])
