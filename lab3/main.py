
import sys
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)


def load_input(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл входных данных: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def poly_to_str(p, var="x", digits=8):
    coeffs = p.coeffs
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-12:
            continue
        sign = "+" if c >= 0 else "-"
        ac = abs(c)
        if power == 0:
            term = f"{ac:.{digits}g}"
        elif power == 1:
            term = f"{ac:.{digits}g}*{var}"
        else:
            term = f"{ac:.{digits}g}*{var}^{power}"
        terms.append((sign, term))
    if not terms:
        return "0"
    result = terms[0][1] if terms[0][0] == "+" else "-" + terms[0][1]
    for sign, term in terms[1:]:
        result += f" {sign} {term}"
    return result


def lagrange_poly(x_nodes, y_nodes):
    n = len(x_nodes)
    P = np.poly1d([0.0])
    for i in range(n):
        Li = np.poly1d([1.0])
        denom = 1.0
        for j in range(n):
            if j != i:
                Li *= np.poly1d([1.0, -x_nodes[j]])
                denom *= (x_nodes[i] - x_nodes[j])
        P += y_nodes[i] * Li / denom
    return P


def divided_differences(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    return table


def newton_poly(x, y):
    n = len(x)
    table = divided_differences(x, y)
    P = np.poly1d([0.0])
    basis = np.poly1d([1.0])
    for j in range(n):
        if j > 0:
            basis *= np.poly1d([1.0, -x[j - 1]])
        P += table[0, j] * basis
    return P, table


def f31(x, func_name):
    if func_name == "1/x + x":
        return 1.0 / np.asarray(x, dtype=float) + np.asarray(x, dtype=float)
    raise ValueError(f"Неизвестная функция в task31: {func_name}")


def task31(cfg):
    print("\n" + "=" * 80)
    print("3.1. Интерполяционные многочлены Лагранжа и Ньютона")
    print(f"Функция: y = {cfg['function']}")
    print("=" * 80)

    x_star = float(cfg["x_star"])
    exact_star = float(f31(x_star, cfg["function"]))

    for case_name, case in cfg["cases"].items():
        x = np.array(case["x"], dtype=float)
        y = np.asarray(f31(x, cfg["function"]), dtype=float)

        print(f"\n--- Вариант {case_name} ---")
        print("Узлы интерполяции:")
        print(f"{'i':>3} {'x_i':>12} {'y_i':>15}")
        for i, (xi, yi) in enumerate(zip(x, y)):
            print(f"{i:3d} {xi:12.8f} {yi:15.10f}")

        L = lagrange_poly(x, y)
        N, dd_table = newton_poly(x, y)

        print("\nМногочлен Лагранжа:")
        print("L(x) =", poly_to_str(L))

        print("\nТаблица разделённых разностей:")
        n = len(x)
        for i in range(n):
            row = [f"{dd_table[i, j]: .10g}" for j in range(n - i)]
            print(f"i={i}: " + "   ".join(row))

        print("\nМногочлен Ньютона:")
        print("N(x) =", poly_to_str(N))

        L_val = float(L(x_star))
        N_val = float(N(x_star))

        print(f"\nТочка x* = {x_star:.10f}")
        print(f"L(x*) = {L_val:.15f}")
        print(f"N(x*) = {N_val:.15f}")
        print(f"f(x*) = {exact_star:.15f}")
        print(f"|L(x*) - f(x*)| = {abs(L_val - exact_star):.15e}")
        print(f"|N(x*) - f(x*)| = {abs(N_val - exact_star):.15e}")
        print(f"|L(x*) - N(x*)| = {abs(L_val - N_val):.15e}")

        x_plot = np.linspace(x.min() - 0.1, x.max() + 0.1, 500)
        plt.figure(figsize=(8, 5))
        plt.plot(x_plot, f31(x_plot, cfg["function"]),
                 label=f"f(x) = {cfg['function']}", linewidth=2)
        plt.plot(x_plot, L(x_plot), "--", label="Лагранж")
        plt.plot(x_plot, N(x_plot), ":", label="Ньютон")
        plt.scatter(x, y, s=50, zorder=5, label="узлы")
        plt.scatter([x_star], [exact_star], s=80, marker="*", zorder=6,
                    label=f"x* = {x_star:g}")
        plt.title(f"3.1, вариант 25{case_name}: интерполяция")
        plt.xlabel("x"); plt.ylabel("y")
        plt.grid(True); plt.legend(); plt.tight_layout()


def natural_cubic_spline(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    A[0, 0] = 1.0
    A[n, n] = 1.0
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    M = np.linalg.solve(A, rhs)

    coeffs = []
    for i in range(n):
        a = y[i]
        b = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0
        c = M[i] / 2.0
        d = (M[i + 1] - M[i]) / (6.0 * h[i])
        coeffs.append((a, b, c, d))
    return coeffs, M


def spline_eval(x_star, x_nodes, coeffs):
    for i, (a, b, c, d) in enumerate(coeffs):
        if x_nodes[i] <= x_star <= x_nodes[i + 1]:
            t = x_star - x_nodes[i]
            return a + b * t + c * t * t + d * t * t * t, i
    raise ValueError("x* вне интервала сплайна")


def task32(cfg):
    print("\n" + "=" * 80)
    print("3.2. Кубический сплайн")
    print("=" * 80)

    x = np.array(cfg["x"], dtype=float)
    y = np.array(cfg["y"], dtype=float)
    x_star = float(cfg["x_star"])

    print("Исходные узлы и значения функции:")
    print(f"{'i':>3} {'x_i':>12} {'f_i':>15}")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"{i:3d} {xi:12.8f} {yi:15.10f}")

    coeffs, M = natural_cubic_spline(x, y)

    print("\nКоэффициенты сплайна:")
    print("S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3")
    for i, (a, b, c, d) in enumerate(coeffs):
        print(f"\ni = {i} (интервал {i + 1}): [{x[i]:.8g}, {x[i + 1]:.8g}]")
        print(f"  a_{i} = {a:.15f}")
        print(f"  b_{i} = {b:.15f}")
        print(f"  c_{i} = {c:.15f}")
        print(f"  d_{i} = {d:.15f}")

    S_star, interval_idx = spline_eval(x_star, x, coeffs)
    print(f"\nТочка x* = {x_star:.10f}")
    print(f"Номер интервала: {interval_idx} (0-based), {interval_idx + 1} (1-based)")
    print(f"Значение сплайна S(x*) = {S_star:.15f}")

    print("\nПроверка прохождения сплайна через исходные узлы:")
    for i, xi in enumerate(x):
        val, _ = spline_eval(xi, x, coeffs)
        print(f"x = {xi:.8f}: S(x) = {val:.15f}, f_i = {y[i]:.15f}, "
              f"разность = {abs(val - y[i]):.3e}")

    a0, b0, c0, d0 = coeffs[0]
    an, bn, cn, dn = coeffs[-1]
    t_last = x[-1] - x[-2]
    second_left = 2.0 * c0
    second_right = 2.0 * cn + 6.0 * dn * t_last

    print("\nПроверка граничных условий:")
    print(f"S''(x0) = {second_left:.15e}")
    print(f"S''(xn) = {second_right:.15e}")

    x_plot = np.linspace(x[0], x[-1], 600)
    y_plot = [spline_eval(xi, x, coeffs)[0] for xi in x_plot]

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, label="кубический сплайн")
    plt.scatter(x, y, s=50, zorder=5, label="исходные узлы")
    plt.scatter([x_star], [S_star], s=80, marker="*", zorder=6,
                label=f"x* = {x_star:g}")
    plt.title("3.2. Кубический сплайн, вариант 25")
    plt.xlabel("x"); plt.ylabel("S(x)")
    plt.grid(True); plt.legend(); plt.tight_layout()


def least_squares_poly(x, y, degree):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vander(x, degree + 1, increasing=True)
    AtA = A.T @ A
    Aty = A.T @ y
    try:
        c = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c


def task33(cfg):
    print("\n" + "=" * 80)
    print("3.3. Метод наименьших квадратов")
    print("=" * 80)

    x = np.array(cfg["x"], dtype=float)
    y = np.array(cfg["y"], dtype=float)

    print("Исходные данные:")
    print(f"{'i':>3} {'x_i':>12} {'y_i':>15}")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"{i:3d} {xi:12.8f} {yi:15.10f}")

    results = []
    for degree in [1, 2, 3]:
        c = least_squares_poly(x, y, degree)
        p = np.poly1d(c[::-1])
        y_pred = p(x)
        mse = float(np.mean((y_pred - y) ** 2))
        print(f"\nМногочлен степени {degree}:")
        print("Коэффициенты (по возрастанию степеней):")
        for k, coeff in enumerate(c):
            print(f"  c_{k} = {coeff:.15f}")
        print(f"P_{degree}(x) = {poly_to_str(p)}")
        print(f"MSE_{degree} = {mse:.15e}")
        results.append({"degree": degree, "coeffs": c, "poly": p, "mse": mse})

    best = min(results, key=lambda r: r["mse"])
    print("\nСравнение ошибок:")
    for r in results:
        print(f"MSE_{r['degree']} = {r['mse']:.15e}")
    print(f"Лучший многочлен: степени {best['degree']} "
          f"(MSE = {best['mse']:.15e})")

    x_plot = np.linspace(x.min() - 0.2, x.max() + 0.2, 600)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=60, zorder=5, label="исходные точки")
    for r in results:
        plt.plot(x_plot, r["poly"](x_plot), label=f"P_{r['degree']}(x)")
    plt.title("3.3. МНК, вариант 25")
    plt.xlabel("x"); plt.ylabel("y")
    plt.grid(True); plt.legend(); plt.tight_layout()


def task34(cfg):
    print("\n" + "=" * 80)
    print("3.4. Численное дифференцирование")
    print("=" * 80)

    x = np.array(cfg["x"], dtype=float)
    y = np.array(cfg["y"], dtype=float)
    x_star = float(cfg["x_star"])

    print("Исходная таблица:")
    print(f"{'i':>3} {'x_i':>12} {'y_i':>15}")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"{i:3d} {xi:12.8f} {yi:15.10f}")
    print(f"\nТочка вычисления: x* = {x_star:.10f}")

    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError("Требуется равномерная сетка")
    idx = int(np.argmin(np.abs(x - x_star)))
    if abs(x[idx] - x_star) > 1e-12:
        raise ValueError("x* должно совпадать с центральным узлом таблицы")
    if idx < 2 or idx > len(x) - 3:
        raise ValueError("Нужно 5 узлов и x* в центральном")

    y0, y1, y2, y3, y4 = y[idx - 2:idx + 3]
    first_derivative = (y0 - 8.0 * y1 + 8.0 * y3 - y4) / (12.0 * h)
    second_derivative = (-y0 + 16.0 * y1 - 30.0 * y2 + 16.0 * y3 - y4) / (12.0 * h * h)

    print("Использованные узлы:")
    print(f"{'i':>3} {'x_i':>12} {'y_i':>15}")
    for i in range(idx - 2, idx + 3):
        print(f"{i:3d} {x[i]:12.8f} {y[i]:15.10f}")

    print(f"\nПервая производная:  y'(x*) ≈ {first_derivative:.15f}")
    print(f"Вторая производная:  y''(x*) ≈ {second_derivative:.15f}")


def _f35_factory(name):
    if name == "x * sqrt(49 - x^2)":
        def f(x):
            return x * math.sqrt(max(0.0, 49.0 - x * x))
        return f
    raise ValueError(f"Неизвестная функция в task35: {name}")


def rect_left(f, a, b, h):
    n = int(round((b - a) / h)); s = 0.0
    for i in range(n):
        s += f(a + i * h)
    return s * h


def rect_right(f, a, b, h):
    n = int(round((b - a) / h)); s = 0.0
    for i in range(1, n + 1):
        s += f(a + i * h)
    return s * h


def rect_mid(f, a, b, h):
    n = int(round((b - a) / h)); s = 0.0
    for i in range(n):
        s += f(a + (i + 0.5) * h)
    return s * h


def trapezoid(f, a, b, h):
    n = int(round((b - a) / h))
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h


def simpson(f, a, b, h):
    n = int(round((b - a) / h))
    if n % 2 != 0:
        raise ValueError("Для Симпсона число интервалов должно быть чётным")
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4.0 * f(a + i * h)
    for i in range(2, n, 2):
        s += 2.0 * f(a + i * h)
    return s * h / 3.0


def task35(cfg):
    print("\n" + "=" * 80)
    print("3.5. Численное интегрирование")
    print(f"Функция: y = {cfg['function']}")
    print("=" * 80)

    f = _f35_factory(cfg["function"])
    a = float(cfg["a"]); b = float(cfg["b"])
    h1 = float(cfg["h1"]); h2 = float(cfg["h2"])
    exact = cfg.get("exact", None)
    if exact is not None:
        exact = float(exact)

    print(f"Интервал: [{a:g}, {b:g}]")
    print(f"h1 = {h1:g}, h2 = {h2:g}")
    if exact is not None:
        print(f"Точное значение интеграла: {exact:.15f}")

    methods = [
        ("лев. прямоугольники", 1, rect_left),
        ("прав. прямоугольники", 1, rect_right),
        ("сред. прямоугольники", 2, rect_mid),
        ("трапеции", 2, trapezoid),
        ("Симпсона", 4, simpson),
    ]

    results = []
    header = (f"{'Метод':<22} {'I(h1)':>16} {'I(h2)':>16} "
              f"{'p':>3} {'R':>16} {'err_RR':>16}")
    if exact is not None:
        header += f" {'|R-exact|':>16}"
    print("\nРезультаты:")
    print(header)
    print("-" * len(header))

    for name, p, method in methods:
        I1 = method(f, a, b, h1)
        I2 = method(f, a, b, h2)
        R = I2 + (I2 - I1) / (2.0 ** p - 1.0)
        err_RR = abs((I2 - I1) / (2.0 ** p - 1.0))
        row = (f"{name:<22} {I1:16.10f} {I2:16.10f} {p:3d} "
               f"{R:16.10f} {err_RR:16.6e}")
        if exact is not None:
            row += f" {abs(R - exact):16.6e}"
        print(row)
        results.append({"name": name, "p": p, "I1": I1, "I2": I2,
                        "R": R, "err_RR": err_RR})

    if exact is not None:
        print("\nАбсолютные погрешности относительно точного значения:")
        for r in results:
            print(f"{r['name']:<22}: |I(h2)-exact| = "
                  f"{abs(r['I2'] - exact):.6e}, "
                  f"|R-exact| = {abs(r['R'] - exact):.6e}")
        best = min(results, key=lambda r: abs(r["R"] - exact))
        print(f"\nНаиболее точный метод: {best['name']}")
        print(f"R = {best['R']:.15f}, "
              f"|R - exact| = {abs(best['R'] - exact):.6e}")


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input3.json")
    cfg = load_input(cfg_path)

    print("Лабораторная работа №3")
    print(f"Вариант: {cfg.get('variant', 25)}")
    print(f"Данные прочитаны из: {cfg_path.resolve()}")

    task31(cfg["task31"])
    task32(cfg["task32"])
    task33(cfg["task33"])
    task34(cfg["task34"])
    task35(cfg["task35"])

    plt.show()


if __name__ == "__main__":
    main()
