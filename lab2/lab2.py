import json
import math
import sys
import matplotlib.pyplot as plt

SAFE = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
SAFE["abs"] = abs


def ev(expr, env):
    return eval(expr, {}, {**SAFE, **env})


def make_f(expr, params):
    return lambda x: ev(expr, {**params, "x": x})


def num_deriv(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def simple_iter_scalar(g, x0, eps, maxit):
    x, hist = x0, [x0]
    for _ in range(maxit):
        xn = g(x)
        hist.append(xn)
        if abs(xn - x) < eps:
            return xn, hist, True
        x = xn
    return x, hist, False


def newton_scalar(f, df, x0, eps, maxit):
    x, hist = x0, [x0]
    for _ in range(maxit):
        xn = x - f(x) / df(x)
        hist.append(xn)
        if abs(xn - x) < eps:
            return xn, hist, True
        x = xn
    return x, hist, False


def make_F(exprs, vars_, params):
    def F(x):
        env = {**params, **dict(zip(vars_, x))}
        return [ev(e, env) for e in exprs]
    return F


def jacobian(F, x, h=1e-6):
    n = len(x)
    J = [[0.0] * n for _ in range(n)]
    for j in range(n):
        xp, xm = list(x), list(x)
        xp[j] += h
        xm[j] -= h
        Fp, Fm = F(xp), F(xm)
        for i in range(n):
            J[i][j] = (Fp[i] - Fm[i]) / (2 * h)
    return J


def gauss(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            c = M[r][col] / M[col][col]
            for cc in range(col, n + 1):
                M[r][cc] -= c * M[col][cc]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x


def simple_iter_system(G, x0, eps, maxit):
    x, hist = list(x0), [list(x0)]
    for _ in range(maxit):
        xn = G(x)
        hist.append(list(xn))
        if max(abs(a - b) for a, b in zip(xn, x)) < eps:
            return xn, hist, True
        x = xn
    return x, hist, False


def newton_system(F, x0, eps, maxit):
    x, hist = list(x0), [list(x0)]
    for _ in range(maxit):
        Fx = F(x)
        delta = gauss(jacobian(F, x), [-v for v in Fx])
        xn = [a + b for a, b in zip(x, delta)]
        hist.append(list(xn))
        if max(abs(d) for d in delta) < eps:
            return xn, hist, True
        x = xn
    return x, hist, False


def plot_scalar(cfg, f, x_it, h_it, x_nt, h_nt, eps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    a, b = cfg["plot_range"]
    xs = [a + (b - a) * i / 800 for i in range(801)]
    ax1.plot(xs, [f(x) for x in xs], "b-", label="f(x)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axvline(x_it, color="g", ls="--", label=f"пр.итер.: {x_it:.8f}")
    ax1.axvline(x_nt, color="r", ls=":",  label=f"Ньютон:  {x_nt:.8f}")
    ax1.plot(x_it, 0, "go", ms=8)
    ax1.plot(x_nt, 0, "ro", ms=8)
    ax1.set_title(f"2.1: f(x) = {cfg['expression']}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    e1 = [abs(h_it[i] - h_it[i - 1]) for i in range(1, len(h_it))]
    e2 = [abs(h_nt[i] - h_nt[i - 1]) for i in range(1, len(h_nt))]
    if e1:
        ax2.semilogy(range(1, len(e1) + 1),
                     [max(v, 1e-16) for v in e1],
                     "g-o", label="простая итерация")
    if e2:
        ax2.semilogy(range(1, len(e2) + 1),
                     [max(v, 1e-16) for v in e2],
                     "r-s", label="Ньютон")
    ax2.axhline(eps, color="k", ls="--", label=f"ε = {eps:g}")
    ax2.set_title("2.1: зависимость погрешности от числа итераций")
    ax2.set_xlabel("k")
    ax2.set_ylabel("|x_k − x_{k−1}|")
    ax2.legend()
    ax2.grid(which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("result_2_1.png", dpi=120)


def plot_system(cfg, F, x_it, h_it, x_nt, h_nt, eps):
    n = len(cfg["variables"])

    if n == 2:
        fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(13, 5))
        r1, r2 = cfg["plot_range"]["x1"], cfg["plot_range"]["x2"]
        N = 200
        xs1 = [r1[0] + (r1[1] - r1[0]) * i / N for i in range(N + 1)]
        xs2 = [r2[0] + (r2[1] - r2[0]) * i / N for i in range(N + 1)]

        Z = []
        for x2 in xs2:
            row = []
            for x1 in xs1:
                env = {**cfg.get("parameters", {}),
                       cfg["variables"][0]: x1,
                       cfg["variables"][1]: x2}
                row.append([ev(e, env) for e in cfg["expressions"]])
            Z.append(row)

        colors = ["blue", "orange", "purple", "brown"]
        for k in range(len(cfg["expressions"])):
            Zk = [[Z[i][j][k] for j in range(N + 1)] for i in range(N + 1)]
            ax.contour(xs1, xs2, Zk, levels=[0],
                       colors=colors[k % 4], linewidths=2)
            ax.plot([], [], color=colors[k % 4], lw=2, label=f"f{k+1}=0")

        ax.set_xlim(r1)
        ax.set_ylim(r2)
        ax.set_title("2.2: линии уровня f₁=0, f₂=0")
        ax.set_xlabel(cfg["variables"][0])
        ax.set_ylabel(cfg["variables"][1])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        fig, ax_err = plt.subplots(1, 1, figsize=(8, 5))

    e1 = [max(abs(a - b) for a, b in zip(h_it[i], h_it[i - 1]))
          for i in range(1, len(h_it))]
    e2 = [max(abs(a - b) for a, b in zip(h_nt[i], h_nt[i - 1]))
          for i in range(1, len(h_nt))]
    if e1:
        ax_err.semilogy(range(1, len(e1) + 1),
                        [max(v, 1e-16) for v in e1],
                        "g-o", label="простая итерация")
    if e2:
        ax_err.semilogy(range(1, len(e2) + 1),
                        [max(v, 1e-16) for v in e2],
                        "r-s", label="Ньютон")
    ax_err.axhline(eps, color="k", ls="--", label=f"ε = {eps:g}")
    ax_err.set_title("2.2: зависимость погрешности от числа итераций")
    ax_err.set_xlabel("k")
    ax_err.set_ylabel("max |Δx|")
    ax_err.legend()
    ax_err.grid(which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("result_2_2.png", dpi=120)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "input2.json"
    cfg = json.load(open(path, encoding="utf-8"))
    eps = float(cfg["precision"])
    maxit = int(cfg.get("max_iterations", 300))
    if "single_equation" in cfg:
        se = cfg["single_equation"]
        params = se.get("parameters", {})
        f = make_f(se["expression"], params)
        x0 = float(se["x0"])

        print("2.1")
        print(f"  f(x) = {se['expression']} = 0")
        print(f"  x0 = {x0},  ε = {eps:g},  max_iter = {maxit}")
        tau = 1.0 / num_deriv(f, x0)
        g = lambda x, _f=f, _tau=tau: x - _tau * _f(x)
        print(f"  g(x) = x - τ·f(x),  τ = 1/f'(x0) = {tau:.10f}")
        gx = num_deriv(g, x0)
        print(f"  g'(x0) = {gx:.6f}   (оценка |g'| в стартовой точке)")
        x_it, h_it, ok_it = simple_iter_scalar(g, x0, eps, maxit)
        x_nt, h_nt, ok_nt = newton_scalar(f, lambda x: num_deriv(f, x), x0, eps, maxit)

        print(f"\nПростая итерация: x* = {x_it:.10f}, "f"k = {len(h_it) - 1}, {'OK' if ok_it else 'FAIL'}")
        print(f"  |f(x*)| = {abs(f(x_it)):.2e}")
        print(f"Ньютон:x* = {x_nt:.10f}, "f"k = {len(h_nt) - 1}, {'OK' if ok_nt else 'FAIL'}")
        print(f"  |f(x*)| = {abs(f(x_nt)):.2e}")

        print("\nИтерации простой итерации:")
        for k, x in enumerate(h_it):
            d = abs(h_it[k] - h_it[k - 1]) if k else 0.0
            print(f"  k={k:2d}  x={x:.10f}  |Δx|={d:.2e}")
        print("\nИтерации Ньютона:")
        for k, x in enumerate(h_nt):
            d = abs(h_nt[k] - h_nt[k - 1]) if k else 0.0
            print(f"  k={k:2d}  x={x:.10f}  |Δx|={d:.2e}")
        plot_scalar(se, f, x_it, h_it, x_nt, h_nt, eps)

    if "system" in cfg:
        sy = cfg["system"]
        vars_ = sy["variables"]
        params = sy.get("parameters", {})
        F = make_F(sy["expressions"], vars_, params)
        x0 = list(sy["x0"])
        print("\n" + "=" * 66)
        print("2.2")
        for e in sy["expressions"]:
            print(f"  {e} = 0")
        print(f"  переменные: {vars_}, параметры: {params}")
        print(f"  x0 = {x0}, ε = {eps:g}")

        J0 = jacobian(F, x0)
        print(f"  J(x0) = {J0}")
        def G(x, _F=F, _J0=J0):
            Fx = _F(x)
            d = gauss(_J0, [-v for v in Fx])
            return [xi + di for xi, di in zip(x, d)]
        x_it, h_it, ok_it = simple_iter_system(G, x0, eps, maxit)
        x_nt, h_nt, ok_nt = newton_system(F, x0, eps, maxit)
        fmt = lambda v: "(" + ", ".join(f"{x:.10f}" for x in v) + ")"
        print(f"\nПростая итерация: x* = {fmt(x_it)}, "f"k = {len(h_it) - 1}, {'OK' if ok_it else 'FAIL'}")
        print(f"  ||F||∞ = {max(abs(v) for v in F(x_it)):.2e}")
        print(f"Ньютон:           x* = {fmt(x_nt)}, "f"k = {len(h_nt) - 1}, {'OK' if ok_nt else 'FAIL'}")
        print(f"  ||F||∞ = {max(abs(v) for v in F(x_nt)):.2e}")
        print("\nИтерации простой итерации:")
        for k, x in enumerate(h_it):
            d = (max(abs(a - b) for a, b in zip(h_it[k], h_it[k - 1]))
                 if k else 0.0)
            print(f"  k={k:2d}  x={fmt(x)}  ||Δx||∞={d:.2e}")
        print("\nИтерации Ньютона:")
        for k, x in enumerate(h_nt):
            d = (max(abs(a - b) for a, b in zip(h_nt[k], h_nt[k - 1]))
                 if k else 0.0)
            print(f"  k={k:2d}  x={fmt(x)}  ||Δx||∞={d:.2e}")

        plot_system(sy, F, x_it, h_it, x_nt, h_nt, eps)
    plt.show()



if __name__ == "__main__":
    main()
