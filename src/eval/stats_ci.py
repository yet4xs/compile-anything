"""Exact small-sample statistics for the 9B-3 run package.

clopper_pearson(k, n, alpha) — exact binomial CI (pure-python
regularized incomplete beta; no scipy dependency).
mcnemar_exact(b, c) — exact paired binomial test on discordant pairs.
"""


def _betacf(a, b, x):
    tiny, eps, fpmin = 1e-300, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    from math import lgamma, exp, log
    ln_bt = lgamma(a + b) - lgamma(a) - lgamma(b) + \
        a * log(x) + b * log(1.0 - x)
    bt = exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betainc_inv(p, a, b):
    """Quantile of the regularized incomplete beta via bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k, n, alpha=0.05):
    """Exact CP CI: lo = B^{-1}(a/2; k, n-k+1),
    hi = B^{-1}(1-a/2; k+1, n-k). 0 successes -> lo=0."""
    lo = 0.0 if k == 0 else _betainc_inv(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else _betainc_inv(1 - alpha / 2.0, k + 1, n - k)
    return lo, hi


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts (b, c).
    Returns p-value."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p)
