"""Exact-rational scalar shrinkage ROC-AUC event map (research prototype).

Purpose: falsifiable toy demonstration of rank events and half-credit ties.
Not a new root-isolation algorithm, not a floating-point EEG certificate,
not a trained decoder, not a measured improvement in target EEG accuracy.
Requires sympy; fails closed if algebraic event ordering/sign cannot be verified.
"""
from __future__ import annotations
from functools import cmp_to_key
import sympy as sp

lam = sp.Symbol('lambda', real=True)


def _cmp_exact(a, b):
    diff = sp.simplify(a - b)
    if diff == 0:
        return 0
    sign = sp.sign(diff)
    if sign == 1:
        return 1
    if sign == -1:
        return -1
    raise ValueError(f'Cannot prove ordering of algebraic event roots: {a}, {b}')


def _sign_exact(expr):
    result = sp.sign(sp.simplify(expr))
    if result == 0:
        return sp.Rational(1, 2)
    if result == 1:
        return sp.Integer(1)
    if result == -1:
        return sp.Integer(0)
    raise ValueError(f'Cannot prove sign: {expr}')


def numerator(spectra, coeffs):
    """N(lambda)=sum_i a_i prod_{j!=i}[(1-lambda)*s_j+lambda*mean(s)]."""
    assert len(spectra) == len(coeffs) and len(spectra) >= 2
    s = list(map(sp.Rational, spectra))
    assert all(x > 0 for x in s), 'Only positive-definite spectral family is supported'
    nu = sum(s) / len(s)
    q = [si + (nu - si) * lam for si in s]
    return sp.Poly(sp.expand(sum(sp.Rational(a) * sp.prod(q[j] for j in range(len(q)) if j != i)
                                 for i, a in enumerate(coeffs))), lam)


def exact_events(pair_polynomials):
    """Rank-change candidate endpoints, rejecting uncertifiable root comparisons."""
    points = [sp.Integer(0), sp.Integer(1)]
    for poly in pair_polynomials:
        if poly.is_zero:
            continue  # this pair is tied for the entire interval
        # SymPy exact algebraic root isolation (not claimed as our novel method)
        for root in poly.real_roots():
            if _cmp_exact(root, 0) >= 0 and _cmp_exact(root, 1) <= 0:
                points.append(root)
    points.sort(key=cmp_to_key(_cmp_exact))
    unique = []
    for point in points:
        if not unique or _cmp_exact(point, unique[-1]) != 0:
            unique.append(point)
    return unique


def auc_at(pair_polynomials, x):
    return sp.simplify(sum(_sign_exact(p.as_expr().subs(lam, x)) for p in pair_polynomials)
                       / sp.Integer(len(pair_polynomials)))


def event_map(spectra, pair_coeffs):
    polynomials = [numerator(spectra, a) for a in pair_coeffs]
    points = exact_events(polynomials)
    cells = []
    boundaries = [(x, auc_at(polynomials, x)) for x in points]
    for a, b in zip(points[:-1], points[1:]):
        probe = (a + b) / 2
        assert _cmp_exact(probe, a) > 0 and _cmp_exact(probe, b) < 0
        cells.append(((a, b), auc_at(polynomials, probe)))
    return polynomials, boundaries, cells


def test_exact_ties_and_grid_miss():
    _, boundaries, cells = event_map([1, 3], [[1, -2]])
    assert boundaries == [(0, 1), (sp.Rational(1, 3), sp.Rational(1, 2)), (1, 0)]
    assert cells == [((0, sp.Rational(1, 3)), 1), ((sp.Rational(1, 3), 1), 0)]

    spectrum = [1, 2, 4]
    r1, r2 = sp.Rational(11, 1000), sp.Rational(14, 1000)
    wanted = sp.expand(-(lam - r1) * (lam - r2))
    s = list(map(sp.Rational, spectrum))
    nu = sum(s) / len(s)
    q = [si + (nu - si) * lam for si in s]
    basis = [sp.prod(q[j] for j in range(len(s)) if j != i) for i in range(len(s))]
    a = sp.symbols('a0:3')
    target_eq = sp.Poly(sp.expand(sum(a[i] * basis[i] for i in range(3)) - wanted), lam)
    sol = sp.solve(target_eq.all_coeffs(), a, dict=True)
    assert len(sol) == 1
    coeffs = [sp.Rational(sol[0][v]) for v in a]
    poly, pts, c = event_map(spectrum, [coeffs])
    assert sp.simplify(poly[0].as_expr() - wanted) == 0
    assert [x for x, _ in pts] == [0, r1, r2, 1]
    assert [y for _, y in c] == [0, 1, 0]
    assert [y for _, y in pts] == [0, sp.Rational(1, 2), sp.Rational(1, 2), 0]
    original_grid = list(map(sp.Rational, ['0','.0005','.001','.002','.003','.005','.0075','.01','.015','.02','.03','.05','.075','.10','.15','.20','.30','.40','.50','.60','.70','.80','.90','.95','.98','.99','1']))
    assert all(auc_at(poly, x) == 0 for x in original_grid)
    assert auc_at(poly, sp.Rational(1, 80)) == 1
    return {'status':'PASS','synthetic_true_max_auc':1,'27_grid_max_auc':0,
            'exact_event_roots':[str(r1), str(r2)], 'synthetic_only':True}


if __name__ == '__main__':
    print(test_exact_ties_and_grid_miss())
