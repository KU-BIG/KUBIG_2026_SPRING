import os
import numpy as np
from sklearn.preprocessing import normalize


# ── dionysus.Diagram 호환 클래스 ─────────────────────────────────────────────
class _Point:
    def __init__(self, birth, death):
        self.birth = float(birth)
        self.death = float(death)

    def __iter__(self):
        yield self.birth
        yield self.death

    def __repr__(self):
        return f"({self.birth}, {self.death})"


class Diagram:
    def __init__(self, points=None):
        self._pts = []
        if points:
            for p in points:
                if isinstance(p, _Point):
                    self._pts.append(p)
                elif hasattr(p, 'birth'):
                    self._pts.append(_Point(p.birth, p.death))
                else:
                    self._pts.append(_Point(p[0], p[1]))

    def append(self, pt):
        if isinstance(pt, _Point):
            self._pts.append(pt)
        elif hasattr(pt, 'birth'):
            self._pts.append(_Point(pt.birth, pt.death))
        else:
            self._pts.append(_Point(pt[0], pt[1]))

    def __iter__(self):
        return iter(self._pts)

    def __len__(self):
        return len(self._pts)

    def __getitem__(self, idx):
        return self._pts[idx]

    def __repr__(self):
        return f"Diagram({self._pts})"
# ────────────────────────────────────────────────────────────────────────────


def tuple2dgm(tup):
    return Diagram(tup)


def diag2array(diag):
    return np.array(diag)


def array2diag(array):
    res = []
    n = len(array)
    for i in range(n):
        p = [array[i, 0], array[i, 1]]
        res.append(p)
    return res


def dgm2diag(dgm):
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag


def diag2dgm(diag):
    if type(diag) == list:
        diag = [tuple(i) for i in diag]
    elif type(diag) == np.ndarray:
        diag = [tuple(i) for i in diag]
    return Diagram(diag)


def assert_dgm_above(dgm):
    for p in dgm:
        try:
            assert p.birth <= p.death
        except AssertionError:
            raise Exception('birth is larger than death')


def assert_dgm_below(dgm):
    for p in dgm:
        try:
            assert p.birth >= p.death
        except AssertionError:
            raise Exception('birth is smaller than death')


def flip_dgm(dgm):
    for p in dgm:
        if float(p.birth) < float(p.death):
            assert_dgm_above(dgm)
            return dgm
        assert float(p.birth) >= float(p.death)
    data = [(float(p.death), float(p.birth)) for p in dgm]
    return Diagram(data)


def print_dgm(dgm):
    for p in dgm:
        print(p)


def precision_format(nbr, precision=1):
    return round(nbr * (10 ** precision)) / (10 ** precision)


def normalize_(x, axis=0):
    return normalize(x, axis=axis)


def dgms_summary(dgms, debug='off'):
    n = len(dgms)
    total_pts = [-1] * n
    unique_total_pts = [-1] * n
    for i in range(len(dgms)):
        total_pts[i] = len(dgms[i])
        unique_total_pts[i] = len(set([(p.birth, p.death) for p in dgms[i]]))
    if debug == 'on':
        print('Total number of points for all dgms')
        print(dgms)
    stat_with_multiplicity = (
        precision_format(np.mean(total_pts), precision=1),
        precision_format(np.std(total_pts), precision=1),
        np.min(total_pts), np.max(total_pts))
    stat_without_multiplicity = (
        precision_format(np.mean(unique_total_pts)),
        precision_format(np.std(unique_total_pts)),
        np.min(unique_total_pts), np.max(unique_total_pts))
    print('Dgms with multiplicity    Mean: %s, Std: %s, Min: %s, Max: %s' % (
        precision_format(np.mean(total_pts)),
        precision_format(np.std(total_pts)),
        precision_format(np.min(total_pts)),
        precision_format(np.max(total_pts))))
    print('Dgms without multiplicity Mean: %s, Std: %s, Min: %s, Max: %s' % (
        precision_format(np.mean(unique_total_pts)),
        precision_format(np.std(unique_total_pts)),
        precision_format(np.min(unique_total_pts)),
        precision_format(np.max(unique_total_pts))))
    return (stat_with_multiplicity, stat_without_multiplicity)
