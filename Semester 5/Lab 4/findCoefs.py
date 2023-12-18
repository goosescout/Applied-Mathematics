import numpy as np


def r1(a: np.ndarray):
    return (a[2] * a[3] + a[1]) / (1 - a[2] - a[1] * a[3] - a[3] * a[3])


def r2(a: np.ndarray):
    return (a[1] + a[3]) * r1(a) + a[2]


def r3(a: np.ndarray):
    return a[1] * r2(a) + a[2] * r1(a) + a[3]


def find():
    while (True):
        a = np.random.normal(0, 1, 4)
        a[0] = 0
        if (a[2] + a[1] * a[3] + a[3] * a[3]) != 1 and \
                abs(r1(a)) < 1 and \
                abs(r2(a)) < 1 and \
                abs(r3(a)) < 1 and \
                a[3] != 0:
            return [a, r1(a), r2(a), r3(a)]
