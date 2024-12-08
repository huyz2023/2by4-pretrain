import triton
import triton.language as tl


@triton.jit
def _MVUE24_approx(x0, x1, x2, x3,
                   random0, random1):
    eps = 1.19209e-07
    a0 = tl.abs(x0) + eps
    a1 = tl.abs(x1) + eps
    a2 = tl.abs(x2) + eps
    a3 = tl.abs(x3) + eps
    sum = a0 + a1 + a2 + a3

    t0 = a0 / sum
    t1 = a1 / sum
    t2 = a2 / sum
    t3 = a3 / sum

    s0 = sum - a0
    s1 = sum - a1
    s2 = sum - a2
    s3 = sum - a3

    k0 = t0 / s0
    k1 = t1 / s1
    k2 = t2 / s2
    k3 = t3 / s3
    k = k0 + k1 + k2 + k3

    p0 = (t0 + a0 * (k - k0))
    p1 = (t1 + a1 * (k - k1))
    p2 = (t2 + a2 * (k - k2))
    p3 = (t3 + a3 * (k - k3))

    m0 = (random0 <= t0)
    m1 = ((random0 <= (t0 + t1)) & ~m0)
    m2 = ((random0 <= (t0 + t1 + t2)) & ~m1 & ~m0)
    m3 = ~m2 & ~m1 & ~m0

    d_a0 = ~m0 * a0
    d_a1 = ~m1 * a1
    d_a2 = ~m2 * a2
    d_a3 = ~m3 * a3
    d_sum = d_a0 + d_a1 + d_a2 + d_a3

    t = random1 * d_sum
    d_m0 = (t <= d_a0)
    d_m1 = ((t <= (d_a0 + d_a1)) & ~d_m0)
    d_m2 = ((t <= (d_a0 + d_a1 + d_a2)) & ~d_m1 & ~d_m0)
    d_m3 = ~d_m2 & ~d_m1 & ~d_m0

    m0, m1, m2, m3 = m0 | d_m0, m1 | d_m1, m2 | d_m2, m3 | d_m3
    a0 = x0 / p0
    a1 = x1 / p1
    a2 = x2 / p2
    a3 = x3 / p3

    return a0, a1, a2, a3, m0, m1, m2, m3


@triton.jit
def _sparse24(x0, x1, x2, x3):
    (a1, a2, a3,
     a4, a5, a6) = (tl.abs(x0) > tl.abs(x1), tl.abs(x0) > tl.abs(x2), tl.abs(x0) > tl.abs(x3),
                    tl.abs(x1) > tl.abs(x2), tl.abs(x1) > tl.abs(x3), tl.abs(x2) > tl.abs(x3))
    m0, m1, m2, m3 = a2 & a3 | a1 & a2 | a1 & a3, ~a1 & a5 | a4 & a5 | ~a1 & a4, ~a2 & ~a4 | ~a2 & a6 | ~a4 & a6, ~a3 & ~a5 | ~a3 & ~a6 | ~a5 & ~a6

    return x0, x1, x2, x3, m0, m1, m2, m3


@triton.jit
def _soft_threshold(a0, a1, a2, a3):
    (x1, x2, x3,
     x4, x5, x6) = (tl.abs(a0) > tl.abs(a1), tl.abs(a0) > tl.abs(a2), tl.abs(a0) > tl.abs(a3),
                    tl.abs(a1) > tl.abs(a2), tl.abs(a1) > tl.abs(a3), tl.abs(a2) > tl.abs(a3))
    m0, m1, m2, m3 = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

    # threshold = tl.where((x1 & ~x2 & ~x3 | ~x1 & x2 & ~x3 | ~x1 & ~x2 & x3), tl.abs(a0),
    #                      tl.where((x1 & ~x4 & x5 | x1 & x4 & ~x5 | ~x1 & ~x4 & ~x5), tl.abs(a1),
    #                               tl.where((x2 & x4 & x6 | x2 & ~x4 & ~x6 | ~x2 & x4 & ~x6), tl.abs(a2), tl.abs(a3))))

    threshold = tl.minimum(tl.maximum(tl.minimum(tl.abs(a0), tl.abs(a1)), tl.minimum(tl.abs(a2), tl.abs(a3))),
                           tl.minimum(tl.maximum(tl.abs(a0), tl.abs(a1)), tl.maximum(tl.abs(a2), tl.abs(a3))))

    s0 = tl.where(a0 > 0, a0 - threshold, a0 + threshold)
    s1 = tl.where(a1 > 0, a1 - threshold, a1 + threshold)
    s2 = tl.where(a2 > 0, a2 - threshold, a2 + threshold)
    s3 = tl.where(a3 > 0, a3 - threshold, a3 + threshold)
    return s0, s1, s2, s3, m0, m1, m2, m3
