import numpy as np


def solve_for_epsilon(k, eps_prime, delta):
    # solve equation
    # \epsilon’= \epsilon\sqrt{2k\ln(1/\delta’)} + k\epsilon(e^\epsilon-1)
    threshold = 1e-8
    low = 1e-10
    high = 10

    def f(eps):
        return eps * np.sqrt(2 * k * np.log(1 / delta)) + \
                k * eps * (np.exp(1) ** eps - 1)

    assert(f(low) < eps_prime and f(high) > eps_prime)

    while 1:
        mid = (low + high) / 2
        value = f(eps=mid)
        if abs(value - eps_prime) <= threshold:
            return mid
        if value > eps_prime:
            high = mid
        else:
            low = mid
