import math


def resource_allocation(storage_cost, production_cost, goods_count, penalty_cost=None, period=365):
    c1, c2, c3 = storage_cost, production_cost, penalty_cost
    Q = goods_count
    T = period
    if penalty_cost is None:
        t0 = T * math.sqrt(c2 / (6 * c1 * Q))
        q0 = Q * t0 / T
    else:
        t0 = T * math.sqrt(c2 / (Q * (12 * c3 - (1 - 0.25 * c1 / c3)**2)))
        q = Q * t0 / T * (1 - 0.25 * c1 / c3)
        print(q)
        q0 = Q * t0 / T
        condition = (2 * T / t0**3 * (c2 + q**2 / Q)) * 24 * c3 * T / (Q * t0) - \
            (24 * c3 * q * T / (Q * t0**2))**2
        if condition < 0:
            raise ValueError("The task has no solution")
        elif condition == 0:
            raise ValueError("One don't know if task has a solution")

    return t0, q0


if __name__ == '__main__':
    storage_cost = 0.1
    production_cost = 350
    goods_count = 24000
    penalty_cost = 0.1
    t0, q0 = resource_allocation(
        storage_cost, production_cost, goods_count, penalty_cost=penalty_cost)
    print("Period is equal to {}".format(t0))
    print("Goods per period is equal to {}".format(q0))
