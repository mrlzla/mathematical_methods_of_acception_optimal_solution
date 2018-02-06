import numpy as np
from cvxopt import matrix, solvers
from utils import make_table


def make_c(cost_table):
    possible_hours = list(sorted(set([v['start_point'] for k, v in cost_table.items()]
                                     + [v['end_point'] for k, v in cost_table.items()])))
    size = len(possible_hours) - 1
    min_hour = possible_hours[0]
    c = np.array([])
    for k, v in cost_table.items():
        part_c = np.full(size, np.inf)
        for i in range(v['start_point'] - min_hour, v['end_point'] - min_hour):
            part_c[i] = v['value']
        c = np.append(c, part_c)
    return c


def make_A_and_b(height, width, c):
    A = np.array([])
    matrix_c = c.reshape([width // height, -1])
    additional_size = 0
    for ind, row in enumerate(matrix_c):
        inds = np.where(row != np.inf)[0]
        size = inds.shape[0]
        new_A = np.zeros([size * (size - 1) // 2, width])
        additional_size += new_A.shape[0]
        k = 0
        for i in range(size):
            for j in range(i + 1, size):
                new_A[k, ind * height + inds[i]] = 1
                new_A[k, ind * height + inds[j]] = -1
                k += 1
        if A.shape[0]:
            A = np.vstack([A, new_A])
        else:
            A = new_A.copy()
        not_used_inds = list(set(range(height)) - set(inds))
        additional_size += len(not_used_inds)
        new_A = np.zeros([len(not_used_inds), width])

        for i, j in enumerate(not_used_inds):
            new_A[i, ind * height + j] = 1

        if A.shape[0]:
            A = np.vstack([A, new_A])
        else:
            A = new_A.copy()

    return A, np.zeros(additional_size)


def make_G(height, width):
    G = -np.eye(width, width)
    new_G = np.zeros([height, width])
    for i in range(height):
        for j in range(width // height):
            new_G[i, j * height + i] = 1
    return np.vstack([G, -new_G])


def make_h(stuff_table, width):
    possible_hours = list(sorted(set([v['start_point'] for k, v in stuff_table.items()]
                                     + [v['end_point'] for k, v in stuff_table.items()])))
    size = len(possible_hours) - 1
    h = np.zeros(width)
    min_hour = possible_hours[0]
    new_h = np.zeros(size)
    for k, v in stuff_table.items():
        new_h[v['start_point'] - min_hour] = v['value']
    return np.append(h, -new_h)


def add_first_condition(G, h, height, width, first_condition_constraint):
    new_G = np.zeros([height, width])
    for i in range(height):
        for j in range(2):
            new_G[i, j * (height) + i] = 1
    return np.vstack([G, -new_G]), np.append(h, np.full(height,
                                                        -first_condition_constraint))


def add_second_condition(G, h, c, height, width, second_condition_constraint):
    matrix_c = c.reshape([width // height, -1])
    new_G = np.zeros([1, width])
    for ind, row in enumerate(matrix_c):
        non_inf_ind = np.where(row != np.inf)[0][0]
        new_G[0, ind * height + non_inf_ind] = 1
    return np.vstack([G, new_G]), np.append(h, second_condition_constraint)


def make_input(stuff_table, cost_table,
               first_condition=False, second_condition=False,
               first_condition_constraint=4, second_condition_constraint=94):
    c = make_c(cost_table)
    height, width = len(stuff_table), c.shape[0]
    A, b = make_A_and_b(height, width, c)
    G = make_G(height, width)
    h = make_h(stuff_table, width)

    if first_condition:
        G, h = add_first_condition(
            G, h, height, width, first_condition_constraint)
    if second_condition:
        G, h = add_second_condition(
            G, h, c, height, width, second_condition_constraint)
    return c, G, h, A, b


def solve(stuff_table, cost_table,
          first_condition=True, second_condition=True,
          first_condition_constraint=4, second_condition_constraint=94):
    c, G, h, A, b = make_input(
        stuff_table, cost_table,
        first_condition=first_condition, second_condition=second_condition,
        first_condition_constraint=first_condition_constraint,
        second_condition_constraint=second_condition_constraint)
    c_mat, G_mat, h_mat, b_mat, A_mat = matrix(
        c), matrix(G), matrix(h), matrix(b), matrix(A)

    sol = solvers.lp(c_mat, G_mat, h_mat, A_mat, b_mat, solver='glpk')
    return np.array(sol['x']).T[0], c


if __name__ == '__main__':
    stuff_timeline = list(zip(range(9, 19), range(10, 20)))
    stuff_values = [16, 30, 31, 45, 66, 72, 61, 34, 16, 10]

    stuff_table = make_table(stuff_timeline, stuff_values)

    cost_timeline = [
        [9, 17],
        [11, 19],
        [9, 13],
        [10, 14],
        [11, 15],
        [12, 16],
        [13, 17],
        [14, 18],
        [15, 19]
    ]
    cost_values = [8, 8, 6, 7, 9, 10, 8, 6, 6]

    cost_table = make_table(cost_timeline, cost_values)
    sol, c = solve(stuff_table, cost_table)
    print("=" * 80)
    print("COST MATRIX = ")
    print(c.reshape(9, 10))
    print("=" * 80)
    print("OPTIMAL SOLUTION = ")
    print(sol.reshape(9, 10))
    print("=" * 80)
    print("COST = ")
    print(np.dot(c[c != np.inf], sol[c != np.inf]))
    print("=" * 80)
