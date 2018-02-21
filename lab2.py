import numpy as np
import random
from cvxopt import matrix, solvers


def generate_random_productivity_matrix(worker_groups_count, jobs_count):
    return np.array([[random.random() * 10 for j in range(worker_groups_count)] for i in range(jobs_count)])


def generate_worker_groups(worker_groups_count):
    return np.array([random.randrange(1, 10) for _ in range(worker_groups_count)])


def generate_jobs(jobs_count, total_jobs):
    res = []
    for i in range(jobs_count):
        res.append(random.randint(1, total_jobs - sum(res) - (jobs_count - i)))
    if sum(res) < total_jobs:
        res[-1] += total_jobs - sum(res)
    return np.array(res)


def make_input(productivity_matrix, worker_groups, jobs):
    assert sum(worker_groups) == sum(jobs)
    height, width = worker_groups.shape[0], jobs.shape[0]
    G = -np.eye(height * width)
    h = np.zeros(height * width)
    A = np.zeros([height + width, height * width])
    for i in range(height):
        for j in range(height * width):
            if j // width == i:
                A[i, j] = 1.
    for i in range(width):
        for j in range(height * width):
            if j % width == i:
                A[height + i, j] = 1.

    #import ipdb
    # ipdb.set_trace()
    b = np.append(worker_groups, jobs).astype(np.float64)
    c = -productivity_matrix.flatten()
    return c, G, h, A, b


def solve(productivity_matrix, worker_groups, jobs):
    c, G, h, A, b = make_input(productivity_matrix, worker_groups, jobs)

    c_mat, G_mat, h_mat, b_mat, A_mat = matrix(
        c), matrix(G), matrix(h), matrix(b), matrix(A)

    sol = solvers.lp(c_mat, G_mat, h_mat, A_mat, b_mat, solver='glpk')
    return np.array(sol['x']).T[0], c


if __name__ == '__main__':
    worker_groups_count = 5
    jobs_count = 8
    worker_groups = generate_worker_groups(worker_groups_count)
    total_jobs = np.sum(worker_groups)
    jobs = generate_jobs(jobs_count, total_jobs)
    productivity_matrix = generate_random_productivity_matrix(
        worker_groups_count, jobs_count)
    sol, c = solve(productivity_matrix, worker_groups, jobs)
    print("=" * 80)
    print("COST MATRIX = ")
    print(c.reshape(5, 8))
    print("=" * 80)
    print("OPTIMAL SOLUTION = ")
    print(sol.reshape(5, 8))
    print("=" * 80)
    print("COST = ")
    print(np.dot(c[c != np.inf], sol[c != np.inf]))
    print("=" * 80)
