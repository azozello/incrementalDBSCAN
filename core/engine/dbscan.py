from itertools import cycle
from math import hypot
from numpy import random

import math as m
import matplotlib.pyplot as plt

EPS = 0.2
min_samples = 4


def dbscan_naive(P, eps, m, distance, C=0):

    NOISE = 0

    visited_points = set()
    clustered_points = set()
    clusters = {NOISE: []}
    cluster_cores = []

    def region_query(p):
        return [q for q in P if distance(p, q) < eps]

    def expand_cluster(p, neighbours):
        if C not in clusters:
            clusters[C] = []
        clusters[C].append(p)
        cluster_cores.append([p[0], p[1], C])
        clustered_points.add(p)
        while neighbours:
            q = neighbours.pop()
            if q not in visited_points:
                visited_points.add(q)
                neighbourz = region_query(q)
                if len(neighbourz) > m:
                    cluster_cores.append([q[0], q[1], C])
                    neighbours.extend(neighbourz)
            if q not in clustered_points:
                clustered_points.add(q)
                clusters[C].append(q)
                if q in clusters[NOISE]:
                    clusters[NOISE].remove(q)

    for p in P:
        if p in visited_points:
            continue
        visited_points.add(p)
        neighbours = region_query(p)
        if len(neighbours) < m:
            clusters[NOISE].append(p)
        else:
            C += 1
            expand_cluster(p, neighbours)

    return clusters, cluster_cores


def distance(p1, p2):
    a = float(p2[1])
    b = float(p1[1])
    c = float(p2[0])
    d = float(p1[0])
    return m.sqrt(m.pow(d - c, 2) + m.pow(b - a, 2))


def increment(cores, clusters, new_points=None, eps=0.2, m=6):
    means = find_all_means(clusters, cores)
    noise = []

    while new_points:
        p = new_points.pop()
        p_nearest_core = find_nearest_core(p, cores)
        p_nearest_mean = find_nearest_mean(p, means)

        if p_nearest_core[2] == p_nearest_mean[1] and distance(p, p_nearest_core) < eps:
            clusters[p_nearest_core[2]].append(p)
        else:
            noise.append(p)

    noise += clusters[0]

    n_clusters, n_cores = dbscan_naive(noise, eps, m, distance, len(clusters)-1)

    cores += n_cores

    for n_c in n_clusters[0]:
        if n_c not in clusters[0]:
            clusters[0].append(n_c)

    cluster_index = len(clusters)
    for i in range(1, len(n_clusters)):
        if cluster_index not in clusters:
            clusters[cluster_index] = []
            clusters[cluster_index] += n_clusters[cluster_index]
            cluster_index += 1

    return clusters, cores


def find_all_means(clusters, cores):
    return [[find_mean([[core[0], core[1]] for core in cores if core[2] == i]), i] for i in range(1, len(clusters))]


def find_mean(coordinates):
    if len(coordinates) != 0:
        sum_x = 0.0
        sum_y = 0.0
        for i in range(len(coordinates)):
            sum_x = sum_x + coordinates[i][0]
            sum_y = sum_y + coordinates[i][1]
        return sum_x / len(coordinates), sum_y / len(coordinates)


def find_nearest_mean(p, mm):
    nearest = mm[0]
    for mean in mm:
        if distance(p, mean[0]) < distance(p, nearest[0]):
            nearest = mean
    return nearest


def find_nearest_core(p, cc):
    nearest = cc[0]
    for core in cc:
        if distance(p, core) < distance(p, nearest):
            nearest = core
    return nearest


if __name__ == "__main__":
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'm', 'g', 'r', 'c', 'm', 'y', 'r', 'g',
              'b', 'g', 'r', 'c', 'm', 'y', 'm', 'g', 'r', 'c', 'm', 'y', 'r', 'g']
    # P = [(random.randn()/6, random.randn()/6) for i in range(150)]
    # P.extend([(random.randn()/4 + 2.5, random.randn()/5) for i in range(150)])
    # P.extend([(random.randn()/5 + 1, random.randn()/2 + 1) for i in range(150)])
    # P.extend([(i/25 - 1, + random.randn()/20 - 1) for i in range(100)])
    # P.extend([(i/25 - 2.5, 3 - (i/50 - 2)**2 + random.randn()/20) for i in range(150)])

    points = [(random.randn()/6, random.randn()/6) for i in range(150)]
    points.extend([(random.randn()/4 + 2.5, random.randn()/5) for i in range(150)])

    clusters_start, cores_start = dbscan_naive(points, EPS, min_samples, lambda x, y: hypot(x[0] - y[0], x[1] - y[1]))
    for c_s in clusters_start[0]:
        plt.plot(c_s[0], c_s[1], 'k+')
    for i in range(1, len(clusters_start)):
        for j in range(len(clusters_start[i])):
            plt.plot(clusters_start[i][j][0], clusters_start[i][j][1], colors[i]+'+')
    # plt.show()
    plt.savefig('../data/start.png')
    plt.clf()

    for i in range(2):
        P1 = [(random.randn()/5 + 1, random.randn()/2 + 1) for i in range(75)]
        points += P1
        clusters_start, cores_start = increment(cores_start, clusters_start, P1, EPS, min_samples)
        for c_s in clusters_start[0]:
            plt.plot(c_s[0], c_s[1], 'k+')
        for i in range(1, len(clusters_start)):
            for j in range(len(clusters_start[i])):
                plt.plot(clusters_start[i][j][0], clusters_start[i][j][1], colors[i] + '+')
        # plt.show()
        plt.savefig('../data/%d_P1.png' % i)
        plt.clf()

        P2 = [(i/25 - 1, + random.randn()/20 - 1) for i in range(50)]
        points.extend(P2)
        clusters_start, cores_start = increment(cores_start, clusters_start, P2, EPS, min_samples)

        for c_s in clusters_start[0]:
            plt.plot(c_s[0], c_s[1], 'k+')
        for i in range(1, len(clusters_start)):
            for j in range(len(clusters_start[i])):
                plt.plot(clusters_start[i][j][0], clusters_start[i][j][1], colors[i] + '+')
        # plt.show()
        plt.savefig('../data/%d_P2.png' % i)
        plt.clf()

        P3 = [(i/25 - 2.5, 3 - (i/50 - 2)**2 + random.randn()/20) for i in range(75)]
        points.extend(P3)
        clusters_start, cores_start = increment(cores_start, clusters_start, P3, EPS, min_samples)
        for c_s in clusters_start[0]:
            plt.plot(c_s[0], c_s[1], 'k+')
        for i in range(1, len(clusters_start)):
            for j in range(len(clusters_start[i])):
                plt.plot(clusters_start[i][j][0], clusters_start[i][j][1], colors[i] + '+')
        # plt.show()
        plt.savefig('../data/%d_P3.png' % i)
        plt.clf()

    clusters_all, cores_all = dbscan_naive(points, EPS, min_samples, lambda x, y: hypot(x[0] - y[0], x[1] - y[1]))

    for c_s in clusters_all[0]:
        plt.plot(c_s[0], c_s[1], 'k+')
    for i in range(1, len(clusters_all)):
        for j in range(len(clusters_all[i])):
            plt.plot(clusters_all[i][j][0], clusters_all[i][j][1], colors[i]+'+')
    plt.savefig('../data/all.png')
    # plt.show()
    plt.clf()

    print(len(clusters_start[0]))
    print(len(clusters_all[0]))
