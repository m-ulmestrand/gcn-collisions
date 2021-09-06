from numba import njit, jit, prange
from numba.typed import Dict
from numpy import argwhere, arange, square, sum, array, concatenate, \
                  append, ones, int64, int32, intp, subtract, fill_diagonal, zeros, sqrt, copy, argsort, empty
import numpy as np
from scipy.spatial import cKDTree


def confine_particles(positions, v, x_max, y_max, r):
    max_boundaries = (x_max, y_max)
    for i in range(2):
        # Check if below lower limit
        is_outside = (((positions[:, i] < r).astype('intc') + (v[:, i] < 0).astype('intc')) == 2).astype('intc')
        outside_indices = argwhere(is_outside).flatten()
        v[outside_indices, i] *= -1
        positions[outside_indices, i] = r

        # Check if above upper limit
        is_outside = (((positions[:, i] > max_boundaries[i] - r).astype('intc') +
                       (v[:, i] > 0).astype('intc')) == 2).astype('intc')
        outside_indices = argwhere(is_outside).flatten()
        v[outside_indices, i] *= -1
        positions[outside_indices, i] = max_boundaries[i] - r


@njit
def handle_collisions(positions, v, radius):
    collision_indices = zeros((2, 1))
    r2 = 2*radius
    population_size = positions.shape[0]
    for i in arange(population_size):
        distances = -(positions - positions[i])
        x_dist = distances[:, 0]
        y_dist = distances[:, 1]
        distances_sq = sum(square(distances), axis=1)
        distances_sq[i] = (2*r2) ** 2
        for j in argwhere(distances_sq < r2 ** 2).flatten():
            collision_indices = append(collision_indices, array([[i], [j]]), axis=1)
            vel_a, vel_b = v[i], v[j]
            x_vel = vel_b[0] - vel_a[0]
            y_vel = vel_b[1] - vel_a[1]
            dot_prod = x_dist[j] * x_vel + y_dist[j] * y_vel
            if dot_prod > 0:
                dist_squared = distances_sq[j]
                collision_scale = dot_prod / dist_squared
                x_collision = x_dist[j] * collision_scale
                y_collision = y_dist[j] * collision_scale
                combined_mass = radius ** 3 + radius ** 3
                collision_weight_a = 2 * radius ** 3 / combined_mass
                collision_weight_b = 2 * radius ** 3 / combined_mass
                v[i, 0] += collision_weight_a * x_collision
                v[i, 1] += collision_weight_a * y_collision
                v[j, 0] -= collision_weight_b * x_collision
                v[j, 1] -= collision_weight_b * y_collision

    collision_indices = collision_indices[:, 1:]
    return collision_indices


@njit
def get_collision_indices(positions, radius):
    r2 = (2 * radius) ** 2
    n = positions.shape[0]
    distances = np.zeros((n, n))

    for coordinate in range(2):
        pos = positions[:, coordinate]
        distances += subtract(pos, np.ascontiguousarray(pos).reshape((n, 1))) ** 2

    fill_diagonal(distances, 2*r2)
    index_tuple = np.where(distances < r2)
    indices = np.zeros((2, index_tuple[0].size))
    indices[0, :], indices[1, :] = index_tuple[0], index_tuple[1]
    return indices


def get_collision_indices_q_tree(positions, radius, limits):
    tree = cKDTree(positions, boxsize=limits)
    return tree.query_pairs(2*radius, p=2, output_type='ndarray')


@njit
def handle_collisions_given_indices(positions, v, radius, indices):
    for n in arange(indices.shape[0]):
        i, j = indices[n, 0], indices[n, 1]
        x_dist, y_dist = (positions[i] - positions[j])
        vel_a, vel_b = v[i], v[j]
        x_vel = vel_b[0] - vel_a[0]
        y_vel = vel_b[1] - vel_a[1]
        dot_prod = x_dist * x_vel + y_dist * y_vel
        if dot_prod > 0:
            dist_squared = x_dist**2 + y_dist**2
            collision_scale = dot_prod / dist_squared
            x_collision = x_dist * collision_scale
            y_collision = y_dist * collision_scale
            combined_mass = radius ** 3 + radius ** 3
            collision_weight_a = 2 * radius ** 3 / combined_mass
            collision_weight_b = 2 * radius ** 3 / combined_mass
            v[i, 0] += collision_weight_a * x_collision
            v[i, 1] += collision_weight_a * y_collision
            v[j, 0] -= collision_weight_b * x_collision
            v[j, 1] -= collision_weight_b * y_collision


@njit
def energy_correction(v_before, v_after, collision_indices):
    indices = unique(collision_indices)
    energy_before = sum(sum(v_before[indices] ** 2, axis=1))
    energy_after = sum(sum(v_after[indices] ** 2, axis=1))
    v_after[indices] *= sqrt(energy_before / energy_after)


@njit
def move(positions, v):
    positions += v


@njit
def reformat_indices(indices):
    unique_indices = unique(indices)
    mapping = Dict.empty(int64, int64)
    for i in arange(len(unique_indices)):
        mapping[unique_indices[i]] = i

    reformatted_indices = zeros(indices.shape, dtype=int64)
    for i in arange(indices.shape[1]):
        reformatted_indices[0, i] = mapping[indices[0, i]]
        reformatted_indices[1, i] = mapping[indices[1, i]]
    return reformatted_indices


@njit
def reformat_indices_given_unique(indices, unique_indices):
    mapping = Dict.empty(int64, int64)
    for i in arange(unique_indices.size):
        mapping[unique_indices[i]] = i

    reformatted_indices = zeros(indices.shape, dtype=int64)
    for i in arange(indices.shape[1]):
        reformatted_indices[0, i] = mapping[indices[0, i]]
        reformatted_indices[1, i] = mapping[indices[1, i]]
    return reformatted_indices


@njit
def get_adjacency_list(indices):
    return append(indices, indices[::-1], axis=1)


@njit
def rank(a):
    arr = a.flatten()
    sorter = argsort(arr)

    inv = empty(sorter.size, dtype=intp)
    inv[sorter] = arange(sorter.size, dtype=intp)

    arr = arr[sorter]
    obs = append(array([True]), arr[1:] != arr[:-1])
    dense = obs.cumsum()[inv]

    return dense.reshape(a.shape)-1


@njit
def unique(arr):
    return np.unique(arr)


@njit
def indices_in_zone(limits, pos):
    return np.where(
        np.logical_and(pos[:, 0] >= limits[0, 0], pos[:, 0] <= limits[0, 1]) &
        np.logical_and(pos[:, 1] >= limits[1, 0], pos[:, 1] <= limits[1, 1]))[0]


@njit
def fully_connected_adjacency(indices):
    m = len(indices)
    n = m - 1
    indices = np.arange(m).astype(int32)
    inds1 = np.array([[j for j in range(m) for i in range(n)]], dtype=int32)
    inds2 = np.zeros((1, m * n), dtype=int32)
    for i in indices:
        if i != n:
            inds2[0, n*i:n*i + n] = np.delete(indices, i)
        else:
            inds2[0, n*i:] = np.delete(indices, i)
    return np.append(inds1, inds2, axis=0)