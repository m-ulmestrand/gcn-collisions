from matplotlib import pyplot as plt
import numpy as np
from collison_detection import move, confine_particles, get_collision_indices_q_tree, \
    handle_collisions_given_indices
import torch
from physics_gnn import GCN
from time import time


x_max = y_max = 20
n_particles = 50
radius = 0.5
positions = np.random.random((n_particles, 2))
positions[:, 0] *= x_max
positions[:, 1] *= y_max
angles = np.random.random(n_particles) * 2 * np.pi
speed = 0.2
v = np.zeros((n_particles, 2))
v[:, 0] = np.cos(angles) * speed
v[:, 1] = np.sin(angles) * speed


def benchmark():
    latent_lim = 500
    time_gnn_tot = 0
    time_algorithm_tot = 0
    time_both = 0
    i = 0
    while i < 2000:
        time_gnn = 0
        time_algorithm = 0
        if not i % 20:
            print(i)
        i += 1
        confine_particles(positions, v, x_max, y_max, radius)
        if i > latent_lim:
            start = time()
        collision_indices = get_collision_indices_q_tree(positions, radius, [x_max, y_max])
        if i > latent_lim:
            time_both += time() - start

        if i > latent_lim:
            start = time()
        handle_collisions_given_indices(positions, v, radius, collision_indices)
        if i > latent_lim:
            time_algorithm = time() - start

        is_collision = collision_indices.size > 0
        if is_collision:
            if i > latent_lim:
                start = time()
            collision_indices = torch.tensor(collision_indices.T, dtype=torch.long, requires_grad=False)
            unique_indices = torch.unique(collision_indices, sorted=False)
            pos = torch.tensor(positions[:], requires_grad=False, device=device, dtype=torch.float)
            pos[:, 0], pos[:, 1] = pos[:, 0] / x_max, pos[:, 1] / y_max
            vel = torch.tensor(v[:], requires_grad=False, device=device, dtype=torch.float) / speed
            features = torch.cat((pos, vel), 1)
            edge_index = torch.cat((collision_indices, collision_indices.flip(0)), 1).to(device)

            gnn(features, edge_index)[unique_indices] * speed
            if i > latent_lim:
                time_gnn = time() - start

        time_algorithm_tot += time_algorithm
        time_gnn_tot += time_gnn
        move(positions, v)

    # time_gnn_tot += time_both
    # time_algorithm_tot += time_both
    print('Time GNN:', time_gnn_tot)
    print('Time algorithm:', time_algorithm_tot)
    print("Shared time:", time_both)
    print('Algorithm/GNN ratio:', time_algorithm_tot/time_gnn_tot)


colormap = plt.cm.get_cmap('plasma')
colors = [colormap(0.2 + 0.6 * i / (n_particles - 1)) for i in range(n_particles)]

print('CUDA available:', torch.cuda.is_available())
device = torch.device("cuda:0")
gnn = GCN(4, 2, 32).to(device)
gnn.load_state_dict(torch.load('collision_gnn.pt'))


def main():
    benchmark()


if __name__ == '__main__':
    main()