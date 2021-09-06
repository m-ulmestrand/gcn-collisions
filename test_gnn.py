from matplotlib import pyplot as plt
import numpy as np
from collison_detection import move, confine_particles, get_collision_indices_q_tree, \
    handle_collisions_given_indices
import torch
from physics_gnn import GCN


x_max = y_max = 20
n_particles = 50
radius = 0.5
positions = np.random.random((n_particles, 2))
positions[:, 0] *= x_max
positions[:, 1] *= y_max
angles = np.random.random(n_particles) * 2*np.pi
speed = 0.2
v = np.zeros((n_particles, 2))
v[:, 0] = np.cos(angles) * speed
v[:, 1] = np.sin(angles) * speed
symmetry_testing = False


def test_symmetry():
    eps = 1e-13
    global x_max, y_max
    x_max = y_max = 21
    n_particles = 3
    radius = 0.5
    positions = np.random.random((n_particles, 2))
    inds = list(np.random.permutation(np.array([0, 1, 2])))
    i, j, k = inds.pop(), inds.pop(), inds.pop()
    positions[i, 0] = 10
    positions[i, 1] = 10 * np.sqrt(3)
    positions[j, 0] = 0 + eps
    positions[j, 1] = 0 + eps
    positions[k, 0] = 20 - eps
    positions[k, 1] = 0 + eps
    positions[:, 1] += 4 * radius
    angles = np.random.random(n_particles) * 2*np.pi
    angles[i] = -np.pi/2
    angles[j] = np.pi/6
    angles[k] = (1-1/6)*np.pi
    speed = 0.1
    v = np.zeros((n_particles, 2))
    v[:, 0] = np.cos(angles) * speed
    v[:, 1] = np.sin(angles) * speed


colormap = plt.cm.get_cmap('plasma')
colors = [colormap(0.2 + 0.6 * i/(n_particles-1)) for i in range(n_particles)]

print('CUDA available:', torch.cuda.is_available())
device = torch.device("cuda:0")
gnn = GCN(4, 2, 32).to(device)
gnn.load_state_dict(torch.load('collision_gnn.pt'))

fig, ax = plt.subplots()
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)
ax.set_aspect('equal', adjustable='box')
scatter_plot = ax.scatter([None]*n_particles, [None]*n_particles, s=750*radius**2, color=colors)

fig.canvas.draw()
plt.show(block=False)


def test_network():
    while True:
        confine_particles(positions, v, x_max, y_max, radius)
        collision_indices = get_collision_indices_q_tree(positions, radius, [x_max, y_max])
        # handle_collisions_given_indices(positions, v, radius, collision_indices)
        if collision_indices.size > 0:
            collision_indices = torch.tensor(collision_indices.T, dtype=torch.long, requires_grad=False)
            unique_indices = torch.unique(collision_indices, sorted=True)
            max_n_index = unique_indices[-1] + 1
            pos = torch.tensor(positions[:max_n_index], requires_grad=False, device=device, dtype=torch.float)
            pos[:, 0], pos[:, 1] = pos[:, 0] / x_max, pos[:, 1] / y_max
            vel = torch.tensor(v[:max_n_index], requires_grad=False, device=device, dtype=torch.float) / speed
            features = torch.cat((pos, vel), 1)
            edge_index = torch.cat((collision_indices, collision_indices.flip(0)), 1).to(device)
            # v_before = np.copy(v)
            new_v = gnn(features, edge_index)[unique_indices] * speed
            v[unique_indices] = new_v.cpu().detach().numpy()
            # energy_correction(v_before, v, collision_indices[0])

        move(positions, v)
        scatter_plot.set_offsets(positions)

        fig.canvas.draw()
        fig.canvas.flush_events()


def test_algorithm():
    while True:
        confine_particles(positions, v, x_max, y_max, radius)
        collision_indices = get_collision_indices_q_tree(positions, radius, [x_max, y_max])
        handle_collisions_given_indices(positions, v, radius, collision_indices)

        move(positions, v)
        scatter_plot.set_offsets(positions)

        fig.canvas.draw()
        fig.canvas.flush_events()


def main():
    if symmetry_testing:
        test_symmetry()
    test_network()


if __name__ == '__main__':
    main()

'''while True:
    move(positions, v)
    confine_particles(positions, v, x_max, y_max, radius)
    # collision_indices = handle_collisions(positions, v, radius)
    collision_indices = get_collision_indices(positions, radius)
    if collision_indices.shape[1] > 0:
        collision_indices = collision_indices.astype('intc')
        pos = positions[collision_indices[0, :]]
        pos[:, 0], pos[:, 1] = pos[:, 0]/x_max, pos[:, 1]/y_max
        vel = v[collision_indices[0, :]] / speed
        features = np.append(pos, vel, axis=1)
        features = torch.tensor(features, dtype=torch.float).to(device)
        edge_index = (np.argsort(np.argsort(collision_indices, axis=1), axis=1))
        print(edge_index)
        print(reformat_indices(get_adjacency_list(collision_indices)))
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
        v_before = np.copy(v)
        v[collision_indices[0, :], :] = gnn(features, edge_index).detach().cpu().numpy() * speed
        # energy_correction(v_before, v, collision_indices[0])

    scatter_plot.set_offsets(positions)

    fig.canvas.draw()
    fig.canvas.flush_events()'''