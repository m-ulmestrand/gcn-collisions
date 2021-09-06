from matplotlib import pyplot as plt
import numpy as np
from collison_detection import move, confine_particles, handle_collisions_given_indices, \
    get_collision_indices_q_tree, reformat_indices, get_adjacency_list


x_max = y_max = 20
n_particles = 300
radius = 0.5
positions = np.random.random((n_particles, 2))
positions[:, 0] *= x_max
positions[:, 1] *= y_max
angles = np.random.random(n_particles) * 2*np.pi
speed = 0.1
v = np.zeros((n_particles, 2))
v[:, 0] = np.cos(angles) * speed
v[:, 1] = np.sin(angles) * speed
colormap = plt.cm.get_cmap('plasma')
colors = [colormap(0.2 + 0.6 * i/(n_particles-1)) for i in range(n_particles)]


fig, ax = plt.subplots()
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)
ax.set_aspect('equal', adjustable='box')
scatter_plot = ax.scatter([None]*n_particles, [None]*n_particles, s=750*radius**2, color=colors)

fig.canvas.draw()
plt.show(block=False)

n_train = 5000
x_train = []   # Features: (x, y, v_x, v_y)
edge_index = []
y_train = []   # Output: (v_x, v_y)

x_train_detection = []
y_train_detection = []

while len(y_train) < n_train:
    confine_particles(positions, v, x_max, y_max, radius)
    features = np.append(positions, v, axis=1)
    features[:, 0] /= x_max
    features[:, 1] /= y_max
    features[:, 2:] /= speed
    collision_indices = get_collision_indices_q_tree(positions, radius, [x_max, y_max])
    if collision_indices.size > 0:
        collision_indices = collision_indices.astype('intc')
        handle_collisions_given_indices(positions, v, radius, collision_indices)
        collision_indices = collision_indices.T
        unique_indices = np.unique(collision_indices)
        features = features[unique_indices]
        outputs = v[unique_indices] / speed
        edge_indices = reformat_indices(get_adjacency_list(collision_indices))
        edge_index.append(edge_indices)

        x_train.append(features)
        y_train.append(outputs)
        x_train_detection.append(positions)
        y_detect = np.zeros(len(positions))
        y_detect[unique_indices] = 1
        y_train_detection.append(y_detect)
        print(len(y_train))
    move(positions, v)

    scatter_plot.set_offsets(positions)

    fig.canvas.draw()
    fig.canvas.flush_events()

np.savez('edge_index.npz', *edge_index)
np.savez('x_train.npz', *x_train)
np.savez('y_train.npz', *y_train)
np.savez('x_train_detection.npz', *x_train_detection)
np.savez('y_train_detection.npz', *y_train_detection)


'''while len(y_train) < n_train:
    confine_particles(positions, v, x_max, y_max, radius)
    features = np.append(positions, v, axis=1)
    features[:, 0] /= x_max
    features[:, 1] /= y_max
    features[:, 2:] /= speed
    collision_indices = handle_collisions(positions, v, radius)
    if collision_indices.size > 0:
        collision_indices = collision_indices.astype('intc')
        features = features[collision_indices[0, :], :]
        outputs = v[collision_indices[0, :], :] / speed
        edge_index.append(np.argsort(np.argsort(collision_indices, axis=1), axis=1))
        x_train.append(features)
        y_train.append(outputs)
        print(len(y_train))
    move(positions, v)

    scatter_plot.set_offsets(positions)

    fig.canvas.draw()
    fig.canvas.flush_events()'''