import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
#import numpy as np
from network import *
import operator

scatter_marker_size = 1

def update(frame, inputs, main, outputs):

    network = inputs[0]
    points = inputs[1]
    #network.apply_random_input(np.random.rand(*network.shape))
    #network.update()
    points.set_array(network.get_values().flatten())

    network = main[0]
    points = main[1]
    lines = main[2]
    #network.update()
    points.set_array(network.get_values().flatten())
    flat = network.neurons.flatten()
    index = 0
    for n in flat:
        nx, ny, nz = n.position
        for u, w in n.upstream.iteritems():
            ux, uy, uz = u.position
            lines[index].set_data([nx, ux], [ny, uy])
            lines[index].set_3d_properties([nz, uz])
            lines[index].set_alpha(w)
            lines[index].set_color('k')
            if u.fired:
                lines[index].set_color('c')
            index = index + 1


    network = outputs[0]
    points = outputs[1]
    lines = outputs[2]
    #network.update()
    points.set_array(network.get_values().flatten())
    flat = network.neurons.flatten()
    index = 0
    for n in flat:
        nx, ny, nz = n.position
        for u, w in n.upstream.iteritems():
            ux, uy, uz = u.position
            lines[index].set_data([nx, ux], [ny, uy])
            lines[index].set_3d_properties([nz, uz])
            lines[index].set_alpha(w)
            if u.fired:
                lines[index].set_color('c')
            index = index + 1

    inputs[0].apply_random_input(np.random.rand(*inputs[0].shape))
    inputs[0].update()
    main[0].update()
    outputs[0].update()


def main():

    # init main network
    main_network_shape = (7, 7, 7)
    main_network_position = (0, 0, 0)
    main_network_spacing = (1, 1, 1)
    main_network = Network(Network.STDP_NEURON, main_network_shape, main_network_position, main_network_spacing, memory_gradient=True, threshold=(0.7, 0.5), decay_rate=(0.7, 0.5), max_weight_association_delta=(0.05, 0.05), max_weight_decay_delta=(0.05, 0.05),  init_random_values=True)
    main_network.connect(main_network, density=0.2, connection_weights=(0.1, 0.05))

    # init input sensor network
    sensor_network_shape = (1, main_network_shape[1], main_network_shape[2])
    sensor_network_offset_x = -sensor_network_shape[0]
    #sensor_network_offset_y = -main_network_shape[1] * main_network_spacing[1]/2.0
    #sensor_network_offset_z = -main_network_shape[2] * main_network_spacing[2]/2.0
    sensor_network_offset_y = 0
    sensor_network_offset_z = 0
    sensor_network_position = tuple(map(operator.add, main_network_position, (sensor_network_offset_x, sensor_network_offset_y, sensor_network_offset_z)))
    sensor_network_spacing = (1.0, 1.0, 1.0)
    sensor_network = Network(Network.SENSOR_NEURON, sensor_network_shape, sensor_network_position, sensor_network_spacing, init_random_values=True)
    sensor_network.connect(main_network, density=0.2, connection_weights=(0.1, 0.05))
    
    # init output neuron network
    output_network_shape = (1, 1, 1)
    output_network_offset_x = (main_network_shape[0] - 1) * main_network_spacing[0] + 1.0
    #output_network_offset_y = -main_network_shape[1] * main_network_spacing[1]/2.0
    #output_network_offset_z = -main_network_shape[2] * main_network_spacing[2]/2.0
    output_network_offset_y = 0
    output_network_offset_z = 0
    output_network_position = tuple(map(operator.add, main_network_position, (output_network_offset_x, output_network_offset_y, output_network_offset_z)))
    output_network_spacing = (1, 1, 1)
    output_network = Network(Network.STDP_NEURON, output_network_shape, output_network_position, output_network_spacing, init_random_values=True, ratio_inhibitory=0)
    output_network.connect(main_network, density=0.2, connection_weights=(0.1, 0.05))
    main_network.connect(output_network, density=0.2, connection_weights=(0.1, 0.05))

    fig = plt.figure()
    network_view = fig.add_subplot(111, projection='3d')

    # main network neuron positions
    main_network_flat = main_network.neurons.flatten()
    main_network_x = [n.position[0] for n in main_network_flat]
    main_network_y = [n.position[1] for n in main_network_flat]
    main_network_z = [n.position[2] for n in main_network_flat]    
   
    # sensor network neuron positions
    sensor_network_flat = sensor_network.neurons.flatten()
    sensor_network_x = [n.position[0] for n in sensor_network_flat]
    sensor_network_y = [n.position[1] for n in sensor_network_flat]
    sensor_network_z = [n.position[2] for n in sensor_network_flat]    
    
    # output network neuron positions
    output_network_flat = output_network.neurons.flatten()
    output_network_x = [n.position[0] for n in output_network_flat]
    output_network_y = [n.position[1] for n in output_network_flat]
    output_network_z = [n.position[2] for n in output_network_flat]


    # show  main_network connections
    flat = main_network.neurons.flatten()
    main_network_lines = []
    for n in flat:
        nx, ny, nz = n.position
        for u, w in n.upstream.iteritems():
            ux, uy, uz = u.position
            main_network_lines = main_network_lines + network_view.plot([nx, ux], [ny, uy], [nz, uz], color='black', alpha=w)

    """
    # show random sensor_network connections
    flat = sensor_network.neurons.flatten()
    index = np.random.randint(len(flat))
    n = flat[index]
    nx, ny, nz = n.position
    for d in n.downstream:
        dx, dy, dz = d.position
        network_view.plot([nx, dx], [ny, dy], [nz, dz], color='red')
    """

    # show output network connections
    flat = output_network.neurons.flatten()
    output_network_lines = []
    for n in flat:
        nx, ny, nz = n.position
        for u, w in n.upstream.iteritems():
            ux, uy, uz = u.position
            output_network_lines = output_network_lines + network_view.plot([nx, ux], [ny, uy], [nz, uz], color='blue', alpha=w)


    main_network_points = network_view.scatter(main_network_x, main_network_y, main_network_z, s=scatter_marker_size, depthshade=True, cmap=cm.hot, vmin=0, vmax=1.0, marker='o', c=main_network.get_values().flatten())
    sensor_network_points = network_view.scatter(sensor_network_x, sensor_network_y, sensor_network_z, s=scatter_marker_size, depthshade=True, cmap=cm.hot, vmin=0, vmax=1.0, marker='^', c=sensor_network.get_values().flatten())
    output_network_points = network_view.scatter(output_network_x, output_network_y, output_network_z, s=scatter_marker_size, depthshade=True, cmap=cm.hot, vmin=0, vmax=1.0, marker='s', c=output_network.get_values().flatten())
    network_center_point = network_view.scatter([main_network.network_center[0]], [main_network.network_center[1]], [main_network.network_center[2]], s=2*scatter_marker_size, cmap=cm.jet, vmin=0, vmax=1.0, c=[1.0])


    network_view.set_xlabel('X')
    network_view.set_ylabel('Y')
    network_view.set_zlabel('Z')

    inputs = [sensor_network, sensor_network_points]
    main = [main_network, main_network_points, main_network_lines]
    outputs = [output_network, output_network_points, output_network_lines]
    ani = animation.FuncAnimation(fig, update, fargs=(inputs, main, outputs), interval=10)
    plt.show()

if __name__ == "__main__":
    main()
