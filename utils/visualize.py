from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2

CUBE = np.array([[-1, -1, -1], [-1, -1,  1], [-1,  1,  1], [1, 1, 1], [ 1, -1,  1], 
                     [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1], [-1, -1, -1], [1, -1, -1], 
                     [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [1, 1, -1], [1, 1, 1]] )

STICK = np.array([[0, 0, -1], [0, 0, 0], [0, 0, 1]])

def visualize_object(x, y, z, transformation, obj='stick', ax=None, 
                     fix_axes=True, plot_origin=False, plot_axis=False, no_plot=False):
    if obj == 'cube':
        obj = CUBE
    elif obj == 'stick':
        obj = STICK
    else:
        raise Exception('No such object: Use either "stick" or "cube".')
    
    line_width = 6. if ax == None else 2.
    
    if ax == None:
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d', aspect='equal', proj_type='ortho')
    else:
        fig = ax.figure
        canvas = FigureCanvas(fig)
        
    axis = transformation[:-1]
    axis /= np.linalg.norm(axis)
    quaternion = Quaternion(*axis, transformation[-1])

    rot_obj = []
    
    for i, p in enumerate(obj):
        new_point = quaternion.rotate(p)
        new_point += [x, y, z]
        rot_obj.append(new_point)
    
    rot_obj = np.array(rot_obj)
    
    for (fx, fy, fz), (tx, ty, tz) in zip(rot_obj[:-1], rot_obj[1:]):
        ax.plot([fx, tx], [fy, ty], [fz, tz], linewidth=line_width)
    ax.axis('off')
    if fix_axes:
        ax.set_xlim((-1.6, 1.6))
        ax.set_ylim((-1.6, 1.6))
        ax.set_zlim((-1.6, 1.6))
    if plot_origin:
        ax.scatter([0.], [0.], [0.], c='b', marker=',')
    if plot_axis:
        ax.quiver([0], [0], [0], [axis[0]], [axis[1]], [axis[2]])
    
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    
    im = np.fromstring(s, np.uint8).reshape(height, width, 4)
    min_col = np.min(np.where((im[:,:,3] != 0.).any(axis=0)))
    max_col = np.max(np.where((im[:,:,3] != 0.).any(axis=0)))
    min_row = np.min(np.where((im[:,:,3] != 0.).any(axis=1)))
    max_row = np.max(np.where((im[:,:,3] != 0.).any(axis=1)))
    im = im[min_row:max_row, min_col:max_col]
    im = cv2.resize(im, tuple((np.array(im.shape)[[1, 0]] * 0.3).astype(np.int32)))
    
    if no_plot:
        plt.close()

    return im

    #return ax


def visualize_in_latent_space(latent_positions, object_quaternions, limits=1., num=500, obj='cube', filter_trajectories=False):
    plt.figure(figsize=(18, 6))
    
    filtered = np.array(list(filter(lambda x: abs(x[0]) < limits and abs(x[1]) < limits, latent_positions)))
    
    # choose positions in the latent space from trajectories
    if filter_trajectories:
        point_idx = np.random.choice(range(len(latent_positions)//100), size=num) * 100
    else:
        point_idx = np.random.choice(range(len(latent_positions)), size=num)
    latent_prop = latent_positions[point_idx]
    acc_object_quats = object_quaternions.to_numpy()[point_idx]

    img = np.ones([1000, 1000, 4])
    max_pos = np.max(latent_prop, axis=0)
    min_pos = np.min(latent_prop, axis=0)
    scale = 800./(max_pos - min_pos)

    positions = []
    taken_positions = np.zeros([20, 20], dtype=np.bool)

    get_position = lambda x: (x-min_pos) * scale + 100

    # create coordinate frame
    measures_start = (np.floor(min_pos * 10)).astype(np.int32)/10.
    measures_end = (np.ceil(max_pos * 10)).astype(np.int32)/10.
    n_measures_x = np.round(measures_end[0] - measures_start[0], 1) * 10 + 1
    n_measures_y = np.round(measures_end[1] - measures_start[1], 1) * 10 + 1

    for y in np.linspace(measures_start[1], measures_end[1], n_measures_y):
        y = np.round(y, 1)
        yp = int(get_position((0,y))[1])
        if y % 0.5 == 0 and y % 1 != 0:
            img = cv2.line(img, (0, yp), (1000, yp), (.8,.8,.8))
        elif y % 1 == 0:
            img = cv2.line(img, (0, yp), (1000, yp), (.4,.4,.4))
            img = cv2.putText(img, str(y), (5, yp-2), 1, 2., 1., 2)
        else:
            img = cv2.line(img, (0, yp), (1000, yp), (.9,.9,.9))
    for x in np.linspace(measures_start[0], measures_end[0], n_measures_x):
        x = np.round(x, 1)
        xp = int(get_position((x,0))[0])
        if x % 0.5 == 0 and x % 1 != 0:
            img = cv2.line(img, (xp, 0), (xp, 1000), (.8,.8,.8))
        elif x % 1 == 0:
            img = cv2.line(img, (xp, 0), (xp, 1000), (.4,.4,.4))
            img = cv2.putText(img, str(x), (xp, 30), 1, 2., 1., 2)
        else:
            img = cv2.line(img, (xp, 0), (xp, 1000), (.9,.9,.9))


    for p, q in zip(latent_prop, acc_object_quats):
        pos = get_position(p)

        y_mesh, x_mesh = int(pos[0]//50.), int(pos[1]//50.)
        if taken_positions[y_mesh][x_mesh]:
            continue
        else:
            taken_positions[y_mesh][x_mesh] = True

        plot = visualize_object(*q[:3], q[3:], obj=obj, no_plot=True)
        h, w = plot.shape[:2]
        px = (pos - [h/2, w/2]).astype(np.int32)
        py = (pos + [h/2, w/2]).astype(np.int32)
        try:
            #dims = (py[1]-px[1])*(py[0]-px[0])*4
            #if np.sum(img[px[1]:py[1], px[0]:py[0], :] == 1) > dims*0.85:
            img[px[1]:py[1], px[0]:py[0], :] *= (plot/255.)
        except:
            pass
        positions.append(pos)

    print(np.min(positions, axis=0), np.max(positions, axis=0))
    return img