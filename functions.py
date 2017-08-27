import numpy as np
import matplotlib.pyplot as plt

def isInCircle(point_x, point_y, circle_x, circle_y, radius):
    """Check if a given point is within a given circle

    :param point_x, point_y: coordinates of the point
    :param circle_x, circle_y: coordinates of the circle center
    :param radius: radius of the circle
    :return: True if the point is inside the circle, False if the point is outside or on the circle
    """
    return ( (point_x - circle_x)**2 + (point_y - circle_y)**2 ) < radius**2

def fillWires(image):
    """Performs hole filling by morphological reconstruction by erosion on the given image"""
    from skimage.morphology import reconstruction

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image

    reconstructed = reconstruction(seed, mask, method='erosion')

    return reconstructed

def plotCircles(axes, circles, kwargs):
    """Plots a collection of circles on the given axes"""
    from matplotlib.collections import PatchCollection

    patches = []

    for circle in circles:
        if len(circle) == 3:
            y, x, r = circle
        elif len(circle) == 2:
            y, x = circle
            r = 1
        else:
            raise RuntimeError('Wrong number of elements to define circle: ' + str(len(circle)))
        patch = plt.Circle((x, y), r, **kwargs)
        patches.append(patch)

    p = PatchCollection(patches, match_original=True)
    axes.add_collection(p)

def randomColors(n):
    """Returns a list of n arrays that can be used as random colors"""
    colors = []
    for x in range(0, n):
        colors.append(np.random.rand(3,1))

    return colors

def arrayPlot(data, title='', colorbar_label='', percentages=False):
    """Plot type used for plotting properties of all fields in an array

    :param data: The data to be plotted, one number for each field. Only datasets with a square number of entries will work.
    :param title: title to be printed above the plot
    :param colorbar_label: colorbar label (duh)
    :param percentages: if True, the tics on the colorbar are labeled as percentages
    :return: the pyplot object, wtf?
    """
    from math import sqrt, ceil
    plt.figure(figsize=(6.5, 5))
    ax = plt.gca()

    l = sqrt(len(data))

    if l != ceil(l):
        raise ValueError('Dataset to be plotted using arrayPlot must have a square number of entries.',
                         len(data), 'is not square!')

    X = np.arange(0.5, l+1.5)
    Y = np.arange(0.5, l+1.5)

    X, Y = np.meshgrid(X, Y)

    Z = np.array(data).reshape((l, l)) # Makes Z a 8x8 2d array
    Z = np.transpose(Z) # Use this if fluence and diameter are flipped. Otherwise comment out.
    plt.pcolor(X, Y, Z, cmap='viridis')

    plt.title(title)

    plt.xlabel('Diameter')
    plt.ylabel('Dose')
    plt.axis([0.5, l+0.5, 0.5, l+0.5])
    ax.set_aspect('equal', adjustable='box-forced')

    if percentages:
        plt.colorbar(format='%1.0f %%', label=colorbar_label)
    else:
        plt.colorbar(label=colorbar_label)

    return plt

def getCircleOfOnes(radius):
    """Returns an array where all elements are 0 except for elements within a circle of the given radius, which are 1"""
    diameter = 2 * radius
    ones = np.ones((diameter, diameter))
    y, x = np.ogrid[-radius: radius, -radius: radius]
    mask = x ** 2 + y ** 2 <= radius ** 2
    circle = ones * mask

    return circle
