import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import misc
import os
import pickle

import functions as f
from lattice import Lattice
import detect

class Field:

    def __init__(self, Na, Nb, path, name, scale, ext='.tif'):
        self.Na = Na                     # Number of lattice sites in the a direction
        self.Nb = Nb                     # Number of lattice sites in the b direction
        self.number_of_points = Na * Nb  # Total number of lattice sites
        self.path = path                 # Path to the folder where images are stored
        self.name = name                 # Name of the file storing the image for this field, excluding file extension
        self.scale = scale               # Lenght of a pixel in nm
        self.ext = ext                   # File extension of the file storing the image for this field
        self.image_path = path + '/' + name + ext  # Full path to the file storing the image for this field
        self.blobs_path = path + '/data/' + name + '_blobs.p'  # Path to the file storing data about detected blobs
        self.lattice_path = path + '/data/' + name + '_lattice.p'  # Path to the file storing data about the lattice
        self.blobs_by_point_path = path + '/data/' + name + '_blobs_by_point.p' # Path to the file storing data about assignment of blobs to lattice points
        self.figure_path = path + '/figures/'  # Path to the folder where figures will be saved
        self.lattice = None        # The lattice
        self.blobs = np.array([])  # The detected blobs
        self.blobs_by_point = []   # The detected blobs assigned to their nearest lattice point

        # Make sure directories for storing data and figures exist
        data_dir = path + '/data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        fig_dir = path + '/figures'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

    def prepImage(self, kernel_size, prep_path):
        """Preprocess the image of the field by applying median filtering"""
        from scipy.signal import medfilt2d
        image = misc.imread(self.image_path, flatten=True)

        image = medfilt2d(image, kernel_size)

        if not os.path.exists(prep_path):
            os.makedirs(prep_path)

        path = prep_path + '/' + self.name + '.png'

        misc.imsave(path, image)
        print('Saved image ' + self.name)

    def detectBlobs(self, methods=(detect.droplets,)):
        """Detect blobs using up to several methods, and store them in self.blobs

            Keyword arguments:
            methods -- a tuple of methods to use for detecting blobs
        """

        image = cv2.imread(self.image_path)
        blobs_a = []
        for method in methods:
            blobs_detected = method(image)
            blobs_a.append(blobs_detected)

        blobs = np.concatenate(blobs_a)

        self.clearBlobsByPoint()

        self.blobs = blobs
        pickle.dump(blobs, open(self.blobs_path, 'wb'))
        print('Blobs detected for field ', self.name, ': ', blobs.shape[0], ' blobs', sep='')

    def getBlobs(self, methods=(detect.droplets,)):
        """Return all detected blobs for field. Load if possible, detect if necessary."""
        if self.blobs.shape[0] > 0:
            return self.blobs
        else:
            try:
                self.blobs = pickle.load(open(self.blobs_path, 'rb'))
                if self.blobs.shape[0] < 1:
                    print('Loaded blobs, but array was empty. Detecting blobs.')
                    self.detectBlobs(methods)
            except FileNotFoundError:
                print('Blobs file not found! Detecting blobs.')
                self.detectBlobs(methods)

        if self.blobs.shape[0] > 0:
            return self.blobs
        else:
            raise RuntimeError('Not able to obtain blobs!')

    def clearBlobs(self):
        """Delete all stored and loaded information about blobs for this tile"""
        self.blobs = np.array([])

        try:
            os.remove(self.blobs_path)
        except FileNotFoundError:
            pass

        self.clearBlobsByPoint()

    def makeLattice(self):
        """Generate and save a lattice for the field by user input and lattice optimization"""
        from math import floor

        image = cv2.imread(self.image_path)

        good_guess = False
        while not good_guess:
            fig, ax = plt.subplots(figsize=(24, 12))
            ax.set_title('Please click 3 points to define an initial guess for lattice. ' +
                         'If you don\'t know what points to click, please find instructions in the documentation. ' +
                         'If these instructions do not yet exist, you might have a problem.')
            ax.imshow(image, cmap='gray')
            plt.get_current_fig_manager().window.showMaximized()

            print('Please input points to define an initial guess for lattice.')
            points = plt.ginput(3)
            plt.close()

            adjust = floor((self.Nb - 1) / 2)

            offset = np.array(points[0])
            vec_a = (np.array(points[1]) - offset) / (self.Na - 1)
            vec_b = (np.array(points[2]) - offset + vec_a * (adjust - self.Na + 1)) / (self.Nb - 1)

            self.lattice = Lattice(self.Na, self.Nb, vec_a, vec_b, offset)

            self.plotLattice()

            answer = input('Does the lattice look decent? (Y/N) ')
            if answer == 'y' or answer == 'Y':
                good_guess = True
            else:
                print('Try again.')

        print('Optimizing lattice')
        self.lattice = self.optimizeLattice(self.lattice)
        print('Lattice optimized')
        self.plotLattice()
        print('Saving new lattice')
        self.clearBlobsByPoint()
        pickle.dump(self.lattice, open(self.lattice_path, 'wb'))

    def optimizeLattice(self, lattice):
        """Optimize the given lattice to fit with the detected blobs"""

        def getRSS(params, Na, Nb, blobs_by_point):
            """Return the sum of squared distances between all blobs and their lattice point"""
            vax, vay, vbx, vby, ox, oy = params
            lattice = Lattice(Na, Nb, [vax, vay], [vbx, vby], [ox, oy])

            lattice_points = lattice.getLatticePoints()
            sum = 0

            for i, point in enumerate(blobs_by_point):
                point_x, point_y = lattice_points[i]
                for blob in point:
                    blob_y, blob_x, r = blob
                    square_dist = (point_x - blob_x) ** 2 + (point_y - blob_y) ** 2

                    sum += square_dist

            return sum

        def fixParams(params):
            """Help function for optimizeLattice

                Format the parameters given by lattice.getParams to be used by scipy.optimize.minimize
            """
            vax = params[2][0]
            vay = params[2][1]
            vbx = params[3][0]
            vby = params[3][1]
            ox = params[4][0]
            oy = params[4][1]

            return vax, vay, vbx, vby, ox, oy

        from scipy.optimize import minimize
        params = np.array(fixParams( lattice.getParams() ))
        res = minimize(getRSS, params, args=(self.Na, self.Nb, self.getBlobsByPoint()), method='Nelder-Mead')

        vax, vay, vbx, vby, ox, oy = res['x']
        lattice = Lattice(self.Na, self.Nb, [vax, vay], [vbx, vby], [ox, oy])

        return lattice

    def readjustLattice(self):
        """If field already has lattice defined, readjust lattice to fit best with current detected blobs"""
        found = True
        if self.lattice == None:
            try:
                self.lattice = pickle.load(open(self.lattice_path, 'rb'))
                if self.lattice == None:
                    found = False
            except FileNotFoundError:
                found = False

        if found:
            self.lattice = self.optimizeLattice(self.lattice)
            pickle.dump(self.lattice, open(self.lattice_path, 'wb'))
            print('Lattice', self.name, 'readjusted')

            return 1

        else:
            print('No lattice to adjust for field', self.name)

            return 0

    def getLattice(self):
        """Return lattice object for field. Load if possible, make if necessary."""
        if self.lattice != None:
            return self.lattice
        else:
            try:
                self.lattice = pickle.load(open(self.lattice_path, 'rb'))
                if self.lattice == None:
                    print('Loaded lattice, but object was empty.')
                    self.makeLattice()
            except FileNotFoundError:
                print('Lattice file not found!')
                self.makeLattice()

        return self.lattice

    def clearLattice(self):
        """Delete all stored and loaded information about the lattice for this tile"""
        self.lattice = None

        try:
            os.remove(self.lattice_path)
        except FileNotFoundError:
            pass

        self.clearBlobsByPoint()

    def generateBlobsByPoint(self):
        """Assigns all blobs to their nearest lattice point, and stores the list of blobs by point"""
        blobs = self.getBlobs()
        lattice = self.getLattice()

        lattice_points = lattice.getLatticePoints()
        lattice_distance = lattice.getMinLatticeDist()
        self.blobs_by_point = []

        for point in lattice_points:
            blobs_for_this_point = []

            for blob in blobs:
                y, x, r = blob

                if f.isInCircle(x, y, point[0], point[1], lattice_distance / 2):
                    blobs_for_this_point.append(blob)

            self.blobs_by_point.append(blobs_for_this_point)

        pickle.dump(self.blobs_by_point, open(self.blobs_by_point_path, 'wb'))
        print('Blobs assigned for field', self.name)

    def getBlobsByPoint(self):
        """Return all detected blobs for field. Load if possible, detect if necessary."""
        if len(self.blobs_by_point) > 0:
            return self.blobs_by_point
        else:
            try:
                self.blobs_by_point = pickle.load(open(self.blobs_by_point_path, 'rb'))
                if len(self.blobs_by_point) < 1:
                    print('Loaded blobs by point, but array was empty. Assigning blobs.')
                    self.generateBlobsByPoint()
            except FileNotFoundError:
                print('Blobs per point file not found! Assigning blobs.')
                self.generateBlobsByPoint()

        if len(self.blobs_by_point) > 0:
            return self.blobs_by_point
        else:
            raise RuntimeError('Not able to obtain blobs by point!')

    def clearBlobsByPoint(self):
        """Delete all stored and loaded information about blobs by point for this tile"""
        self.blobs_by_point = []

        try:
            os.remove(self.blobs_by_point_path)
        except FileNotFoundError:
            pass

    def getBlobCount(self):
        """Return the number of detected blobs"""
        blobs = self.getBlobs()
        return blobs.shape[0]

    def getDiameters(self):
        """Return list of diameters of detected blobs in nm"""
        blobs = self.getBlobs()
        diameters = blobs[:, 2] * 2 * self.scale

        return diameters

    def getMeanDiameter(self):
        """Return mean diameter of detected blobs in nm"""
        return np.mean(self.getDiameters())

    def getMedianDiameter(self):
        """Return median diameter of detected blobs in nm"""
        return np.median(self.getDiameters())

    def getBlobCountByPoint(self):
        """Return list of count of assigned blobs for each lattice point"""
        blobs_by_point = self.getBlobsByPoint()
        return [len(point) for point in blobs_by_point]

    def getDisplacements(self):
        """Returns an array of x and y displacement of blobs from their lattice point"""
        lattice_points = self.getLattice().getLatticePoints()
        blobs_by_point = self.getBlobsByPoint()
        displacements = []

        for i, point in enumerate(blobs_by_point):
            point_x, point_y = lattice_points[i]
            for blob in point:
                blob_y, blob_x, r = blob

                displacements.append([ blob_x - point_x, blob_y - point_y ])

        return displacements

    def getDisplacementMagnitudes(self):
        """Returns an array of displacement magnitudes of blobs from their lattice point"""
        displacements = self.getDisplacements()
        return [np.linalg.norm(displacement) * self.scale for displacement in displacements]

    def getDisplacementAngles(self):
        """Returns an array of displacement angles of blobs from their lattice point"""
        displacements = self.getDisplacements()
        angles = [np.angle( d[0] - 1j*d[1] ) for d in displacements]

        return angles

    def getYields(self):
        """Returns an array of the yield numbers for n blobs per point for all applicable values of n"""
        blob_count_by_point = self.getBlobCountByPoint()
        yields = []

        for n in range(0, max(blob_count_by_point) + 1):
            yields.append( blob_count_by_point.count(n) )

        return yields

    def getYield(self, n, percentage=False):
        """Returns the number or yield percentage for n blobs per point for a given value of n"""
        yields = self.getYields()

        try:
            number = yields[n]
        except IndexError:
            number = 0

        if percentage:
            return number * 100 / self.number_of_points
        else:
            return number

    def plotBlobs(self, show_image=True, save=False, prefix='', postfix=''):
        """Plot detected blobs

            Keyword arguments:
            show_image -- if False, blobs are plotted without the background image
            save -- if True, the figure is saved as a file, if False, the figure is displayed
            prefix, postfix -- appended and prepended to the name of the saved figure
        """

        blobs = self.getBlobs()

        fig, ax = plt.subplots(figsize=(24, 24))
        ax.set_aspect('equal', adjustable='box-forced')
        plt.axis((0, 1024, 883, 0))
        if show_image:
            image = cv2.imread(self.image_path)
            plt.imshow(image, cmap='gray', interpolation='nearest')
            plt.axis((0, image.shape[1], image.shape[0], 0))
        f.plotCircles(ax, blobs, fig, dict(color='red', linewidth=2, fill=False))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.tight_layout()
        fig.subplots_adjust(0, 0, 1, 1)

        fig_name = 'blobs'
        if save:
            full_path = self.figure_path + fig_name + '_' + prefix + self.name + postfix + '.png'
            plt.savefig(full_path)
            print('Saved', fig_name, 'plot for field', self.name)
        else:
            plt.show()
        plt.close()

    def plotLattice(self, lattice_color='red', figsize=(10, 10), save=False, prefix='', postfix=''):
        """Plot lattice points

            Keyword arguments:
            lattice_color -- the color with which to plot the lattice points
            figsize -- size of the plotted figure
            save -- if True, the figure is saved as a file, if False, the figure is displayed
            prefix, postfix -- appended and prepended to the name of the saved figure
        """
        image = cv2.imread(self.image_path)
        lattice_points = self.getLattice().getLatticePoints()

        plt.figure(figsize=figsize)
        plt.imshow(image, cmap='gray')
        ax = plt.gca()
        ax.set_axis_off()

        x = [x for [x, y] in lattice_points]
        y = [y for [x, y] in lattice_points]

        plt.scatter(x, y, marker='.', color=lattice_color, zorder=10)

        plt.tight_layout()

        fig_name = 'lattice'
        if save:
            full_path = self.figure_path + fig_name + '_' + prefix + self.name + postfix + '.png'
            plt.savefig(full_path)
            print('Saved', fig_name, 'plot for field', self.name)
        else:
            plt.show()
        plt.close()

    def plotLatticeAndBlobs(self, blob_color='', lattice_color='cyan', figsize=(10, 10), save=False, prefix='', postfix=''):
        """Plot lattice points, and detected blobs colored by lattice point

            Keyword arguments:
            blob_color -- the color with which to plot the detected blobs, if set to '', blobs are given a random color for each lattice point
            lattice_color -- the color with which to plot the lattice points
            figsize -- size of the plotted figure
            save -- if True, the figure is saved as a file, if False, the figure is displayed
            prefix, postfix -- appended and prepended to the name of the saved figure
        """
        from matplotlib.collections import PatchCollection

        lattice_points = self.getLattice().getLatticePoints()
        blobs_by_point = self.getBlobsByPoint()
        image = cv2.imread(self.image_path)

        plt.figure(figsize=figsize)
        plt.imshow(image, cmap='gray')
        ax = plt.gca()
        ax.set_axis_off()

        if blob_color == '':
            colors = f.randomColors(len(lattice_points))
        else:
            colors = [blob_color] * len(lattice_points)

        patches = []

        for i, point in enumerate(blobs_by_point):
            color = colors[i]

            for blob in point: # Plot blobs
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                patches.append(c)

        x = [x for [x, y] in lattice_points]
        y = [y for [x, y] in lattice_points]

        plt.scatter(x, y, marker='.', color=lattice_color, zorder=10)

        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)

        plt.tight_layout()

        fig_name = 'blobs+lattice'
        if save:
            full_path = self.figure_path + fig_name + '_' + prefix + self.name + postfix + '.png'
            plt.savefig(full_path)
            print('Saved', fig_name, 'plot for field', self.name)
        else:
            plt.show()
        plt.close()

    def plotHistogram(self, property, bins=40, fontsize=20, save=False, prefix='', postfix=''):
        """Plot a histogram of a given property of the detected blobs

            :param property: the property to be plotted. Can be either 'diameter', 'distance' or 'angle'

            Keyword arguments:
            bins -- the number of bins used for the histogram
            fontsize -- size of the font used in the plot
            save -- if True, the figure is saved as a file, if False, the figure is displayed
            prefix, postfix -- appended and prepended to the name of the saved figure
        """
        if property == 'diameter':
            label = 'diameter [nm]'
            data = self.getDiameters()
        elif property == 'distance':
            label = 'displacement from lattice point [nm]'
            data = self.getDisplacementMagnitudes()
        elif property == 'angle':
            label = 'angle'
            data = self.getDisplacementAngles()
        else:
            raise ValueError("'" + property + "' is not a valid property")

        fig, ax = plt.subplots(1, 1, figsize=(6, 3), subplot_kw={'adjustable': 'box-forced'})

        ax.set_ylim((0, 70))
        ax.hist(data, bins=bins, range = [0, 300], edgecolor='none', color='#033A87')
        plt.xlabel(label, fontsize=fontsize)
        plt.ylabel('count', fontsize=fontsize)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        plt.tight_layout()

        fig_name = property + ' histogram'
        if save:
            full_path = self.figure_path + fig_name + '_' + prefix + self.name + postfix + '.png'
            plt.savefig(full_path)
            print('Saved', fig_name, 'plot for field', self.name)
        else:
            plt.show()
        plt.close()