import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import random
import pickle
import os

from math import pi

import functions as f
from arbitraryLattice import Lattice, makeLatticeByAngles, loadLattice

from timeCheckpoint import timeCheckpoint
from time import clock

class Tileset:
    default_padding = 100

    def __init__(self, path, cols, rows, tilew, tileh, scale, detection_method, ext='.tif'):
        self.path = path    # Path to the folder where everything is stored
        self.ext = ext      # File extension for the tile images
        self.rows = rows    # Number of tile rows
        self.cols = cols    # Number of tile columns
        self.tileh = tileh  # Height of each tile in pixels
        self.tilew = tilew  # Width of each tile in pixels
        self.scale = scale  # Size of a pixel in nm
        self.detection_method = detection_method  # Method to be used for blob detection
        self.blobs = np.array([])  # The detected blobs
        self.assigned_blobs = []   # The detected blobs assigned to their nearest lattice point
        self.lattice = None        # The lattice


    def getTile(self, col, row):
        """ Load from file and return a tile specified by row and column.
        If there is no tile at the specified position, returns an array of zeroes with the same size as a tile.

        :param col: the column of the tile to be returned
        :param row: the row of the tile to be returned
        :return: numpy array of the tile image
        """
        file_path = self.path + '/c_' + str(col) + '/tile_' + str(row) + self.ext
        try:
            tile = misc.imread(file_path)
        except FileNotFoundError:
            tile = np.zeros((self.tileh, self.tilew), dtype=np.uint8)

        return tile

    def getTileRegion(self, col_min, col_max, row_min, row_max):
        """Return a region of the image by concatenating a set of tiles

        :param col_min, col_max, row_min, row_max: bounding rows and columns of the region to return
        :return: numpy array of an image spanning the specified region
        """
        r_width = self.tilew * (col_max - col_min + 1)
        r_height = self.tileh * (row_max - row_min + 1)
        region = np.zeros((r_height, r_width), dtype=np.uint8)

        for col in range(col_min, col_max+1):
            for row in range(row_min, row_max+1):

                tile = self.getTile(col, row)

                h_min = self.tileh * (row - row_min)
                h_max = self.tileh * (row - row_min + 1)
                w_min = self.tilew * (col - col_min)
                w_max = self.tilew * (col - col_min + 1)

                region[h_min:h_max, w_min:w_max] = tile

        return region

    def getPaddedTile(self, col, row, padding=default_padding):
        """Return a tile with padding from adjacent tiles

        :param col: the column of the tile to be returned
        :param row: the row of the tile to be returned
        :param padding: size of the padding in pixels
        :return: numpy array of the padded tile
        """
        if padding > min(self.tilew, self.tileh):
            raise RuntimeError('Padding of ' + str(padding) + ' is too large!')

        region = self.getTileRegion(col-1, col+1, row-1, row+1)

        h_crop = self.tileh - padding
        v_crop = self.tilew - padding

        padded = region[h_crop:-h_crop, v_crop:-v_crop]

        return padded

    def prepTiles(self, output_path, kernel_size, fill=True):
        """Do preprocessing on all tiles in tileset, and save the preprocessed tiles to output_path
        Preprocessing consists of filling in using reconstruction, and median filtering

        :param output_path: the path where the processed tiles will be saved
        :param kernel_size: width of the kernel used for median filtering
        :return: Tileset object containing the preprocessed tiles
        """
        from scipy.signal import medfilt2d

        for col in range(0, self.cols):
            col_path = output_path + '/c_' + str(col)
            if not os.path.exists(col_path):
                os.makedirs(col_path)

            for row in range(0, self.rows):
                tile = self.getPaddedTile(col, row)

                if fill:
                    tile = f.fillWires(tile)

                tile = medfilt2d(tile, kernel_size)

                if fill:
                    tile = f.fillWires(tile)

                p = self.default_padding
                cropped_tile = tile[p:-p, p:-p]

                filename = col_path + '/tile_' + str(row) + self.ext
                misc.imsave(filename, cropped_tile)
                print('Saved tile ' + str(col) + ', ' + str(row))

        return Tileset(output_path, self.cols, self.rows, self.tilew, self.tileh, self.scale, self.detection_method,
                       self.ext)

    def detectBlobs(self, col, row, globalize=False):
        """Run detection and return array of detected blobs for the specified tile

        :param col: column number of the tile on which to perform detection
        :param row: row number of the tile on which to perform detection
        :param globalize: if true, blobs will be returned with global coordinates
        :return: numpy array of the detected blobs
        """
        padded_tile = self.getPaddedTile(col, row)

        blobs = self.detection_method(padded_tile)  # detect blobs

        padding = self.default_padding
        outside = []
        for i, blob in enumerate(blobs):  # figure out which blobs lie outside the non-padded tile
            if min(blob[0:2]) < padding or blob[0] >= (self.tileh+padding) or blob[1] >= (self.tilew+padding):
                outside.append(i)

        blobs = np.delete(blobs, outside, 0)  # delete blobs that lie outside the non-padded tile, to avoid duplicates

        blobs[:, 0:2] -= padding  # readjust the coordinates of the blobs to be relative to the non-padded tile
        if globalize:  # convert the coordinates of the blob from coords within the tile to coords for the whole tileset
            blobs[:, 1] += col * self.tilew
            blobs[:, 0] += row * self.tileh

        print('Blobs found:', blobs.shape[0])

        return blobs

    def detectAllBlobs(self):
        """Run detection on all tiles, and save the result."""
        blobs = False
        for col in range(0, self.cols):
            for row in range(0, self.rows):
                found = self.detectBlobs(col, row, globalize=True)
                print('Detected blobs for tile ' + str(col) + ', ' + str(row))
                if blobs is False:
                    blobs = found
                else:
                    blobs = np.append(blobs, found, axis=0)

        self.blobs = blobs
        self.assigned_blobs = []

        self.saveBlobs()

    def saveBlobs(self):
        """Store all currently detected blobs to a file located at self.path"""
        full_path = self.path + '/blobs.p'
        pickle.dump(self.blobs, open(full_path, 'wb'))

    def deleteBlobs(self):
        """Delete the file and clear the variable containing detected blobs."""
        full_path = self.path + '/blobs.p'
        try:
            os.remove(full_path)
        except FileNotFoundError:
            print('No file to delete')
        self.blobs = np.array([])

    def getBlobs(self):
        """Return all detected blobs for tileset. Load if possible, detect if necessary."""
        if self.blobs.shape[0] > 0:
            return self.blobs
        else:
            try:
                full_path = self.path + '/blobs.p'
                self.blobs = pickle.load(open(full_path, 'rb'))
                if self.blobs.shape[0] < 1:
                    print('Loaded blobs, but array was empty. Detecting blobs.')
                    self.detectAllBlobs()
            except FileNotFoundError:
                print('Blobs file not found. Detecting blobs.')
                self.detectAllBlobs()

        if self.blobs.shape[0] > 0:
            return self.blobs
        else:
            raise Exception('Not able to obtain blobs!')

    def getSubsetOfBlobs(self, x_min, x_max, y_min, y_max):
        """Return all detected blobs for specified coordinate region. Load if possible, detect if necessary."""
        # Get all blobs
        blobs = self.getBlobs()

        # Remove the ones outside the specified area
        outside = []
        for i, blob in enumerate(blobs):
            if blob[0] < y_min or blob[0] > y_max or blob[1] < x_min or blob[1] > x_max:
                outside.append(i)

        blobs = np.delete(blobs, outside, 0)

        return blobs

    @staticmethod
    def findFirstBlob(blobs):
        """Helper function for makeLattice: Return the most top left blob in the given set of blobs."""
        vals = []
        for blob in blobs:
            vals.append(blob[0] + blob[1])  # x + y coordinate of blob

        i = np.argmin(vals)  # minimum x + y is top left

        return blobs[i]

    @staticmethod
    def getAnglesFromInput(tile, blobs, offset):
        """Helper function for makeLattice:
        Display an image with detected blobs plotted, and get user input to define angles for lattice vectors.

        :param tile: the tile to be displayed as background image
        :param blobs: the detected blobs for the tile
        :param offset: the point representing the origin of the lattice, angles are defined relative to this point
        :return: the two angles defined by the user input, in radians
        """
        fig, ax = plt.subplots(figsize=(24, 12))
        ax.set_aspect('equal', adjustable='box-forced')
        plt.axis((0, tile.shape[1], tile.shape[0], 0))
        plt.title("Please click two points")
        plt.tight_layout()

        plt.imshow(tile, cmap='gray', interpolation='nearest')
        f.plotCircles(ax, blobs, fig, dict(color='#114400', linewidth=4, fill=False))
        plt.plot(offset[0], offset[1], '.', color='red', markersize=10)

        # Get input
        points = plt.ginput(2)
        plt.close()

        # Calculate angles from input
        displacements = [np.array(point) - offset for point in points]
        angles = [np.angle(dis[0] + 1j*dis[1]) for dis in displacements]

        return angles

    @staticmethod
    def getTypicalDistance(blobs):
        """Helper function for makeLattice: Get an initial guess for the magnitude of lattice vectors by finding the
        typical distance between blobs.

        :param blobs: the blobs between which to find the typical distance
        :return: float representing the typical distance between blobs in pixels
        """

        def reject_outliers(data, m):
            """Removes outliers from a dataset using deviations from the median instead of the mean, since this is
            more robust, and less affected by outliers

            :param data: the data to be filtered, must be a numpy array of numbers
            :param m: the amount of median deviations from the median beyond which to discard data
            :return: all the data points laying within m median deviations from the median
            """
            d = np.abs(data - np.median(data))  # array of each number's deviation from the median value
            mdev = np.median(d)  # the median deviation from the median value
            s = d / mdev if mdev else 0.  # array of each number's deviation from the median value, given in
                                          # multiples of the median deviation from the median value
            return data[s < m]

        from scipy.spatial import KDTree
        points = blobs[:, 0:2]  # get just x and y coordinates of blobs, not the radii

        tree = KDTree(points)  # put the points into a data structure allowing for quick neighbor distance lookup
        results = [tree.query(points, 7)]  # return the six nearest points to each point

        distances = [result[1:7] for result in results[0][0]]  # get the actual distances
        distances = np.array(distances).flatten()  # make a flattened array of all the inter-point distances
        distances = reject_outliers(distances, 5)  # remove outliers beyond 5 median deviations from the median

        return np.mean(distances)  # return the median value of the filtered distances

    @staticmethod
    def optimizeLattice(lattice, assigned_blobs, debug=False):
        """Use given lattice as an initial guess, and numerically optimize lattice to minimize
        the sum of square distances between each blob and it's nearest lattice point.

        :param lattice: the initial guess for a lattice
        :param assigned_blobs: the blobs to which to fit the lattice, allready assigned to their nearest lattice point
        :param debug: if True, debug info will be printed
        :return: the optimized lattice
        """
        def getRSS(params, assigned_blobs):
            """Return the sum of the squared distance between each of the given blobs and it's nearest lattice point."""
            mag_a, ang_a, mag_b, ang_b, ox, oy = params
            lattice = makeLatticeByAngles(mag_a, ang_a, mag_b, ang_b, [ox, oy])

            sum = 0

            for blob_p in assigned_blobs:
                if blob_p['point'] != []:
                    blob_y, blob_x, r = blob_p['blob']
                    [point_x, point_y] = lattice.getCoordinates(*blob_p['point'])
                    square_dist = (point_x - blob_x) ** 2 + (point_y - blob_y) ** 2

                    sum += square_dist

            return sum

        from scipy.optimize import minimize
        params = np.array([lattice.len_a, lattice.ang_a, lattice.len_b, lattice.ang_b,
                           lattice.offset[0], lattice.offset[1]])

        print('Blobs: ' + str(len(assigned_blobs)))

        # Minimize the sum of square distances between blobs and their nearest lattice point, by adjusting the given
        # parameters.
        res = minimize(getRSS, params, args=(assigned_blobs), method='Nelder-Mead')

        mag_a, ang_a, mag_b, ang_b, ox, oy = res['x']  # the parameters found to give the best lattice fit
        lattice = makeLatticeByAngles(mag_a, ang_a, mag_b, ang_b, [ox, oy])

        if debug:
            print(mag_a, ang_a, mag_b, ang_b, ox, oy)

        return lattice

    def makeLattice(self, max_blobs=500, final_blobs=4000, step=3, debug=False):
        """Run the whole process necessary to get a lattice defined for the tileset, and save it to file.

        :param max_blobs: the maximum number of blobs to use for each round of optimization
        :param final_blobs: the maximum number of blobs to use for the final round of optimization
        :param step: how many new rows/columns to add for each new round of optimization
        :param debug: if True, some debug info will be printed, and extra steps will be shown
        """
        # Setup
        tw = self.tilew
        th = self.tileh
        bounds = (0, tw, 0, th)

        # The process starts with an initial guess based on the top left tile.
        tile = self.getTile(0, 0)
        blobs = self.getSubsetOfBlobs(*bounds)  # get the blobs for the top left tile
        # The top left blob is used as the offset for the initial lattice guess.
        first = self.findFirstBlob(blobs)
        offset = [first[1], first[0]]

        # Angles of the lattice vectors for the initial lattice guess are given by manual input.
        angles = self.getAnglesFromInput(tile, blobs, offset)
        if len(angles) < 2:
            raise RuntimeError("Insufficient input received.")
        # The magnitude of the lattice vectors for the initial lattice guess is given by the typical neighbor distance.
        magnitude = self.getTypicalDistance(self.getSubsetOfBlobs(0, 4*tw, 0, 4*th))

        lattice = makeLatticeByAngles(magnitude, angles[0], magnitude, angles[1], offset)
        assigned_blobs = self.assignBlobs(blobs, lattice)

        # Show the initial guess lattice to the user, to ensure input was not completely wrong
        self.lattice = lattice  # needs to be set for displayTileRegion
        self.displayTileRegion(0, 0, 0, 0, blob_color='green', lattice_color='red')

        lattice = self.optimizeLattice(lattice, assigned_blobs)
        print('Lattice optimized for first tile.')

        if debug:
            self.lattice = lattice  # needs to be set for displayTileRegion
            self.displayTileRegion(0, 0, 0, 0, blob_color='green', lattice_color='red')

        def optimizeWithBounds(self, lattice, bounds, max_blobs):
            """Optimize the given lattice to fit best with blobs selected from a region of the tileset

            :param self: the tileset object
            :param lattice:  the lattice to optimize
            :param bounds: bounds of the region from which to select blobs
            :param max_blobs: the max number of blobs to optimize against. If the total number of blobs in the region
                              specified by bounds is larger than max_blobs, a random selection of mox_blobs blobs from
                              the region is used
            :return: optimized lattice
            """
            blobs = self.getSubsetOfBlobs(*bounds)
            # If there are more than max_blobs blobs within bounds, get a random selection of max_blobs blobs
            if blobs.shape[0] > max_blobs:
                blobs_list = list(blobs)
                blobs_list = [blobs_list[i] for i in random.sample(range(len(blobs_list)), max_blobs)]
                blobs = np.array(blobs_list)

            assigned_blobs = self.assignBlobs(blobs, lattice)
            optimized_lattice = self.optimizeLattice(lattice, assigned_blobs)

            return optimized_lattice

        # Gradually expand the area for which the lattice is being optimized column by column
        for n in range(1, self.cols, step):
            bounds = (0, (n+1)*tw, 0, th)
            lattice = optimizeWithBounds(self, lattice, bounds, max_blobs)
            print('Lattice optimized for', n+1, 'of', self.cols, 'columns.')

        # Gradually expand the area for which the lattice is being optimized row by row
        for n in range(1, self.rows, step):
            bounds = (0, self.cols*tw, 0, (n+1)*th)
            lattice = optimizeWithBounds(self, lattice, bounds, max_blobs)
            print('Lattice optimized for', n+1, 'of', self.rows, 'rows.')

        # Run one last optimization, using a larger selection of blobs taken from the entire tileset
        # Optimization is never done for all blobs, as this would take a very long time, and a random selection is
        # sufficient if the selection is large enough.
        bounds = (0, self.cols * tw, 0, self.rows * th)
        lattice = optimizeWithBounds(self, lattice, bounds, final_blobs)
        print('Final optimization finished.')

        if debug:
            self.lattice = lattice  # needs to be set for displayTileRegion
            self.assignBlobs()
            self.displayTileRegion(0, 0, 0, 0, blob_color='green', lattice_color='red')

        self.lattice = lattice
        self.saveLattice()
        self.deleteAssignedBlobs()

    def saveLattice(self):
        """Save the lattice stored in self.lattice to a file located at self.path"""
        full_path = self.path + '/lattice.p'
        pickle.dump(self.lattice, open(full_path, 'wb'))

    def deleteLattice(self):
        """Delete the file and clear the variable containing the lattice."""
        full_path = self.path + '/lattice.p'
        try:
            os.remove(full_path)
        except FileNotFoundError:
            print('No file to delete')
        self.lattice = None

    def getLattice(self):
        """Obtain a lattice by whatever means necessary. Try the following order:
        1: return self.lattice
        2: load lattice from file
        3: generate new lattice
        """
        if self.lattice != None:
            return self.lattice
        else:
            try:
                full_path = self.path + '/lattice.p'
                self.lattice = pickle.load(open(full_path, 'rb'))
                if self.lattice == None:
                    print('Loaded lattice, but array was empty.')
                    self.makeLattice()
            except FileNotFoundError:
                print('Lattice file not found!')
                self.makeLattice()

        return self.lattice

    def assignBlobs(self, blobs=None, lattice=None, save=True):
        """Assign a set of blobs to a lattice. Each blob is assigned to it's nearest lattice point.
        Return an array of dictionaries, each dictionary representing a blob, and containing the following:
        ['blob']: y, x, and r of the blob
        ['point']: lattice indices of the nearest lattice point
        ['distance']: absolute distance to the nearest lattice point
        ['angle']: angle of the displacement vector from blob to point

        :param blobs: the blobs to be assigned to lattice points, if none is given, self.getBlobs() is used
        :param lattice: the lattice to which to assign the bobs, if none is given, self.getLattice() is used
        :param save: if True, self.blobs will be set to the result, and assigned blobs will be saved to file
                     if False, the result will be returned, but not saved
        :return: described above
        """
        from scipy.spatial import KDTree
        checkpoint = clock()

        if blobs == None:
            blobs = self.getBlobs()
        if lattice == None:
            lattice = self.getLattice()

        assigned_blobs = [{'blob': blob} for blob in blobs]
        radius = lattice.getMinLatticeDist()/2

        x_min = min(blobs[:, 1]) - radius
        x_max = max(blobs[:, 1]) + radius
        y_min = min(blobs[:, 0]) - radius
        y_max = max(blobs[:, 0]) + radius

        points = lattice.getLatticePoints(x_min, x_max, y_min, y_max)
        tree = KDTree(points)

        for a_blob in assigned_blobs:
            y, x, r = a_blob['blob']
            distance, index = tree.query([x, y])
            point = tree.data[index]
            a_blob['point'] = lattice.getIndices(point[0], point[1])
            a_blob['distance'] = distance
            dis = np.array(point) - np.array([x, y])
            a_blob['angle'] = np.angle(dis[0] + 1j * dis[1])

        timeCheckpoint(checkpoint, 'assigning blobs')

        if save:
            self.assigned_blobs = assigned_blobs
            self.saveAssignedBlobs()

        return assigned_blobs

    def saveAssignedBlobs(self):
        """Store all currently assigned blobs to a file located at self.path."""
        full_path = self.path + '/assigned_blobs.p'
        pickle.dump(self.assigned_blobs, open(full_path, 'wb'))

    def deleteAssignedBlobs(self):
        """Delete the file and clear the variable containing assigned blobs."""
        full_path = self.path + '/assigned_blobs.p'
        try:
            os.remove(full_path)
        except FileNotFoundError:
            print('No file to delete')
        self.assigned_blobs = []

    def getAssignedBlobs(self):
        """Return assigned blobs for tileset. Load if possible, detect and assign if necessary."""
        if len(self.assigned_blobs) > 0:
            return self.assigned_blobs
        else:
            try:
                full_path = self.path + '/assigned_blobs.p'
                self.assigned_blobs = pickle.load(open(full_path, 'rb'))
                if len(self.assigned_blobs) < 1:
                    print('Loaded assigned blobs, but array was empty. Assigning blobs.')
                    blobs = self.getBlobs()
                    lattice = self.getLattice()
                    self.assignBlobs(blobs, lattice)
            except FileNotFoundError:
                print('Assigned blobs file not found. Assigning.')
                blobs = self.getBlobs()
                lattice = self.getLattice()
                self.assignBlobs(blobs, lattice)

        if len(self.assigned_blobs) > 0:
            return self.assigned_blobs
        else:
            raise Exception('Not able to obtain assigned blobs!')

    def clearAll(self):
        """Clear all stored data about blobs, lattice and assignment of blobs to lattice"""
        self.deleteBlobs()
        self.deleteLattice()
        self.deleteAssignedBlobs()

    def getBlobCountPerPoint(self, region=None):
        """
        Get a list of all lattice points containing blobs, and the number of blobs they contain.

        :param region: list of 4 ints
            if given, this denotes the limits of the subregion from which to get the list of points
            if nothing is given, the whole tileset is used
        :return: list of dicts
            a list of dictionaries containing indices of each lattice point containing blobs, and the number of blobs
            in it's neighborhood
            the dictionary has the following elements:
            'point': a list of 2 ints, representing the indices of the lattice point
            'count': the number of blobs that have this lattice point as their nearest lattice point
        """
        checkpoint = clock()

        if region == None:
            assignedBlobs = self.getAssignedBlobs()
        elif len(region) != 4:
            raise RuntimeError("'region' must have exactly 4 elements (x_min, x_max, y_min, y_max)")
        else:
            blobs = self.getSubsetOfBlobs(*region)
            assignedBlobs = self.assignBlobs(blobs, self.getLattice())

        checkpoint = timeCheckpoint(checkpoint, 'get blobs')

        # Sort the blobs by lattice point
        sortedBlobs = sorted(assignedBlobs, key = lambda x: (x['point'][0], x['point'][1]))

        points = [{'indices': sortedBlobs[0]['point'], 'count': 1}]  # Initialize the dict

        for i, blob in enumerate(sortedBlobs):  # Go through the sorted blobs
            if i == 0:
                continue  # Skip the first blob, as it is allready counted
            if blob['point'] == sortedBlobs[i-1]['point']:  # If this blob belongs to the same point as the last blob
                points[-1]['count'] += 1  # Increment the count of the last listed point by 1
            else:
                points.append({'indices': blob['point'], 'count': 1})  # Append a new point to the list

        timeCheckpoint(checkpoint, 'count points')

        return points

    def getYield(self, count=1):
        """Get the yield of n nanowires per point, default = 1 nanowire"""
        lattice_points = self.getLattice().getLatticePoints(*self.getExtremes())
        counts = self.getBlobCountPerPoint()

        total = len(lattice_points)
        good = sum(1 for point in counts if point['count'] == count)

        return good/total

    def getBlobsOfCount(self, count):
        """
        Get all blobs located near lattice points with a given number of blobs in their neighborhood

        :param count: int > 1
            the number of blobs that have to be in a lattice point's neighborhood for those blobs to be in the returned list
        :return: a list on assigned blobs format containing blobs meeting the criterion described above
        """
        if count < 1:
            raise RuntimeError('count must be 1 or larger')

        points = self.getBlobCountPerPoint()  # get a list of blob counts for each lattice point
        # filter the list to only contain points with the desired blob count
        points_with_count = [point['indices'] for point in points if point['count'] == count]

        assigned_blobs = self.getAssignedBlobs()
        # filter the list of blobs to only contain blobs belonging to points listed in the filtered point list
        assigned_blobs_of_count = [a_blob for a_blob in assigned_blobs if a_blob['point'] in points_with_count]

        return assigned_blobs_of_count

    def getExtremes(self, plus_radius=False, flip=False, region=None):
        """
        Get the coordinates limiting a set of blobs

        :param plus_radius: if True, the radius of the largest blob will be added as padding to ensure the entirety of all blobs lie within the bounds
        :param flip: if True, values will be returned in different order
        :param region: the region from within which to get the blobs, if none is given, all blobs will be used
        :return: the bounding x and y values of the smallest region containing all the blobs
        """
        if region == None:
            blobs = self.getBlobs()
        else:
            blobs = self.getSubsetOfBlobs(*region)

        if plus_radius:
            r_max = blobs[:, 2].max()  # Largest radius
        else:
            r_max = 0

        x_min = blobs[:, 1].min() - r_max
        x_max = blobs[:, 1].max() + r_max
        y_min = blobs[:, 0].min() - r_max
        y_max = blobs[:, 0].max() + r_max

        if flip:
            return (x_min, x_max, y_max, y_min)
        else:
            return (x_min, x_max, y_min, y_max)

    def displayTileRegion(self, col_min, col_max, row_min, row_max, plot_lattice='auto', blob_color='red', lattice_color='cyan',
                          connector_color='yellow', figsize=(24, 12), path='', hide_axes=False, feature_scale=1):
        """Display a figure showing a region of the image, with blobs, lattice points and displacement vectors marked

        :param col_min, col_max, row_min, row_max: bounding rows and columns of the region to display
        """
        plot_lattice = plot_lattice.lower()
        if plot_lattice not in ['yes', 'no', 'auto']:
            raise RuntimeError("Invalid plot_lattice value: '" + plot_lattice + "'! Must be 'yes', 'no' or 'auto'.")

        checkpoint = clock()
        total_checkpoint = clock()
        tiles = self.getTileRegion(col_min, col_max, row_min, row_max)

        x_min = self.tilew * col_min
        x_max = self.tilew * (col_max + 1) - 1
        x_len = x_max - x_min
        y_min = self.tileh * row_min
        y_max = self.tileh * (row_max + 1) - 1
        y_len = y_max - y_min
        checkpoint = timeCheckpoint(checkpoint, 'initialize')

        blobs = self.getSubsetOfBlobs(x_min, x_max, y_min, y_max)

        checkpoint = timeCheckpoint(checkpoint, 'filter blobs')

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal', adjustable='box-forced')
        plt.axis((x_min, x_max, y_max, y_min))
        checkpoint = timeCheckpoint(checkpoint, 'setup plot')

        plt.imshow(tiles, extent=[x_min, x_max, y_max, y_min], cmap='gray', interpolation='nearest')
        checkpoint = timeCheckpoint(checkpoint, 'plot tiles')
        f.plotCircles(ax, blobs, fig, dict(color=blob_color, linewidth=1*feature_scale, fill=False))
        checkpoint = timeCheckpoint(checkpoint, 'plot blobs')

        if plot_lattice == 'yes' or (plot_lattice == 'auto' and self.lattice):
            lattice = self.getLattice()
            points = self.lattice.getLatticePoints(x_min, x_max, y_min, y_max)
            flip_points = np.fliplr(points)
            f.plotCircles(ax, flip_points, fig, dict(color=lattice_color, linewidth=5*feature_scale, fill=True))
            checkpoint = timeCheckpoint(checkpoint, 'plot lattice')

            assigned_blobs = self.getAssignedBlobs()
            checkpoint = timeCheckpoint(checkpoint, 'get assigned blobs')

            from matplotlib.collections import LineCollection
            from matplotlib.colors import colorConverter

            connectors = np.zeros((len(assigned_blobs), 2, 2), float)
            for i, a_blob in enumerate(assigned_blobs):
                if len(a_blob['point']) > 0:
                    bx = a_blob['blob'][1]
                    by = a_blob['blob'][0]
                    [px, py] = lattice.getCoordinates(*a_blob['point'])
                    connectors[i, :, :] = [[bx, by], [px, py]]

            colors = colorConverter.to_rgba(connector_color)
            line_segments = LineCollection(connectors, colors=colors, linewidths=1*feature_scale)
            ax.add_collection(line_segments)

        timeCheckpoint(total_checkpoint, 'total time')

        if hide_axes:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.tight_layout()

        if path == '':
            plt.show()
            plt.close()
        else:
            plt.savefig(path)
            print('Saved figure to', path)

    def plotBlobs(self, col, row):
        """Display a figure showing a given tile, and blobs detected for that tile

        :param col: column number of the tile to be displayed
        :param row: row number of the tile to be displayed
        """
        blobs = self.detectBlobs(col, row)
        tile = self.getTile(col, row)

        fig, ax = plt.subplots(figsize=(24, 12))
        ax.set_aspect('equal', adjustable='box-forced')
        plt.axis((0, 1023, 1023, 0))

        plt.imshow(tile, cmap='gray', interpolation='nearest')
        f.plotCircles(ax, blobs, fig, dict(color='red', linewidth=1, fill=False))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.tight_layout()

        plt.show()
        plt.close()

    def plotBlobRegion(self, col_min=0, col_max=None, row_min=0, row_max=None, property='radius', hide_axes=False, colormap='',
                       bg_color='', auto_limits=False):
        """Show a figure plotting all detected blobs from the specified tile region without any background image

        :param col_min, col_max, row_min, row_max: bounding rows and columns of the region to plot, if none are given, the entire dataset is plotted
        :param property: the property determining the coloring of the blobs. Can be either 'diameter', 'distance' or 'angle'
        :auto_limits: if True the plot will not show any of the region beyond the blobs
        """
        if col_max == None:
            col_max = self.cols - 1
        if row_max == None:
            row_max = self.rows - 1

        checkpoint = clock()
        x_min = self.tilew * col_min
        x_max = self.tilew * (col_max + 1) - 1
        y_min = self.tileh * row_min
        y_max = self.tileh * (row_max + 1) - 1

        label = ''
        if property == 'radius' or property == 'diameter':
            property = 'radius'
            label = 'diameter [nm]'
            if colormap == '': colormap = 'jet'
        elif property == 'displacement' or property == 'distance':
            property = 'distance'
            label = 'Displacement from lattice point [nm]'
            if colormap == '': colormap = 'viridis'
        elif property == 'angle':
            label = 'Angle of displacement from lattice point'
            if colormap == '': colormap = 'hsv'
        else:
            raise RuntimeError("Invalid property '" + str(property) + "'")

        checkpoint = timeCheckpoint(checkpoint, 'setup')

        def isInside(a_blob, x_min, x_max, y_min, y_max):
            inside = False
            blob_x = a_blob['blob'][1]
            blob_y = a_blob['blob'][0]
            if x_min <= blob_x <= x_max and y_min <= blob_y <= y_max:
                inside = True

            return inside

        assigned_blobs = self.getAssignedBlobs()
        assigned_blobs = [a_blob for a_blob in assigned_blobs if isInside(a_blob, x_min, x_max, y_min, y_max)]

        blobs = np.zeros((len(assigned_blobs), 4))
        for i, a_blob in enumerate(assigned_blobs):
            blobs[i, 0] = a_blob['blob'][0]
            blobs[i, 1] = a_blob['blob'][1]
            blobs[i, 2] = self.getLattice().getMinLatticeDist() * 0.5
            if property == 'radius':
                blobs[i, 3] = a_blob['blob'][2] * 2 * self.scale
            elif property == 'distance':
                blobs[i, 3] = a_blob['distance'] * self.scale
            elif property == 'angle':
                blobs[i, 3] = a_blob['angle']

        checkpoint = timeCheckpoint(checkpoint, 'getting stuff')

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_aspect('equal', adjustable='box-forced')

        if auto_limits:
            plt.axis(self.getExtremes(plus_radius=True, flip=True))
        else:
            plt.axis((x_min, x_max, y_max, y_min))

        from matplotlib.collections import PatchCollection

        patches = []
        colors = []

        checkpoint = timeCheckpoint(checkpoint, 'figure setup')

        for circle in blobs:
            y, x, r, c = circle
            colors.append(c)
            patch = plt.Circle((x, y), r, linewidth=0, fill=True)
            patches.append(patch)

        checkpoint = timeCheckpoint(checkpoint, 'figure loop')

        p = PatchCollection(patches, match_original=True, cmap=colormap)
        p.set_array(np.array(colors))
        fig.colorbar(p, ax=ax, label=label)
        ax.add_collection(p)

        checkpoint = timeCheckpoint(checkpoint, 'figure end')

        plt.tight_layout()
        if hide_axes:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        if bg_color != '':
            ax.set_axis_bgcolor(bg_color)

        plt.show()
        plt.close()

    def plotBlobCountPerPoint(self, region=None, only_ones=False):
        """A plot coloring each lattice point by the number of blobs near it

        :param region: list of 4 ints
            if given, this denotes the limits of the subregion to plot
            if nothing is given, the whole tileset is used
        :param only_ones: if True, only lattice points containing exactly one blob will be colored
        """
        counts = self.getBlobCountPerPoint(region)
        if only_ones:
            counts = [point for point in counts if point['count']==1]
        lattice = self.getLattice()

        blobs = np.zeros((len(counts), 4))
        for i, point in enumerate(counts):
            coordinates = lattice.getCoordinates(point['indices'][0], point['indices'][1])
            blobs[i, 0] = coordinates[1]
            blobs[i, 1] = coordinates[0]
            blobs[i, 2] = self.getLattice().getMinLatticeDist() * 0.55
            blobs[i, 3] = point['count']

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_aspect('equal', adjustable='box-forced')
        plt.axis(self.getExtremes(plus_radius=True, flip=True, region=region))

        from matplotlib.collections import PatchCollection

        patches = []
        colors = []

        for circle in blobs:
            y, x, r, c = circle
            colors.append(c)
            patch = plt.Circle((x, y), r, linewidth=0, fill=True)
            patches.append(patch)

        p = PatchCollection(patches, match_original=True, cmap='jet')
        p.set_array(np.array(colors))
        fig.colorbar(p, ax=ax, ticks=range(0, 10), label='Blobs per lattice point')
        ax.add_collection(p)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        plt.tight_layout()
        plt.show()
        plt.close()

    def printYields(self, region=None):
        """Print the percentage yields of lattice points containing n nanowires from n from 0 to 10

        :param region: list of 4 ints
            if given, this denotes the limits of the subregion for which to calculate yield
            if nothing is given, the whole tileset is used
        """
        extremes = self.getExtremes(region=region)
        lattice_points = self.getLattice().getLatticePoints(*extremes)
        counts = self.getBlobCountPerPoint(region)

        total = len(lattice_points)
        print(total)
        empty = total

        for n in range(1, 10):
            good = sum(1 for point in counts if point['count'] == n)
            empty -= good
            ratio = good / total * 100

            print('Yield ', n, ': ', ratio, sep='')

        ratio = empty / total * 100
        print('Yield 0:', ratio)

    def plotHistogram(self, property, bins=100, fontsize=16, normalized=True):
        """Plot a histogram of a given property of the detected blobs

        :param property: the property to be plotted. Can be either 'diameter', 'distance' or 'angle'
        :param bins: the number of bins used for the histogram
        :param fontsize: size of the font used in the plot
        :param normalized: when plotting displacement distance, determines weather to normalize histogram bins by the
                           area they represent, to get a plot of radial density
        """
        if property == 'diameter':
            label = 'diameter [nm]'
            blobs = self.getBlobs()
            data = blobs[:, 2] * self.scale * 2
            normalized = False
        elif property == 'distance':
            label = 'displacement from lattice point [nm]'
            assigned_blobs = self.getAssignedBlobs()
            data = [a_blob['distance'] * self.scale for a_blob in assigned_blobs]
        elif property == 'angle':
            label = 'angle'
            assigned_blobs = self.getAssignedBlobs()
            data = [a_blob['angle'] for a_blob in assigned_blobs]
            normalized = False
        else:
            raise ValueError("'" + property + "' is not a valid property")

        fig, ax = plt.subplots(1, 1, figsize=(9, 6), subplot_kw={'adjustable': 'box-forced'})

        plt.grid()
        plt.xlabel(label, fontsize=fontsize)

        if normalized:
            hist, bin_edges = np.histogram(data, bins=bins)
            adjusted_hist = np.zeros(hist.shape[0])

            for i, count in enumerate(hist):
                r0 = bin_edges[i]  # inner radius of the region represented by the bin
                r1 = bin_edges[i+1]  # outer radius of the region represented by the bin
                # divide the counts by the area of the region
                adjusted_hist[i] = float(count) / ( (pi * r1**2) - (pi*r0**2) )

            center = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = (bin_edges[1] - bin_edges[0])
            ax.bar(center, adjusted_hist, align='center', width=width, edgecolor='none', color='#033A87')

            plt.ylabel('density', fontsize=fontsize)
            ax.get_yaxis().set_ticks([])
        else:
            ax.hist(data, bins=bins, histtype='stepfilled', edgecolor='none', color='#033A87')
            plt.ylabel('count', fontsize=fontsize)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        plt.tight_layout()
        plt.show()

    def plotRadialHistogram(self, bins=90, fontsize=20):
        """Plot a radial histogram of the displacement angles of the detected blobs

        :param bins: the number of bins used for the histogram
        :param fontsize: size of the font used in the plot
        """
        assigned_blobs = self.getAssignedBlobs()
        data = [a_blob['angle'] for a_blob in assigned_blobs]

        plt.figure(figsize=[8, 8])
        ax = plt.subplot(111, projection='polar')

        ax.hist(data, bins=bins, histtype='stepfilled', edgecolor='none', color='#033A87')

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        plt.show()

    def scatterPlotDisplacements(self, ylim=0):
        """Show a scatter plot of the displacements from lattice point for all blobs.

        :param ylim: if set, sets the upper limit for the radial axis
        """
        assigned_blobs = self.getAssignedBlobs()
        angles = [blob['angle'] for blob in assigned_blobs]
        displacements = [blob['distance'] * self.scale for blob in assigned_blobs]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

        ax.scatter(angles, displacements, color='mediumblue', alpha=1, s=10, edgecolor='none')
        if ylim:
            ax.set_ylim([0, ylim])
        plt.show()

        ### The three functions below are quite redundant. Sorry about that, they were created in a hurry.

    def scatterPlotDisplacementsFiltered(self, lower_diameter, upper_diameter, ylim=0):
        """Show a scatter plot of the displacements from lattice point for with a diameter within a certain range

        :param lower_diameter, upper_diameter: only blobs with a diameter between these two values will be plotted
        :param ylim: if set, sets the upper limit for the radial axis
        """
        assigned_blobs = self.getAssignedBlobs()
        assigned_blobs = [a_blob for a_blob in assigned_blobs if 185 < a_blob['blob'][2]*2*self.scale < 202]
        angles = [blob['angle'] for blob in assigned_blobs]
        displacements = [blob['distance'] * self.scale for blob in assigned_blobs]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

        ax.scatter(angles, displacements, color='mediumblue', alpha=0.5, s=3, edgecolor='none')
        if ylim:
            ax.set_ylim([0, ylim])
        plt.show()

    def scatterSizeVsDisplacement(self, ylim=0):
        """Show a scatter plot of displacement distance versus diameter for all blobs

        :param ylim: if set, sets the upper limit for the displacement distance axis
        """
        assigned_blobs = self.getAssignedBlobs()
        sizes = [blob['blob'][2] * 2 * self.scale for blob in assigned_blobs]
        displacements = [blob['distance'] * self.scale for blob in assigned_blobs]

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.xlabel('Droplet diameter [nm]')
        plt.ylabel('Displacement from lattice point [nm]')
        plt.grid()

        ax.scatter(sizes, displacements, color='mediumblue', alpha=0.5, s=3, edgecolor='none')
        if ylim:
            ax.set_ylim([0, ylim])
        plt.show()

    def scatterPlotDisplacementsByCount(self, count, ylim=0):
        """Show a scatter plot of the displacements from lattice point for all blobs at lattice points containing exactly the given number of blobs

        :param count: the number of blobs specifying which lattice points to use (see above explanation)
        :param ylim: if set, sets the upper limit for the radial axis
        """
        assigned_blobs = self.getBlobsOfCount(count)
        angles = [blob['angle'] for blob in assigned_blobs]
        displacements = [blob['distance'] * self.scale for blob in assigned_blobs]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

        ax.scatter(angles, displacements, color='mediumblue', alpha=0.5, s=3, edgecolor='none')
        if ylim:
            ax.set_ylim([0, ylim])
        plt.show()

    @staticmethod
    def showMap(map):
        plt.imshow(map, cmap='viridis')
        plt.colorbar()
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.gca().set_axis_bgcolor('black')
        plt.show()
        plt.close()

    def getDensityMap(self, scale_factor, radius):
        from math import floor, ceil, pi

        sf = scale_factor
        r = ceil(radius/scale_factor)
        d = 2 * r  # diameter

        x_min = self.tilew * 0
        x_max = self.tilew * self.cols - 1
        x_len = x_max - x_min
        y_min = self.tileh * 0
        y_max = self.tileh * self.rows - 1
        y_len = y_max - y_min

        blobs = self.getBlobs()

        add_array = f.getCircleOfOnes(r)

        blob_points = [(floor(blob[0] / sf), floor(blob[1] / sf)) for blob in blobs]

        data = np.zeros((ceil(y_len / sf) + d, ceil(x_len / sf) + d))

        for point in blob_points:
            data[point[0]:point[0] + d, point[1]:point[1] + d] += add_array

        px_area = self.scale**2 / 10**6
        c_area = pi * radius**2 * px_area
        data = data / c_area

        return data

    def getRadiusMap(self, scale_factor, radius):
        from math import floor, ceil

        sf = scale_factor
        r = ceil(radius/scale_factor)
        d = 2 * r  # diameter

        x_min = self.tilew * 0
        x_max = self.tilew * self.cols - 1
        x_len = x_max - x_min
        y_min = self.tileh * 0
        y_max = self.tileh * self.rows - 1
        y_len = y_max - y_min

        blobs = self.getBlobs()

        add_array = f.getCircleOfOnes(r)

        blob_points = [(floor(blob[0] / sf), floor(blob[1] / sf), blob[2] * 2 * self.scale) for blob in blobs]

        data = np.zeros((ceil(y_len / sf) + d, ceil(x_len / sf) + d))

        for point in blob_points:
            data[point[0]:point[0] + d, point[1]:point[1] + d] += add_array*point[2]

        return data

    def plotDensity(self, scale_factor, radius):
        """Plot the blob density across the tileset

        :param scale_factor: lower scale factor gives better resolution, but too low results in poor performance
        :param radius: the radius with which to "blur" the plot, experiment with changing it to see what it does
        :return:
        """
        map = self.getDensityMap(scale_factor, radius)

        A = np.argwhere(map)
        (y_start, x_start), (y_stop, x_stop) = A.min(0), A.max(0) + 1
        map = map[y_start:y_stop, x_start:x_stop]

        self.showMap(map)

    def plotRadius(self, scale_factor, radius):
        """Plot the average blob radius of nearby blobs across the tileset

        :param scale_factor: lower scale factor gives better resolution, but too low results in poor performance
        :param radius: the radius with which to "blur" the plot, experiment with changing it to see what it does
        :return:
        """
        r = self.getRadiusMap(scale_factor, radius)
        d = self.getDensityMap(scale_factor, radius)
        map = r / d

        crop_map = np.where(np.isnan(map), 0, 1)
        A = np.argwhere(crop_map)
        (y_start, x_start), (y_stop, x_stop) = A.min(0), A.max(0) + 1
        map = map[y_start:y_stop, x_start:x_stop]

        self.showMap(map)


def createTilesFromImage(path, image_name, tilew=1024, tileh=1024):
    """Cut the given image into tiles of the specified size, and store them in specified path.

    :param path: path to the directory where the tiles will be stored, and where the image to be cut is located
    :param image_name: file name, including extension, of the image to be cut up
    :param tilew: width of the resulting tiles in pixels
    :param tileh: height of the resulting tiles in pixels
    """
    from math import ceil

    image_path = path + '/' + image_name
    image = misc.imread(image_path, flatten=True)

    print(image.shape)
    im_h = image.shape[0]
    im_w = image.shape[1]

    rows = ceil(im_h / tileh)
    cols = ceil(im_w / tilew)

    padded_height = tileh * rows
    padded_width  = tilew * cols

    padding_bottom = padded_height - im_h
    padding_right  = padded_width  - im_w

    padded_image = np.pad(image, ((0, padding_bottom), (0, padding_right)), 'constant')

    for c in range(0, cols):
        col = []

        col_path = path + '/c_' + str(c)
        if not os.path.exists(col_path):
            os.makedirs(col_path)

        for r in range(0, rows):
            tile = padded_image[r*tileh:r*tileh+1024, c*tilew:c*tilew+1024]
            col.append(tile)

            filename = col_path + '/tile_' + str(r) + '.png'
            misc.imsave(filename, tile)
            print('Saved tile ', c, ', ', r, sep='')
