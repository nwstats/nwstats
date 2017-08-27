import numpy as np
import matplotlib.pyplot as plt

from math import ceil, log10

import functions as f
import detect
from field import Field

class FieldArray:
    field_ext = '.fld'

    def __init__(self, nfa, nfb, Na, Nb, path, scale, ext='.tif'):
        self.nfa = nfa      # Number of fields in the a dimension
        self.nfb = nfb      # Number of fields in the b dimension
        self.Na = Na        # Number of lattice points in a direction (not related to a dimension mentioned above, maybe find better way to explain it?)
        self.Nb = Nb        # Number of lattice points in b direction
        self.path = path    # Path to the folder where images of the fields are stored
        self.scale = scale  # Size of a pixel in nm
        self.ext = ext      # File extension of the images

        self.num_fields = nfa * nfb  # Total number of fields
        self.fields = [Field(Na, Nb, path, str(num+1).zfill(3), scale, ext) for num in range(0, self.num_fields)]  # List of field objects for all the fields

    def prepFields(self, kernel_size, prep_path):
        """Preprocess all images, using median filtering of the given kernel size

        :param kernel_size: the kernel size to use for median filtering
        :param prep_path: the path in which to save the preprocessed images
        :return: a FieldArray object using the preprocessed images
        """
        for field in self.fields:
            field.prepImage(kernel_size, prep_path)

        return FieldArray(self.nfa, self.nfb, self.Na, self.Nb, prep_path, self.scale, self.ext)

    def detectBlobs(self, methods=(detect.droplets,)):
        """Detect blobs on all fields in the array, and store the data for later use, overwriting any existing data.

        :param methods: detection methods to be used, if more than one is used, the result is the combined results
        """
        for field in self.fields:
            field.detectBlobs(methods)

    def ensureBlobs(self, methods=(detect.droplets,)):
        """Ensure that blob data exists for all fields, by detecting blobs for fields where no stored data is found.

        :param methods: detection methods to be used, if more than one is used, the result is the combined results
        """
        for field in self.fields:
            field.getBlobs(methods)

    def clearBlobs(self):
        """Delete all stored data about detected blobs."""
        for field in self.fields:
            field.clearBlobs()

    def makeLattices(self):
        """Define lattices for all fields, and store the data for later use, overwriting any existing data."""
        for field in self.fields:
            field.makeLattice()

    def ensureLattices(self):
        """Ensure that lattice data exists for all fields, by defining lattices for fields where no stored data is found."""
        for field in self.fields:
            field.getLattice()

    def readjustLattices(self):
        """Readjust the stored lattices for all fields to fit with the currently stored detected blobs."""
        for field in self.fields:
            success = field.readjustLattice()
            if not success:
                raise RuntimeError('No lattice to adjust for field ' + field.name)

    def clearLattices(self):
        """Delete all stored data about lattices."""
        for field in self.fields:
            field.clearLattice()

    def generateBlobsByPoint(self):
        """Assign blobs to lattice points for all fields, and store the data for later use, overwriting any existing data."""
        for field in self.fields:
            field.generateBlobsByPoint()

    def ensureBlobsByPoint(self):
        """Ensure that blob assignment data exists for all fields, by assigning blobs to lattice points for fields where no stored data is found."""
        for field in self.fields:
            field.getBlobsByPoint()

    def clearBlobsByPoint(self):
        """Delete all stored data about blob assignment to lattice points."""
        for field in self.fields:
            field.clearBlobsByPoint()

    def getMeanDiameters(self):
        """Return a list of the mean blob diameter of each field."""
        return [field.getMeanDiameter() for field in self.fields]

    def getMedianDiameters(self):
        """Return a list of the median blob diameter of each field."""
        return [field.getMedianDiameter() for field in self.fields]

    def getYields(self, n):
        """Return a list of the percentage yields of n nanowires per hole for each field."""
        return [field.getYield(n, percentage=True) for field in self.fields]

    def listFieldsByYield(self, n):
        """Print a sorted list of percentage yields of n nanowires per hole for each field."""
        yields = self.getYields(n)
        field_list = list(enumerate(yields, start=1))
        field_list = sorted(field_list, key=lambda x: (x[1]), reverse=True)
        for field in field_list:
            print(str(field[0]).rjust(ceil(log10(len(field_list)))), ': ', round(field[1], 1), sep='')

    def plotBlobs(self):
        """Create and store images of each field with detected blobs marked."""
        for field in self.fields:
            field.plotBlobs(save=True)

    def plotLattices(self):
        """Create and store images of each field with lattice points marked."""
        for field in self.fields:
            field.plotLattice(save=True)

    def plotLatticesWithBlobs(self):
        """Create and store images of each field with detected blobs and lattice points marked."""
        for field in self.fields:
            field.plotLatticeAndBlobs(save=True)

    def plotAvgBlobs(self, kwargs):
        """Display a plot of the average number of blobs per lattice point for each field."""
        average_blobs = [field.getBlobCount() / field.number_of_points for field in self.fields]
        plt = f.arrayPlot(average_blobs, **kwargs)
        plt.show()

    def plotMeanDiameters(self, kwargs):
        """Display a plot of the mean blob diameter for each field."""
        mean_diameters = self.getMeanDiameters()
        plt = f.arrayPlot(mean_diameters, **kwargs)
        plt.show()

    def plotMedianDiameters(self, kwargs):
        """Display a plot of the median blob diameter for each field."""
        median_diameters = self.getMedianDiameters()
        plt = f.arrayPlot(median_diameters, **kwargs)
        plt.show()

    def plotMeanDisplacements(self, kwargs):
        """Display a plot of the mean blob displacement from lattice for each field."""
        mean_displacements = [np.mean(field.getDisplacementMagnitudes()) for field in self.fields]
        for i, d in enumerate(mean_displacements):
            print(i+1, d)
        plt = f.arrayPlot(mean_displacements, **kwargs)
        plt.show()

    def plotMedianDisplacements(self, kwargs):
        """Display a plot of the median blob displacement from lattice for each field."""
        median_displacements = [np.median(field.getDisplacementMagnitudes()) for field in self.fields]
        plt = f.arrayPlot(median_displacements, **kwargs)
        plt.show()

    def plotDisplacementStd(self, kwargs):
        """Display a plot of the standard deviation of the blob displacement from lattice for each field."""
        std_displacements = [np.std(field.getDisplacementMagnitudes()) for field in self.fields]
        plt = f.arrayPlot(std_displacements, **kwargs)
        plt.show()

    def plotDisplacementMdev(self, kwargs):
        """Display a plot of the median deviation from the median of the blob displacement from lattice for each field."""
        mdevs = []
        for field in self.fields:
            data = field.getDisplacementMagnitudes()
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            mdevs.append(mdev)

        plt = f.arrayPlot(mdevs, **kwargs)
        plt.show()

    def plotYield(self, n):
        """Display a plot of the yields of n nanowires per hole for each field."""
        yields = self.getYields(n)
        title = str(n) + ' blobs'
        plt = f.arrayPlot(yields, title=title, percentages=True)
        plt.show()

    def plotDiameterHistograms(self):
        """Plot histograms of the droplet diameter distributions for each field

        WARNING: Not properly generalized. Will look strange for anything else than 8x8 fields. Axis scales are also hard coded.
        """
        from math import ceil
        diametersPerField = [field.getDiameters() for field in self.fields]
        fig, axes = plt.subplots(8, 8, figsize=(21.5, 10), subplot_kw={'adjustable': 'box-forced'})
        for n, diameters in enumerate(diametersPerField):
            row = 7 - (n) % 8
            col = ceil((n + 1) / 8) - 1

            x_max = 300
            ax = axes[row, col]
            ax.set_title(n + 1)
            ax.set_xlim((0, x_max))
            ax.set_ylim((0, 70))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.hist(diameters, bins=40, histtype='stepfilled', color='#033A87', edgecolor='none', range=[0, x_max])

        plt.tight_layout()
        plt.show()

    def plotAmountOverLimit(self, limit):
        """Plot the fraction of detected blobs with a diameter larger than the given limit, for each field."""
        values = []
        for field in self.fields:
            diameters = field.getDiameters()
            amount_over_limit = len([d for d in diameters if d > limit])
            ratio = amount_over_limit / len(diameters) * 100
            values.append(ratio)

        f.arrayPlot(values, percentages=True)
        plt.show()

    def plotDisplacementHistograms(self):
        """Plot histograms of the droplet displacements from the lattice for each field

        WARNING: Not properly generalized. Will look strange for anything else than 8x8 fields. Axis scales are also hard coded.
        """
        from math import ceil
        displacements_per_field = [field.getDisplacementMagnitudes() for field in self.fields]

        fig, axes = plt.subplots(8, 8, figsize=(21.5, 10), subplot_kw={'adjustable': 'box-forced'})
        for n, displacements in enumerate(displacements_per_field):

            displacements_per_field[n] = displacements

            row = 7 - (n) % 8
            col = ceil((n + 1) / 8) - 1

            ax = axes[row, col]
            ax.set_title(n + 1)
            ax.set_xlim((0, 30))
            ax.set_ylim((0, 70))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.hist(displacements, bins=50, histtype='stepfilled', edgecolor='none', range=(0, 30))

        plt.tight_layout()
        plt.show()
        plt.close()

    def plotDisplacementAngleHistograms(self):
        """Plot histograms of the droplet displacements angles for each field

        WARNING: Not properly generalized. Will look strange for anything else than 8x8 fields. Axis scales are also hard coded.
        """
        from math import ceil
        from math import pi
        displacement_angles_per_field = [field.getDisplacementAngles() for field in self.fields]
        fig, axes = plt.subplots(8, 8, figsize=(21.5, 10), subplot_kw={'adjustable': 'box-forced'})
        for n, angles in enumerate(displacement_angles_per_field):

            row = 7 - (n) % 8
            col = ceil((n + 1) / 8) - 1

            ax = axes[row, col]
            ax.set_title(n + 1)
            ax.set_xlim((-pi, pi))
            ax.set_ylim((0, 25))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.hist(angles, bins=50, histtype='stepfilled', edgecolor='none')

        plt.tight_layout()
        plt.show()

    def plotDisplacementScatterPlots(self):
        """Plot scatter plots of the droplet displacements from the lattice for each field

        WARNING: Not properly generalized. Will look strange for anything else than 8x8 fields. Axis scales are also hard coded.
        """
        from math import ceil, pi
        fig, axes = plt.subplots(8, 8, figsize=(12, 12), subplot_kw=dict(projection='polar'))
        for n, field in enumerate(self.fields):

            row = 7 - (n) % 8
            col = ceil((n + 1) / 8) - 1
            ax = axes[row, col]

            r = field.getDisplacementMagnitudes()
            angles = field.getDisplacementAngles()

            ax.scatter(angles, r, color='mediumblue', s=3, alpha=0.5, edgecolor='none')
            ax.grid(color='#EEEEEE', linestyle='-', linewidth=1)
            ax.set_axisbelow(True)

            ax.set_ylim((0, 500))
            ax.get_xaxis().set_ticklabels([])
            ax.get_xaxis().set_ticks([0, pi/2, pi, -pi/2])
            ax.get_yaxis().set_ticklabels([])

        plt.tight_layout()
        plt.show()

    def plotSingleScatterPlot(self, n):
        """Plot a scatter plot of the droplet displacements from the lattice for a single field, along with a histogram of the radial displacement distribution.

        WARNING: Axis scales are hard coded.
        """
        from math import pi

        ax1 = plt.subplot(121, projection='polar')

        field = self.fields[n-1]

        r = field.getDisplacementMagnitudes()
        angles = field.getDisplacementAngles()

        ax1.scatter(angles, r, color='mediumblue', s=10, alpha=0.5, edgecolor='none')
        ax1.grid(color='#EEEEEE', linestyle='-', linewidth=1)
        ax1.set_axisbelow(True)

        ax1.set_ylim((0, 500))
        ax1.get_xaxis().set_ticklabels([])
        ax1.get_xaxis().set_ticks([0, pi / 2, pi, -pi / 2])
        ax1.text(2.5, 34*self.scale, str(n).rjust(2))

        ax2 = plt.subplot(122)
        ax2.hist(r, bins=70, range=[0, 500], histtype='stepfilled', color='limegreen', edgecolor='none')

        plt.show()

    def plotSingleDisplacementHistogram(self, n):
        """Plot a histogram of the droplet displacements from the lattice for a single field

        WARNING: Hard coded numbers.
        """
        displacements = self.fields[n-1].getDisplacementMagnitudes()
        plt.hist(displacements, bins=26, range=(0, 26), histtype='stepfilled', edgecolor='none', color='#033A87')
        plt.show()

    def plotOverallDiameterHistogram(self):
        """Plot a histogram of blob diameter for all blobs across all fields."""
        diameters = [field.getDiameters() for field in self.fields]
        data = [item for sublist in diameters for item in sublist]
        plt.hist(data, bins=150, histtype='stepfilled', edgecolor='none', color='#033A87')
        plt.show()

    def plotFancyDiameterHistogram(self):
        """Plot histograms showing how the diameter distribution depends on the number of blobs at a lattice point.

        All droplets for all fields are used.
        Droples are sorted by how many droplets are found around the same lattice point.
        Diameter histograms are plotted separately for each category (one for all single blobs, one for all blobs sharing a point with 1 other, and so on.)
        These are plotted in the same plot.
        This is the worst docstring ever, sorry. :\

        WARNING: Hard coded numbers.
        """
        bbp = [field.getBlobsByPoint() for field in self.fields]
        bbp = [item for sublist in bbp for item in sublist]  # Flatten list

        points_single = [point for point in bbp if len(point) == 1]
        diameters_single = [blob[2] * 2 * self.scale for point in points_single for blob in point]

        max_diameter = max(diameters_single)
        min_diameter = min(diameters_single)
        bins = 50

        max_num = 3
        diameters_sorted = []
        for n in range(1, max_num+1):
            points = [point for point in bbp if len(point) == n]
            diameters = [blob[2] * 2 * self.scale for point in points for blob in point]

            diameters_sorted.append(diameters)
            max_diameter = max(max(diameters), max_diameter)
            min_diameter = min(min(diameters), min_diameter)

        points = [point for point in bbp if len(point) > max_num]
        diameters = [blob[2] * 2 * self.scale for point in points for blob in point]

        diameters_sorted.append(diameters)

        colors2 = ['dodgerblue', 'red', 'orange', 'limegreen', 'purple', 'magenta']
        for n, data in enumerate(diameters_sorted):
            if n == 0:
                label = '1 nanowire'
            else:
                label = str(n+1) + ' nanowires'
            plt.hist(data, bins=bins, zorder=0.5+n, label=label, histtype='step', edgecolor=colors2[n%6], alpha=1, range=[min_diameter, max_diameter], linewidth=3)

        plt.xlabel('diameter [nm]')
        plt.ylabel('count')

        plt.legend()
        plt.show()

    def plotOverallDisplacementHistogram(self):
        """Plot a histogram of blob displacement from lattice point for all blobs across all fields."""
        displacements_per_field = [field.getDisplacementMagnitudes() for field in self.fields]
        data = [item for sublist in displacements_per_field for item in sublist]
        plt.hist(data, bins=100, histtype='stepfilled', edgecolor='none', color='#033A87')
        plt.show()

    def plotOverallDisplacementAngleHistogram(self):
        """Plot a histogram of blob displacement angle for all blobs across all fields."""
        displacements_per_field = [field.getDisplacementAngles() for field in self.fields]
        data = [item for sublist in displacements_per_field for item in sublist]
        plt.hist(data, bins=20, histtype='stepfilled', edgecolor='none', color='#033A87')
        plt.show()
