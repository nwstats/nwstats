import numpy as np
import matplotlib.pyplot as plt
from tileset import Tileset, createTilesFromImage
from scipy import misc
import functions as f
import detect

# Scales:
# NIL 1: 9.767
# NIL 2 guess: 3.272
# scale = 9.767
scale = 3.272

path = 'data/images/tiled/originals'
prep_path = 'data/images/tiled/prepped'

# path = 'data/test/cookies'

# tiles = Tileset(prep_path, 28, 18, 1024, 1024, scale, detect.tiled, '.tif')

# tiles.plotHistogram('diameter', bins=500)
# tiles.detectAllBlobs()
# tiles.makeLattice(debug=True, max_blobs=100)

# tiles.assignBlobs()
# tiles.plotBlobRegion()

# tiles.plotRadialHistogram(bins=30)

# createTilesFromImage(path, 'kjeks.png')

# tiles.detectAllBlobs()

# tiles.detectBlobs(0, 0)
# tiles.plotBlobs(0, 0)

# tiles.makeLattice()

# tiles.detectAllBlobs()
# tiles.plotOnlyBlobs()

# region = None
# region = [8000, 25000, 0, 16000]
# tiles.plotBlobCountPerPoint(region)
# tiles.printYields(region)

# tiles.makeLattice()
#
# tiles.assignBlobs()
# tiles.plotBlobRegion(0, 27, 0, 17, property='distance', bg_color='black', hide_axes=True, auto_limits=True)
# tiles.scatterPlotDisplacementsByCount(3)
# tiles.scatterPlotDisplacementsFiltered()
# print(tiles.getBlobs().shape[0])
# tiles.scatterSizeVsDisplacement()

# tiles.plotBlobs(5, 5)

# tiles.getLattice()
# tiles.displayTileRegion(0, 0, 3, 3, feature_scale=1)

# region = tiles.getTileRegion(1, 1, 1, 1)
# plt.imshow(region, cmap='gray')
# plt.show()

# tiles.plotHistogram('distance', bins=10)
# tiles.plotRadialHistogram('angle', bins=90)

# tiles.scatterPlotDisplacements()

# print(tiles.getRDF()[4])
# blobs = tiles.getSubsetOfBlobs(0, 4*1024, 0, 4*1024)
# distances = tiles.getDistances(blobs)
# # tiles.detectAllBlobs()
#
# print(np.median(distances))
# print('cake')
# print('cake')
#
# fig, ax = plt.subplots(1, 1, figsize=(21.5, 10), subplot_kw={'adjustable': 'box-forced'})
# plt.plot((101.5, 101.5), (0, 500))
#
# ax.hist(distances, histtype='stepfilled', edgecolor='none', bins=200)
#
# plt.tight_layout()
# plt.show()

# tiles.makeLattice(debug=True)
# tiles.detectAllBlobs()
# tiles.makeLattice()
# tiles.plotBlobRegion(0, 21, 0, 9, property='distance')

# counts = tiles.getBlobCountPerPoint()

# tiles.detectAllBlobs()
# tiles.assignBlobs(tiles.getBlobs(), tiles.getLattice())
# tiles.deleteAssignedBlobs()
# tiles.makeLattice()
# tiles.plotBlobRegion(0, 27, 0, 17, property='angle', bg_color='black', hide_axes=True, auto_limits=True)
# tiles.plotBlobCountPerPoint()
# print('Yield 1:', tiles.getNaiveYield())
# print('Yield 2:', tiles.getYield())
# print(tiles.getExtremes())
# tiles.plotHistogram('distance', bins=100)
# tiles.plotRadialHistogram('angle', bins=45)

# tiles.plotHistogram('diameter', bins=500)

# print(tiles.getBlobs().shape[0])

# tiles.prepTiles(prep_path)

# region = tiles.getTileRegion(0, 28, 0, 18)
# # region = tiles.getPaddedTile(0, 0)
#
# misc.imsave('big.png', region)
# plt.imshow(region, cmap='gray')
# plt.show()

# tiles.plotBlobs(2, 3)

# tiles.plotDensity(10, 100)
# tiles.plotRadius(10, 100)

# tiles.detectAllBlobs()

# tiles.makeLattice()
# tiles.getLattice()
# tiles.displayTileRegion(1, 6, 7, 10, hide_axes=True, figsize=(112, 80), feature_scale=3, path='background_3.png', lattice_color='#c2d5e8')

# tiles.plotYield(0, 28, 0, 18)
# tiles.plotBlobRegion(0, 10, 0, 6, property='distance')
# tiles.plotBlobRegion(0, 27, 0, 17, property='radius', hide_axes=True)
# tiles.plotRadius(10, 1000)

# tiles.plotRDF()
# tiles.makeLattice(step=3)
# blobs = tiles.getBlobs()
# a_blobs = tiles.getAssignedBlobs()
# print(blobs.shape[0])
# print(len(a_blobs))

# tiles.makeLattice(step=10, max_blobs=50, final_blobs=4000)
# tiles.plotBlobRegion(0, 27, 0, 17, 'distance')

# tiles.plotHistogram('distance')

path = 'data/images/random'
prep_path = 'data/images/random/prepped'
#
# tiles = Tileset(path, 19, 4, 1024, 1024, detection_method=detect.random, ext='.png')
# tiles = Tileset(prep_path, 19, 4, 1024, 1024, 13.02, detection_method=detect.random, ext='.png')
# tiles.prepTiles(prep_path, 5)
# tiles.detectAllBlobs()
# tiles.deleteLattice()
# tiles.displayTileRegion(0, 5, 0, 2)
# tiles.plotBlobs(3, 1)
# tiles.plotBlobRegion(0, 3, 0, 18)

# tiles.plotHistogram('distance', bins=500)
# tiles.plotDensity(2, 100)
tiles.plotRadius(2, 100)

# print(tiles.getBlobs().shape[0])
