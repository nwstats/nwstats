from tileset import Tileset
import detect

path = 'data/images/tiled/originals'
prep_path = 'data/images/tiled/prepped'

tiles = Tileset(path, cols=3, rows=2, tilew=1024, tileh=1024, scale= 9.767, detection_method=detect.tiled)
tiles = tiles.prepTiles(prep_path, 5)

tiles.displayTileRegion(0, 2, 0, 1)
tiles.plotDensity(1, 400)

print('Yield:', round(tiles.getYield()*100, 2), '%')

tiles.plotHistogram('diameter', bins=100)
tiles.plotBlobRegion(property='radius', colormap='viridis')

tiles.plotHistogram('distance', bins=50)
tiles.plotBlobRegion(property='distance')

tiles.plotRadialHistogram(bins=30, fontsize = 10)
tiles.plotBlobRegion(property='angle')

tiles.scatterPlotDisplacements()
tiles.scatterSizeVsDisplacement()

tiles.plotBlobCountPerPoint()