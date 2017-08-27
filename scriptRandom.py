from tileset import Tileset, createTilesFromImage
import detect

path = 'data/test_random'
image_filename = 'stitch.png'

createTilesFromImage(path, image_filename)

tiles = Tileset(path, 3, 2, 1024, 1024, 13.02, detection_method=detect.random, ext='.png')

tiles.plotHistogram('diameter')
tiles.plotDensity(1, 200)
tiles.plotRadius(1, 200)
