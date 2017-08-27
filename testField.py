from fieldArray import FieldArray, Field
import detect
from lattice import Lattice
import pickle
import numpy as np

path = 'data/full_size_cropped/prep_2'
figure_path = path + '/figures'

fields = FieldArray(8, 8, 15, 18, path, 4.8, '.png')

fields.plotMedianDisplacements(kwargs={'real_axes': True, 'colorbar_label': 'median displacement [nm]'})

# fields.plotDiameterHistograms()
# fields.plotAmountOverLimit(215)
# fields.plotFancyDiameterHistogram()
# fields.fields[5].plotHistogram('diameter', fontsize=14)

# fields.plotMedianDisplacements(kwargs={'real_axes': True, 'colorbar_label': '[nm]'})
# fields.plotMedianDisplacements(kwargs={'real_axes': True, 'title': 'Median droplet displacement [nm]'})
# fields.plotMedianDisplacements(kwargs={'real_axes': True, 'colorbar_label': '[nm]', 'title': 'Median droplet displacement'})

# fields.plotSomeDisplacementScatterPlots()

# fields.plotAvgBlobs({'real_axes': True, 'colorbar_label': 'average droplets per hole'})
# fields.plotMeanDisplacements({})

n = 64
# fields.fields[n-1].detectBlobs(methods=(detect.dropletsNew,))
# fields.fields[n-1].plotBlobs()
# fields.fields[4].plotOnlyLattice()
# fields.readjustLattices()
# for n in range(0, 64):
#     fields.fields[n].prepImage()


print('woop')
print('woop')