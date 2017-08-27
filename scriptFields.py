from fieldArray import FieldArray

path = 'data/images/test_fields'
prep_path = 'data/images/test_fields/prep'

fields = FieldArray(8, 8, 15, 18, path, 4.8, '.png')
fields = fields.prepFields(kernel_size=3, prep_path=prep_path)

fields.makeLattices()

fields.plotLatticesWithBlobs()
fields.plotYield(1)
fields.plotYield(2)
fields.plotMeanDiameters({'title': 'Mean droplet diameter'})
fields.plotDiameterHistograms()
fields.plotFancyDiameterHistogram()
fields.plotDisplacementScatterPlots()
