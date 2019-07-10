'''Create commands to clip shapefile'''

cmdFile='commands_clip.txt'
#gridcell size
size = 100000
parentDsName = 'mtngla'
#dataSet = 'RGIv60'
dataSet = 'SDCv10'
#maskType = 'Glacier'
maskType = 'Debris'
region = 'HMA'
#shpFile = '/data/puma1/scratch/mtngla/masks/himalayas/glacier-outlines/13_14_15_rgi60_Himalayas_mask.shp'
shpFile = '/data/puma1/scratch/mtngla/masks/himalayas/debris/13_14_15_rgi60_Himalayas_debris.shp'

# get min Max boundingbox
environmentName = 'DEV'
query = DataSetQuery.DataSetQuery('http://localhost:9000',environmentName)

# TODO: add region here!!!
bbx = query.getDataSetBoundingBox( parentDsName,'tandemX' )
bbx = json.loads(bbx)
minX=bbx['gridCellMinX']
maxX=bbx['gridCellMaxX']
minY=bbx['gridCellMinY']
maxY=bbx['gridCellMaxY']



with open(cmdFile, 'w') as file:
    for x in range(minX, maxX, size):
        for y in range(minY, maxY, size):
            file.write('%s,%s,%s,%s,%s,%s,%s,%s, %s\n' % (x, x+size, y, y+size, parentDsName, dataSet, maskType, region, shpFile))

