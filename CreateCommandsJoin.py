'''Create commands to clip shapefile
e.g. HMA gives
'''
import DataSetQuery
import json

run='TestRun7-Para'
dataSet = 'ReadyData7'

cmdFile='commands_join.txt'
size = 100000
parentDsName = 'mtngla'
region = 'HMA'
maskDataSet = 'RGIv60'
debrisDataSet = 'SDCv10'


# get min Max boundingbox
environmentName = 'DEV'
query = DataSetQuery.DataSetQuery('http://localhost:9000',environmentName)

# TODO: add region here!!!
bbx = query.getDataSetBoundingBox( parentDsName,'tandemx' )
bbx = json.loads(bbx)
minX=bbx['gridCellMinX']
maxX=bbx['gridCellMaxX']
minY=bbx['gridCellMinY']
maxY=bbx['gridCellMaxY']




with open(cmdFile, 'w') as file:
    for x in range(minX, maxX, size):
        for y in range(minY, maxY, size):
            file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (x, x+size, y, y+size, parentDsName, dataSet, region, maskDataSet, debrisDataSet, run))

