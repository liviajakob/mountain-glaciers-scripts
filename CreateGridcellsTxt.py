'''Create commands to clip shapefile
e.g. HMA gives
'''
import DataSetQuery
import json

cmdFile='hma-gridcells.txt'
parentDsName = 'mtngla'
region = 'himalayas'
size=100000


# get min Max boundingbox
environmentName = 'DEVv2'
query = DataSetQuery.DataSetQuery('http://localhost:9000',environmentName)

bbx = query.getDataSetBoundingBox(parentDsName, 'tdx', region)
bbx = json.loads(bbx)
minX=bbx['gridCellMinX']
maxX=bbx['gridCellMaxX']
minY=bbx['gridCellMinY']
maxY=bbx['gridCellMaxY']


with open(cmdFile, 'w') as file:
    for x in range(minX, maxX, size):
        for y in range(minY, maxY, size):
            file.write('%s,%s,%s,%s\n' % (x, x+size, y, y+size))


