'''Create commands to clip shapefile
e.g. HMA gives
'''
import DataSetQuery
import json

# HIMALAYAS
#cmdFile='hma-gridcells.txt'
#parentDsName = 'mtngla'
#region = 'himalayas'
#size=100000
#dataset= 'tdx2'

# ALASKA
cmdFile='alaska-gridcells.txt'
parentDsName = 'mtngla'
region = 'alaska'
size=100000
dataset = 'ADwithTDX'


# get min Max boundingbox
environmentName = 'DEVv2'
query = DataSetQuery.DataSetQuery('http://localhost:9000',environmentName)

bbx = query.getDataSetBoundingBox(parentDsName, dataset, region)
bbx = json.loads(bbx)
minX=bbx['gridCellMinX']
maxX=bbx['gridCellMaxX']
minY=bbx['gridCellMinY']
maxY=bbx['gridCellMaxY']


with open(cmdFile, 'w') as file:
    for x in range(minX, maxX, size):
        for y in range(minY, maxY, size):
            file.write('%s,%s,%s,%s\n' % (x, x+size, y, y+size))


