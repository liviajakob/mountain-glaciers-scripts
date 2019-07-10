'''Script to prepare glacier masks and debris (cut into bounding box extent)

'''
from shapely.geometry import Point, Polygon
import shapely.speedups
import time
import DataSetQuery
import geopandas as gp
import logging
import sys
import os
import argparse
import json
shapely.speedups.enable()

# logger
#@TODO change the logger before running it
logging.basicConfig(filename='clipping.log', filemode='a', format='%(asctime)s, %(threadName)s %(thread)d: %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# himalayas crs
# @TODO change this
#crs = {'proj': 'aea', 'lat_1': 25, 'lat_2': 47, 'lat_0': 36, 'lon_0': 85, 'x_0': 0, 'y_0': 0, 'datum': 'WGS84', 'units': 'm', 'no_defs': True}

def setupLogger():
    #Set up Logging

    # create a file handler
    #handler = logging.FileHandler('clipping.log')
    #handler.setLevel(logging.INFO)

    # create a logging format
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #handler.setFormatter(formatter)

    # add the handlers to the logger
    #logger.addHandler(handler)

    # Install exception handler
    sys.excepthook = uncaughtErrorHandler
    logger.info('Logger started...')

def uncaughtErrorHandler(type, value, tb):
    logger.error("Uncaught exception", exc_info=(type, value, tb))

def clipMask(crs, shpFile, minX, maxX, minY, maxY, **kwargs):
    '''Clip a shapefile to extent
    For better performance first spatial join with intersect is computed, and then overlapping polygons are clipped with geopandas overlay
    New polygon area is calculated. Note that it assumes that crs is projected for area calculation

    :param minX:
    :param maxX
    :param minY:
    :param maxY:
    :param shpFile:
    :return:
    '''


    logger.info('Starting to clip polygon minX=%s, maxX=%s, minY=%s, maxY=%s', minX, maxX, minY, maxY)
    logger.info('Opening mask file: %s', shpFile)
    # create extent geodataframe

    start=time.time()

    # open mask shapefile
    masksShp = gp.read_file(shpFile)

    logger.info('Creating clip polygon from extent...')

    extent = gp.GeoSeries([Polygon([(minX,minY), (minX,maxY), (maxX,maxY), (maxX,minY), (minX,minY)])])
    dfExtent = gp.GeoDataFrame(geometry=extent)
    dfExtent.crs = crs

    logger.info('Intersecting Shapefile with extent...')
    # intersect with extent
    maskIntersect = gp.sjoin(masksShp, dfExtent, how='inner', op='intersects')

    #drop columns except geometry and Area
    maskIntersect.drop(maskIntersect.columns.difference(['geometry', 'Area']), 1, inplace=True)
    #rename Area column to area_old
    maskIntersect.rename(columns={'Area':'area_old'}, inplace=True)

    logger.info('Clip overlapping polygons...')
    maskClipped = gp.overlay(maskIntersect, dfExtent, how='intersection')

    logger.info('Total time used for clipping %s seconds', '{0:.3g}'.format(time.time()-start))

    return maskClipped



def calculateArea(maskClipped):
    '''Calculate new polygon area'''
    logger.info('Calculating new polygon areas...')
    # compute new Area in km2
    maskClipped['area'] = maskClipped['geometry'].area/ 10**6

    logger.info('Sum of poly area is: %s km2', '{0:.3g}'.format(maskClipped['area'].sum()))


def publishMask(mask, minX, maxX, minY, maxY, parentDsName, dataSet, maskType, region, **kwargs):
    size=abs(maxX-minX)

    logger.info('Saving shapefile file...')
    # save mask temporarily
    sourcePath = '/data/puma1/scratch/mtngla/clippedMasks/'
    fileName = "mask_%s_x%s_y%s_s%s.gpkg" % (maskType, minX, minY, size)
    fullPath = os.path.join(sourcePath, fileName)
    mask.to_file(driver='GPKG', filename=fullPath)
    #close file
    #@TODO close file


    logger.info('File temporarily saved under: %s', fullPath)

    # publish
    logger.info('Publishing file with params: Parent Dataset: %s, Dataset %s,  Region: %s, Mask type: %s', parentDsName, dataSet, region, maskType)
    query = DataSetQuery.DataSetQuery('http://localhost:9000')
    msg = query.publishMask(sourcePath, fileName, parentDsName, dataSet, maskType, region, minX, minY, size)
    logger.info(msg)

    #delete file
    os.remove(fullPath)
    logger.info('Temporary file removed')



def parseArguments():
    parser = argparse.ArgumentParser(description='Clip and publish mask of an extent.')
    parser.add_argument('minX', type=int, help='minX of extent')
    parser.add_argument('maxX', type=int, help='maxX of extent')
    parser.add_argument('minY', type=int, help='minY of extent')
    parser.add_argument('maxY', type=int, help='maxY of extent')
    parser.add_argument('parentDsName', type=str, help='parent dataset name')
    parser.add_argument('dataSet', type=str, help='dataset name')
    parser.add_argument('maskType', type=str, help='mask type, e.g. Glacier or Debris')
    parser.add_argument('region', type=str, help='data region, e.g. HM for Himalayas')
    parser.add_argument('shpFile', type=str, help='Path to shapeFile')

    return parser.parse_args()


if __name__=="__main__":
    args=parseArguments()
    setupLogger()

    #crs
    #query = DataSetQuery.DataSetQuery('http://localhost:9000')
    #crs = json.loads(query.getProjection('HMA'))['proj4']
    #logger.info(crs)
    #crs= '+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=85 +x_0=0 +y_0=0 +ddatum=WGS84 +units=m +no_defs'
    #crs = {'proj': 'aea', 'lat_1': 25, 'lat_2': 47, 'lat_0': 36, 'lon_0': 85, 'x_0': 0, 'y_0': 0, 'ellps': 'WGS84', 'datum': 'WGS84', 'units': 'm', 'no_defs': True}
    crs = {'proj': 'aea', 'lat_1': 25, 'lat_2': 47, 'lat_0': 36, 'lon_0': 85, 'x_0': 0, 'y_0': 0, 'datum': 'WGS84', 'units': 'm', 'no_defs': True}
    #clip
    maskClipped = clipMask(crs,**vars(args))

    # check if clipped mask has content
    if maskClipped.empty:
        logger.info('Clipped mask is empty / does not contain any shapes. Process is ended.')
    else:
        calculateArea(maskClipped)
        publishMask(maskClipped, **vars(args))

    logger.info('-------------------------------------------------------------')
    logging.shutdown()


