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


class ClipMaskProcess:

    # Himalayas Debris
    # "SDCv10"
    # "/data/puma1/scratch/mtngla/masks/himalayas/debris/13_14_15_rgi60_Himalayas_debris.shp"

    # Himalayas Glacier
    # "RGIv60"
    # "/data/puma1/scratch/mtngla/masks/himalayas/glacier-outlines/13_14_15_rgi60_Himalayas_mask.shp"

    # ----

    # Alaska Debris
    # "SDCv10"
    # "/data/puma1/scratch/mtngla/masks/alaska/debris/PCR-debris-outline.shp"

    # Alaska Glacier
    # "RGIv60"
    # "/data/puma1/scratch/mtngla/masks/alaska/glacier/PCR-glacier-outlines.shp"

    __conf = {
        "maskType": "Glacier",
        "maskPath": "/data/puma1/scratch/mtngla/masks/alaska/glacier/PCR-glacier-outlines.shp",
        "dataSet": "RGIv60",
        "region": "alaska",
        "parentDsName": "mtngla",
        "inputDataSet": "tdx2",
        "malardEnvironmentName": "DEVv2",
        "malardSyncURL": "http://localhost:9000",
        "malardAsyncURL": "ws://localhost:9000",
        "temporaryPublishPath": "/data/puma1/scratch/mtngla/clippedMasks/"
    }


    def __init__(self, minX, maxX, minY, maxY, logFile=None):
        '''

        :param minX:
        :param maxX:
        :param minY:
        :param maxY:
        :param logFile: if logfile is specified logger will write into file instead of the terminal
        '''
        if logFile is None:
            logging.basicConfig(format='%(asctime)s, %(threadName)s %(thread)d: %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S', level=logging.INFO)
        else:
            logging.basicConfig(filename=logFile, filemode='a', format='%(asctime)s, %(threadName)s %(thread)d: %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S', level=logging.INFO)
        sys.excepthook = self.uncaughtErrorHandler

        self.logger = logging.getLogger(__name__)
        self.logger.info('Process started')

        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY
        self.parentDsName = self.config('parentDsName')
        self.maskType = self.config('maskType')
        self.maskPath = self.config('maskPath')
        self.region = self.config('region')
        self.dataSet = self.config('dataSet')


        assert (self.maxX-self.minX) == (self.maxY-self.minY)
        self.size = maxX-minX


    def startProcess(self):
        self.logger.info('Starting gridcell: minX=%s, minY=%s, parentDs=%s, dataSet=%s, maskType=%s, maskPath=%s,', self.minX, self.minY, self.parentDsName, self.dataSet, self.maskType, self.maskPath)
        self.query_sync = DataSetQuery.DataSetQuery(self.config('malardSyncURL'), self.config('malardEnvironmentName'))
        # get projection
        self.projection = json.loads(self.query_sync.getProjection(self.parentDsName, self.region))['proj4']

        self.clipMask()
        if self.clippedMask.empty:
            self.logger.info('Clipped mask is empty / does not contain any shapes. Process is ended.')
        else:
            self.calculateArea()
            self.publishMask()
        # shutdown
        self.logger.info("Finished process for: minX=%s, minY=%s, size=%s", self.minX, self.minY, self.size)
        self.logger.info('------------------------------------------------------------------')
        logging.shutdown()

        # clear variables
        sys.modules[__name__].__dict__.clear()

    def clipMask(self):
        '''Clip a shapefile to extent
        For better performance first spatial join with intersect is computed, and then overlapping polygons are clipped with geopandas overlay
        New polygon area is calculated. Note that it assumes that crs is projected for area calculation
        '''

        self.logger.info('Starting to clip polygon minX=%s, maxX=%s, minY=%s, maxY=%s', self.minX, self.maxX, self.minY, self.maxY)
        self.logger.info('Opening mask file: %s', self.maskPath)
        # create extent geodataframe

        # open mask shapefile
        masksShp = gp.read_file(self.maskPath)
        self.logger.info('Creating clip polygon from extent...')

        extent = gp.GeoSeries([Polygon([(self.minX,self.minY), (self.minX,self.maxY), (self.maxX,self.maxY), (self.maxX,self.minY), (self.minX,self.minY)])])
        dfExtent = gp.GeoDataFrame(geometry=extent)
        dfExtent.crs = self.projection

        self.logger.info('Intersecting Shapefile with extent...')
        # intersect with extent
        maskIntersect = gp.sjoin(masksShp, dfExtent, how='inner', op='intersects')

        #drop columns except geometry and Area
        maskIntersect.drop(maskIntersect.columns.difference(['geometry', 'Area']), 1, inplace=True)
        #rename Area column to area_old
        maskIntersect.rename(columns={'Area':'area_old'}, inplace=True)

        self.logger.info('Clip overlapping polygons...')
        self.clippedMask = gp.overlay(maskIntersect, dfExtent, how='intersection')

        del masksShp
        del maskIntersect



    def calculateArea(self):
        '''Calculate new polygon area'''
        self.logger.info('Calculating new polygon areas...')
        # compute new Area in km2
        self.clippedMask['area'] = self.clippedMask['geometry'].area/ 10**6

        self.logger.info('Sum of poly area is: %s km2', '{0:.3g}'.format(self.clippedMask['area'].sum()))


    def publishMask(self):
        self.logger.info('Saving shapefile file...')
        fileName = "mask_%s_x%s_y%s_s%s.gpkg" % (self.maskType, self.minX, self.minY, self.size)
        fullPath = os.path.join(self.config("temporaryPublishPath"), fileName)

        self.clippedMask.to_file(driver='GPKG', filename=fullPath)
        del self.clippedMask

        self.logger.info('File temporarily saved under: %s', fullPath)

        # publish
        self.logger.info('Publishing file with params: Parent Dataset: %s, Dataset %s,  Region: %s, Mask type: %s', self.parentDsName, self.dataSet, self.region, self.maskType)
        query = DataSetQuery.DataSetQuery('http://localhost:9000')
        msg = query.publishMask(self.config("temporaryPublishPath"), fileName, self.parentDsName, self.dataSet, self.maskType, self.region, self.minX, self.minY, self.size)
        self.logger.info(msg)

        #delete file
        os.remove(fullPath)
        self.logger.info('Temporary file removed')



    @staticmethod
    def config(name):
        return ClipMaskProcess.__conf[name]
    def uncaughtErrorHandler(self, type, value, tb):
        self.logger.error("Uncaught exception", exc_info=(type, value, tb))



if __name__ ==  '__main__':
    # himalayas
    clip = ClipMaskProcess(100000, 200000, 0, 100000)
    clip.startProcess()
