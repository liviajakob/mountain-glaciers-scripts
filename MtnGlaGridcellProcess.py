import logging
import sys
import DataSetQuery
import AsyncDataSetQuery
import datetime
import json
import os
import calendar
from DataSets import *


class MtnGlaGridcellProcess:

#"referenceDem":"/data/puma1/scratch/DEMs/srtm_test.tif"
#"referenceDem":"/data/puma1/scratch/mtngla/dems/HMA_TDX_Masked_SRTM_Merged_coreg_aea_clip.tif"

    __conf = {
        "runName": "TestRun8",
        "outputDataSet": "Ready8",
        "parentDsName": "mtngla",
        "region":"himalayas",
        "maskDataSet": "RGIv60",
        "withinDataSets": ["SDCv10", "/data/puma1/scratch/mtngla/dems/Tdx_SRTM_SurfaceSplit.tiff"],
        "withinDataSetTypes": ["Debris", "Tdx"],
        "referenceDem": "/data/puma1/scratch/mtngla/dems/HMA_TDX_Masked_SRTM_Merged_coreg_aea_clip.tif",
        "inputDataSet": "tdx2",
        "malardEnvironmentName": "DEVv2",
        "malardSyncURL": "http://localhost:9000",
        "malardAsyncURL": "ws://localhost:9000",
        "filters" : [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.8}, \
                     {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMad','op':'lt','threshold':10}, \
                     {'column':'demDiff','op':'gt','threshold':-100}, {'column':'demDiffMad','op':'gt','threshold':-10}]

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
        self.outputDataSet = self.config('outputDataSet')
        self.inputDataSet = self.config('inputDataSet')
        self.region = self.config('region')
        self.maskDataSet = self.config('maskDataSet')
        self.withinDataSets = self.config('withinDataSets')
        self.withinDataSetTypes = self.config('withinDataSetTypes')
        self.runName = self.config('runName')

        assert (self.maxX-self.minX) == (self.maxY-self.minY)
        self.size = maxX-minX





    def startProcess(self):
        self.logger.info('Starting gridcell: minX=%s, minY=%s, parentDs=%s, inputDataSet=%s, outputDataSet=%s, runName=%s,', self.minX, self.minY, self.parentDsName, self.inputDataSet, self.outputDataSet, self.runName)
        self.defineVariables()
        if os.path.exists(self.maskDataSetFile):
            self.data = self.filter(self.inputDataSet)
            # @TODO SRTM filter and join?

            # To Geodata
            self.logger.info('Converting to Geodataset...')
            self.data = self.data.asGeoDataSet()
            self.applyMasks()

            # Calculate elevation difference
            if self.data.hasData():
                raster = RasterDataSet(self.config('referenceDem'))
                assert (self.maxX-self.minX)==(self.maxY-self.minY)
                buffer = (self.maxX-self.minX)*0.1
                self.data.calculateElevationDifference(raster, buffer=buffer)

                self.addStatistics()
                self.publish()
                self.logger.info("STATISTICS: %s", self.data.getStats())
        else:
            self.logger.info("No valid mask (fp=%s) found for %s, %s, %s, minX=%s, minY=%s, size=%s", self.maskDataSetFile, self.maskDataSet, 'Glacier', self.region, self.minX, self.minY, self.size)

        # shutdown
        self.logger.info("Finished process for: minX=%s, minY=%s, size=%s", self.minX, self.minY, self.size)
        self.logger.info('------------------------------------------------------------------')
        logging.shutdown()

        # clear variables
        sys.modules[__name__].__dict__.clear()


    def filter(self, datasetName):
        filters = self.config('filters')
        self.logger.info("Filtering dataset=%s with criteria %s" % (datasetName, filters))
        result = self.query_async.executeQuery(self.parentDsName, datasetName, self.region, self.minX, self.maxX, self.minY, self.maxY, self.minT, self.maxT,[],filters)
        self.logger.info("Result message: %s, %s" % (result.status, result.message))
        fp = result.resultFileName
        data = PointDataSet(fp, self.projection)
        # release cache of file
        self.query_async.releaseCache(fp)
        data.addStatistic('%s_filtered' % datasetName, data.length())
        self.logger.info("Filter %s result count [%d]" % (datasetName, data.length()))
        return data

    def applyMasks(self):
        # Mask
        self.data.applyMask(self.maskDataSetFile,'Glacier')

        # Add column if point is inside masks
        for idx, i in enumerate(self.withinDataSetFiles):
            self.data.withinMask(i, self.withinDataSetTypes[idx])

    def addStatistics(self):
        self.logger.info('Adding additional statistics')
        # number of srtm and number of tandemX
        self.data.addStatistic('result_total', self.data.length())
        #stats['result_srtm'] = float(data.loc[data.dataset == 'SRTM', 'dataset'].count())
        #stats['result_tandemx'] = float(data.loc[data.dataset == 'TandemX', 'dataset'].count())
        self.data.addStatistic('result_avgX', self.data.mean('x'))
        self.data.addStatistic('result_avgY', self.data.mean('y'))
        self.data.addStatistic('result_offsetX', self.data.getStats()['result_avgX']-(self.minX+(self.size/2)))
        self.data.addStatistic('result_offsetY', self.data.getStats()['result_avgY']-(self.minX+(self.size/2)))

        # counts per year
        # @TODO do this in glacier years
        years=[x for x in range(self.minT.year, self.maxT.year+1)]
        for year in years:
            start = datetime.datetime(year,1,1,0,0)
            end = datetime.datetime(year+1,1,1,0,0)
            start = calendar.timegm(start.utctimetuple())
            end = calendar.timegm(end.utctimetuple())
            # count
            keyCount = "result_count_%s" % (year)
            peryear = float(self.data.data.loc[(self.data.data.time >= start) & (self.data.data.time <end)].shape[0])
            self.data.addStatistic(keyCount, peryear)
            # elevation difference
            elevDiff = "result_refDifference_%s" % (year)
            if peryear > 0.0:
                self.data.addStatistic(elevDiff, float(self.data.data.loc[(self.data.data.time >= start) & (self.data.data.time <end), 'refDifference'].mean()))
            else:
                self.data.addStatistic(elevDiff, 0.0)
            # offset
            #keyOffsetX = "result_offsetX_%s" % (year)
            #keyOffsetY = "result_offsetY_%s" % (year)
            #if stats[keyCount] > 0:
            #    stats[keyOffsetX] = float(data.loc[(data.time >= start) & (data.time <end), 'x'].mean()-(minX+(size/2)))
            #    stats[keyOffsetY] = float(data.loc[(data.time >= start) & (data.time <end), 'y'].mean()-(minY+(size/2)))
            #else:
            #    stats[keyOffsetX] = 0.0
            #    stats[keyOffsetY] = 0.0

    def publish(self, outEnvironment='/data/puma1/scratch/mtngla/ReadyData'):
        # get data as normal pandas dataframe without the geo ref
        data = self.data.getData(geo=False)

        outPath = os.path.join(outEnvironment, "ReadyData_%s_x%s_y%s.nc" % (self.minX, self.minY, self.size))
        xarr = data.to_xarray()
        xarr.to_netcdf(outPath)

        # publish
        self.logger.info('Publish new dataset...')
        result=self.query_async.publishGridCellPoints(self.parentDsName, self.outputDataSet, self.region, self.minX, self.minY, self.data.min('time'), self.size, outPath, self.projection)
        self.logger.info('Response: %s' %  result.json)
        # delete temporary file
        os.remove(outPath)

        # publish stats
        self.logger.info('Publish gridcell statistics...')
        response = self.query_sync.publishGridCellStats(self.parentDsName, self.runName, self.minX, self.minY, self.size, self.data.getStats())
        self.logger.info('Response: %s' % response)



    def defineVariables(self):
        self.query_sync = DataSetQuery.DataSetQuery(self.config('malardSyncURL'), self.config('malardEnvironmentName'))
        self.query_async = AsyncDataSetQuery.AsyncDataSetQuery(self.config('malardAsyncURL'), self.config('malardEnvironmentName'), False)
        # minT and maxT
        bbx = self.query_sync.getDataSetBoundingBox(self.parentDsName, self.config('inputDataSet'), self.region )
        bbx = json.loads(bbx)
        self.minT = datetime.datetime.utcfromtimestamp(bbx['minTime'])
        self.maxT = datetime.datetime.utcfromtimestamp(bbx['maxTime'])
        # get projection
        self.projection = json.loads(self.query_sync.getProjection(self.parentDsName, self.region))['proj4']

        # masks
        mGla = self.query_sync.getGridCellMask(self.parentDsName, self.maskDataSet, 'Glacier', self.region, self.minX, self.minY, self.size)
        self.maskDataSetFile = json.loads(mGla)['fileName']

        self.withinDataSetFiles = []
        for i, el in enumerate(self.withinDataSets):
            # @TODO not just Debris
            if os.path.exists(el):
                self.withinDataSetFiles.append(el)
            else:
                mask = self.query_sync.getGridCellMask(self.parentDsName, el, self.withinDataSetTypes[i], self.region, self.minX, self.minY, self.size)
                self.withinDataSetFiles.append(json.loads(mask)['fileName'])
        print(self.withinDataSetFiles)

    @staticmethod
    def config(name):
        return MtnGlaGridcellProcess.__conf[name]
    def uncaughtErrorHandler(self, type, value, tb):
        self.logger.error("Uncaught exception", exc_info=(type, value, tb))

if __name__ ==  '__main__':
    mtngla = MtnGlaGridcellProcess(400000, 500000, 0, 100000)
    #mtngla = MtnGlaGridcellProcess(500000, 600000, 0, 100000)
    #mtngla = MtnGlaGridcellProcess(700000, 800000, 0, 100000)
    #mtngla = MtnGlaGridcellProcess(500000, 600000, 100000, 200000)
    #mtngla = MtnGlaGridcellProcess(500000, 600000, -100000, 0)

    # error in this one for dem diff
    #mtngla = MtnGlaGridcellProcess(200000, 300000, -100000, 0)

    # mask file not found
    #mtngla = MtnGlaGridcellProcess(-200000, -100000, -200000, -100000)
    mtngla.startProcess()
