'''Example parameters: 400000 500000 0 100000 mtngla testdataset HMA RGIv60 SDCv10 DebugRun

Example with Empty result set
500000 600000 -100000 0 mtngla testdataset himalayas RGIv60 SDCv10 DebugRun

200000 300000 800000 900000 mtngla testdataset himalayas RGIv60 SDCv10 DebugRun

'''
import DataSetQuery
import datetime
import pandas as pd
import MalardHelpers
import geopandas as gp
import json
import logging
import sys
from shapely.geometry import Point, Polygon
import shapely.speedups
import os
import argparse
import gc
import calendar
import numpy as np
from osgeo import gdal
from scipy.interpolate import griddata
import AsyncDataSetQuery


shapely.speedups.enable()

# logger
#@TODO: uncomment and comment the basic config this
#logging.basicConfig(filename='masking.log', filemode='a', format='%(asctime)s, %(threadName)s %(thread)d: %(name)s %(levelname)s %(message)s',
#                    datefmt='%H:%M:%S', level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
stats={}

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




def filter(query_async,query_sync,parentDsName, region, minX,maxX,minY,maxY,minT,maxT, **kwargs):
    '''FILTER with the usual criteria

    :param query:
    :param minX:
    :param maxX:
    :param minY:
    :param maxY:
    :param minT:
    :param maxT:
    :param kwargs:
    :return: dataframes of filtered tandemX and srtm
    '''
    logger.info("Starting to filter... ")

    # filter for usual criteria
    filters = [{'column':'power','op':'gt','threshold':10000},{'column':'coh','op':'gt','threshold':0.8}, \
               {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMad','op':'lt','threshold':10}, \
               {'column':'demDiff','op':'gt','threshold':-100}, {'column':'demDiffMad','op':'gt','threshold':-10}]
    #filters = [{'column':'power','op':'gt','threshold':1000},{'column':'coh','op':'gt','threshold':0.5}, \
    #           {'column':'demDiff','op':'lt','threshold':100}, {'column':'demDiffMad','op':'lt','threshold':10}, \
    #           {'column':'demDiff','op':'gt','threshold':-100}, {'column':'demDiffMad','op':'gt','threshold':-10}]

    logger.info("Filtering with criteria %s" % filters)

    #TandemX
    logger.info("Filter tandemX... ")

    tandemX = query_async.executeQuery( parentDsName,'tdx', region,minX,maxX,minY,maxY,minT,maxT,[],filters)

    if os.path.isfile(tandemX.resultFileName):
        dfTandemX = MalardHelpers.getDataFrameFromNetCDF(tandemX.resultFileName)
        query_sync.releaseCache(tandemX.resultFileName)
    else:
        logger.info('TandemX file=%s for minX=%s maxX=%s minY=%s maxY=%s empty. Empty dataframe is created', tandemX, minX, maxX, minY, maxY)
        dfTandemX = pd.DataFrame()

    stats['tandemX_filtered'] = float(len(dfTandemX.index))

    logger.info("TandemX result count [%d]" % len(dfTandemX.index))

    #Srtm
    logger.info("Filter SRTM... ")
    srtm = query_async.executeQuery(parentDsName,'srtm',region,minX,maxX,minY,maxY,minT,maxT,[],filters)
    if os.path.isfile(srtm.resultFileName):
        dfSrtm = MalardHelpers.getDataFrameFromNetCDF(srtm.resultFileName)
        query_sync.releaseCache(srtm.resultFileName)
    else:
        logger.info('SRTM file=%s for minX=%s maxX=%s minY=%s maxY=%s empty. Empty dataframe is created', srtm, minX, maxX, minY, maxY)
        dfSrtm = pd.DataFrame()

    stats['srtm_filtered'] = float(len(dfSrtm.index))

    logger.info("SRTM result count [%d]" % len(dfSrtm.index))

    return dfTandemX, dfSrtm



def notInJoin(tandemX, srtm):
    '''identify SRTM that are not in TandemX

    :param dfTandemX:
    :param dfSrtm:
    :return: SRTM points that are not in TandemX
    '''
    logger.info("Starting not in join... ")
    if not(tandemX.empty or srtm.empty):
        #indicator adds a column named 'i' to output dataframe, then filtered to right_only and finally dropped the column
        dfNotInJoin = tandemX[['sampleNb', 'swathFileId', 'wf_number']].merge(srtm, indicator='i', how='outer', on=['sampleNb', 'swathFileId', 'wf_number']).query('i == "right_only"').drop('i', 1)
    elif tandemX.empty:
        dfNotInJoin = srtm
    elif srtm.empty:
        dfNotInJoin = pd.DataFrame()

    stats['srtm_not_in'] = float(len(dfNotInJoin.index))
    logger.info("SRTMNotIn result count [%d]" % len(dfNotInJoin.index))

    return dfNotInJoin


def joinDatasets(tandemX, srtmNotIn):
    logger.info("Starting to join tandemX and additional srtm datasets... ")
    tandemX['dataset'] = 'TandemX'
    srtmNotIn['dataset'] = 'SRTM'
    # check that we don't try to join empty dataframes
    if not (tandemX.empty or srtmNotIn.empty):
        joined = pd.concat([tandemX, srtmNotIn], ignore_index=True, sort=False)
    elif srtmNotIn.empty:
        joined = tandemX
    elif tandemX.empty:
        joined = srtmNotIn
    stats['joined_data'] = float(len(joined.index))

    logger.info("Joined data count [%d]" % len(joined.index))
    return joined

def toGeoData(data, crs):
    if not data.empty:
        logger.info("Convert dataframe to Geodataframe... ")
        # Convert points to geometries
        geometry = [Point(xy) for xy in zip(data.x, data.y)]
        geoData = gp.GeoDataFrame(data, crs=crs, geometry=geometry)
        logger.info("Conversion successful!")
        return geoData
    else:
        # return empty dataframe
        return gp.GeoDataFrame()

def applyGlacierMask(geoData, maskPath):
    '''Exclude points which are not within glaciers

    :param geoData:
    :param mask:
    :return:
    '''
    if not geoData.empty:
        logger.info("Read mask file... ")
        mask = gp.read_file(maskPath)
        stats['glacierMaskArea'] = float(mask['area'].sum())
        # drop all columns except geometry
        mask.drop(mask.columns.difference(['geometry']), 1, inplace=True)

        #logger.info("Reproject points to same projection as mask... ")
        #geoD = geoData.to_crs(mask.crs)

        logger.info("Apply glacier mask... ")
        maskedGla=gp.sjoin(geoData, mask, how='inner', op='within')
        maskedGla.drop(columns=['index_right'], axis=1, inplace=True)
        stats['afterGlacierMask'] = float(len(maskedGla.index))

        logger.info("After Glacier mask: count [%d]" % len(maskedGla.index))
        return maskedGla
    else:
        return geoData


def applyDebrisMask(data, maskPath):
    '''Exclude points within Debris

    :param data:
    :param mask:
    :return:
    '''
    if maskPath == 'NoMask':
        logger.info("No mask found for %s, %s, %s, %s, minX=%s, minY=%s, size=%s", parentDsName, debrisDataSet, 'Debris', region,minX,minY,size)
        return data
    elif data.empty:
        return data
    else:
        logger.info("Read debris file... ")
        mask = gp.read_file(maskPath)
        stats['debrisMaskArea'] = float(mask['area'].sum())
        # drop all columns except geometry
        mask.drop(mask.columns.difference(['geometry']), 1, inplace=True)

        #logger.info("Reproject points to same projection as mask... ")
        #geoD = data.to_crs(mask.crs)

        logger.info("Apply debris mask... ")
        leftJoin = gp.sjoin(data, mask, how='left', op='within')
        # exclude values with index_right not null
        logger.info('Left joined')

        # assign 1 to points that are inside debris mask and 0 to points that are not inside
        leftJoin['debris'] = (leftJoin['index_right'] >= 0).astype(int)
        leftJoin.drop('index_right', axis=1, inplace=True)

        countDebris = leftJoin.loc[(leftJoin.debris == 1), 'x'].count()

        stats['pointsOnDebris'] = float(countDebris)
        logger.info("Points in debris mask: count [%d]" % countDebris)

        return leftJoin


def calculateElevationDifference(data, minX,maxX,minY,maxY, crs, fpDem):
    logger.info("Starting to calculate elevation difference from reference DEM... ")
    out_tif = "/vsimem/tile_%s_%s_%s_%s.tif" % (minX, maxX, minY, maxY)
    out_xyz = '/data/puma1/scratch/DEMs/tile_%s_%s_%s_%s.xyz' % (minX, maxX, minY, maxY) # file will be deleted after
    ds = gdal.Open(fpDem)
    side = (maxX-minX)*0.1 # size of one side, add a 10% buffer on each side

    logger.info("Clipping raster file... ")
    tile = gdal.Translate(out_tif, ds, projWin = [minX-side, maxY+side, maxX+side, minY-side])
    logger.info("Translating raster to XYZ... ")
    gdal.Translate(out_xyz, tile, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])
    del tile

    logger.info("Reading in XYZ file... ")
    xy,values = [], []
    with open(out_xyz,'r') as f:
        f.readline() # skip first row
        for l in f:
            row = l.split()
            xy.append([float(row[0]) , float(row[1])])
            values.append(float(row[2]))
    # remove xyz  file
    os.remove(out_xyz)

    # get point coordinates as numpy
    coords = np.asarray([data['x'], data['y']])
    coords = np.transpose(coords)

    # interpolate
    logger.info("Interpolating... ")
    result=griddata(xy, values, coords, method='cubic')

    # calculate difference
    data['refElevation'] = result
    data['refDifference'] = data['elev']-data['refElevation']
    logger.info("Finished calculating difference, mean difference =  %s... " % data['refDifference'].mean())
    return data



def publish(masked, query, parentDsName, dataSet, region, minX, minY, size, projection):
    logger.info('Converting geodataframe to dataframe...')
    masked.drop(columns=['geometry'], axis=1, inplace=True) #drop geometry column
    if 'index_right' in masked:
        masked.drop(columns=['index_right'], axis=1, inplace=True)
    df = pd.DataFrame(masked)
    #save as netcdf
    logger.info('Saving as netCDF...')
    sourcePath = '/data/puma1/scratch/mtngla/ReadyData'
    fileName = "ReadyData_%s_x%s_y%s.nc" % (minX, minY, size)
    fullPath = os.path.join(sourcePath, fileName)
    xarr = df.to_xarray()
    xarr.to_netcdf(fullPath)

    # publish netCDF
    logger.info('Publish new dataset...')
    logger.info('parentDsName=%s, dataSet=%s, minX=%s, minY=%s, size=%s, filePath=%s, projection=%s',parentDsName, dataSet, minX, minY, size, fullPath, projection)
    msg=query.publishGridCellPoints(parentDsName, dataSet, region, minX, minY, size, fullPath, projection)
    logger.info(msg)

    # delete temporary file
    os.remove(fullPath)
    logger.info('Temporary file removed')

def addStatistics(data, minX, maxX, size, minT, maxT):
    logger.info('Adding additional statistics')
    # number of srtm and number of tandemX
    stats['result_total'] = float(len(data.index))
    stats['result_srtm'] = float(data.loc[data.dataset == 'SRTM', 'dataset'].count())
    stats['result_tandemx'] = float(data.loc[data.dataset == 'TandemX', 'dataset'].count())
    stats['result_avgX'] = data['x'].mean()
    stats['result_avgY'] = data['y'].mean()
    stats['result_offsetX'] = stats['result_avgX']-(minX+(size/2))
    stats['result_offsetY'] = stats['result_avgY']-(minY+(size/2))

    # counts per year
    years=[x for x in range(minT.year, maxT.year+1)]
    for year in years:
        start = datetime.datetime(year,1,1,0,0)
        end = datetime.datetime(year+1,1,1,0,0)
        start = calendar.timegm(start.utctimetuple())
        end = calendar.timegm(end.utctimetuple())
        # count
        keyCount = "result_count_%s" % (year)
        stats[keyCount] = float(data.loc[(data.time >= start) & (data.time <end), 'elev'].count())
        # elevation difference
        elevDiff = "result_refDifference_%s" % (year)
        stats[elevDiff] = float(data.loc[(data.time >= start) & (data.time <end), 'refDifference'].mean())
        # offset
        keyOffsetX = "result_offsetX_%s" % (year)
        keyOffsetY = "result_offsetY_%s" % (year)
        if stats[keyCount] > 0:
            stats[keyOffsetX] = float(data.loc[(data.time >= start) & (data.time <end), 'x'].mean()-(minX+(size/2)))
            stats[keyOffsetY] = float(data.loc[(data.time >= start) & (data.time <end), 'y'].mean()-(minY+(size/2)))
        else:
            stats[keyOffsetX] = 0.0
            stats[keyOffsetY] = 0.0





def parseArguments():
    parser = argparse.ArgumentParser(description='Clip and publish mask of an extent.')
    parser.add_argument('minX', type=int, help='minX of extent')
    parser.add_argument('maxX', type=int, help='maxX of extent')
    parser.add_argument('minY', type=int, help='minY of extent')
    parser.add_argument('maxY', type=int, help='maxY of extent')
    parser.add_argument('parentDsName', type=str, help='parent dataset name')
    parser.add_argument('dataSet', type=str, help='name of the dataset to be published')
    parser.add_argument('region', type=str, help='data region, e.g. HM for Himalayas')
    parser.add_argument('maskDataSet', type=str, help='name of the mask dataset')
    parser.add_argument('debrisDataSet', type=str, help='name of the debris dataset')
    parser.add_argument('runName', type=str, help='Name of the run')

    return parser.parse_args()


if __name__=="__main__":
    args=parseArguments()
    setupLogger()
    environmentName = 'DEVv2'
    #Sets the URL of the Malard ServiceGateway.
    query = DataSetQuery.DataSetQuery('http://localhost:9000',environmentName)
    query_async = AsyncDataSetQuery.AsyncDataSetQuery( 'ws://localhost:9000',environmentName, False)

    ######### VARIABLES ##############################################################



    #Setup the bounding box
    minX=args.minX
    maxX=args.maxX
    minY=args.minY
    maxY=args.maxY
    size=maxX-minX

    parentDsName = args.parentDsName
    dataSet = args.dataSet
    region = args.region
    maskDataSet = args.maskDataSet
    debrisDataSet = args.debrisDataSet
    runName = args.runName

    #minT and maxT
    bbx = query.getDataSetBoundingBox( parentDsName, 'tdx', region )
    bbx = json.loads(bbx)

    minT = datetime.datetime.utcfromtimestamp(bbx['minTime'])
    maxT = datetime.datetime.utcfromtimestamp(bbx['maxTime'])

    projection = json.loads(query.getProjection(parentDsName, region))['proj4']
    referenceDem = "/data/puma1/scratch/DEMs/srtm_test.tif"

    #################################################################################

    # STEP 0: Get mask and check if any mask for that gridcell, skip process if not...
    # @TODO change this
    mGla = query.getGridCellMask(parentDsName, maskDataSet, 'Glacier', 'HMA',minX,minY,size)
    glacierMask = json.loads(mGla)['fileName']

    logger.info('Starting gridcell: minX=%s, minY=%s, parentDs=%s, dataSet=%s, runName=%s,', minX, minY, parentDsName, dataSet, runName)

    if glacierMask == 'NoMask':
        logger.info("No mask found for %s, %s, %s, %s, minX=%s, minY=%s, size=%s", parentDsName, maskDataSet, 'Glacier', region,minX,minY,size)
    else:
        # STEP 1: Filter
        tandemX, srtm = filter(query_async,query,parentDsName,region,minX,maxX,minY,maxY,minT,maxT)

        # STEP 2: NotInJoin
        srtmNotIn = notInJoin(tandemX, srtm)

        # STEP 3: Join Datasets
        # tests if tandemX or srtmNotIN empty, returns joined data
        joinedData = joinDatasets(tandemX, srtmNotIn)

        #free up some memory
        del tandemX
        del srtm
        del srtmNotIn
        gc.collect()

        # STEP 4: Convert to Geodataframe
        geoDf = toGeoData(joinedData, projection)

        # STEP 5: Mask Glaciers
        masked = applyGlacierMask(geoDf, glacierMask)

        #free up some memory: delete local variables
        gc.collect()

        # STEP 6: Mask Debris
        # checked that masked not empty
        # @TODO: change region
        mDeb = query.getGridCellMask(parentDsName, debrisDataSet, 'Debris', 'HMA',minX,minY,size)
        debrisMask = json.loads(mDeb)['fileName']
        masked = applyDebrisMask(masked, debrisMask)
        #delete local variables
        gc.collect()

        # checked that not empty
        if not masked.empty:

            # STEP 7: Add elevation difference from reference DEM as column
            data = calculateElevationDifference(masked, minX,maxX,minY,maxY, projection, referenceDem)

            # STEP 8: Publish
            publish(data, query, parentDsName, dataSet, region, minX, minY, size, projection)

            # STEP 9: Publish metadata
            # first add some more stats
            addStatistics(masked, minX, maxX, size, minT, maxT)

            response = query.publishGridCellStats(parentDsName, runName, minX, minY, size, stats)
            logger.info(response)

            logger.info("METADATA: %s", stats)
    logger.info("Finished %s, %s, %s, minX=%s, minY=%s, size=%s", parentDsName, dataSet, region,minX,minY,size)
    logger.info('------------------------------------------------------------------')
    logging.shutdown()

    # clear variables
    sys.modules[__name__].__dict__.clear()

