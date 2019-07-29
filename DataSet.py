import MalardHelpers
import logging
import pandas as pd
import os
import geopandas as gp
from shapely.geometry import Point, Polygon
import shapely.speedups

class DataSet:
    def __init__(self, data, dataSetName, projection, stats={}):
        '''

        :param data: can be a dataframe or a filename to a netcdf
        :param dataSetName:
        :param projection:
        '''
        self.logger = logging.getLogger(__name__)
        self.dataSetName = dataSetName
        self.projection = projection
        self.stats = stats
        if isinstance(data, pd.DataFrame) or isinstance(data, gp.GeoDataFrame):
            self.data = data
        else:
            self._readData(data)



    def _readData(self, filename):
        if os.path.isfile(filename):
            self.data = MalardHelpers.getDataFrameFromNetCDF(filename)
        else:
            self.logger.info('%s file=%s for minX=%s maxX=%s minY=%s maxY=%s empty. Empty dataframe is created', self.dataSetName, filename, minX, maxX, minY, maxY)
            self.df = pd.DataFrame()


    def asGeoDataSet(self):
        if hasattr(self, 'geoDataSet'):
            return self.geoDataSet
        else:
            geoDf = self._df_to_geodf()
            geoDs = GeoDataSet(geoDf, self.dataSetName, self.projection, stats=self.stats)
            self.geoDataSet = geoDs
            return self.geoDataSet


    def _df_to_geodf(self):
        '''Helper function to convert a pandas dataframe to a geopandas geodataframe'''
        if not self.data.empty:
            shapely.speedups.enable()
            self.logger.info("Convert DataFrame to GeoDataFrame... ")
            # Convert points to geometries
            geometry = [Point(xy) for xy in zip(self.data.x, self.data.y)]
            geoDf = gp.GeoDataFrame(self.data, crs=self.projection, geometry=geometry)
            self.logger.info("Conversion to GeoDataFrame successful!")
            return geoDf
        else:
            return gp.GeoDataFrame()

    def hasData(self):
        '''Returns True if dataframe not empty, returns false if empty'''
        return not self.data.empty




class GeoDataSet(DataSet):

    def applyMask(self, maskPath, maskType):
        '''Filters out data points that aren't inside the mask

        :param maskPath: path to mask file
        :param maskType: mask type, e.g. "Glacier" or "Debris"
        :return:
        '''
        if self.hasData():
            self.logger.info("Read %s mask file... " % maskType)
            if os.path.exists(maskPath):
                mask = gp.read_file(maskPath)

                self.stats['%sMaskArea'%maskType] = float(mask['area'].sum())
                # drop all columns except geometry
                mask.drop(mask.columns.difference(['geometry']), 1, inplace=True)

                self.logger.info("Apply %s filter mask... (deletes points which aren't within masks) " % maskType)
                maskedGla=gp.sjoin(self.data, mask, how='inner', op='within')
                maskedGla.drop(columns=['index_right'], axis=1, inplace=True)

                self.stats['after%sMask' % maskType] = float(len(maskedGla.index))
                self.logger.info("After applying %s mask: point count [%d]" % (maskType,len(maskedGla.index)))
                self.data = maskedGla
            else:
                self.logger.error('Error: File path=%s of type=s% is invalid' % (maskPath, maskType))


    def withinMask(self, maskPath, maskType):
        '''Adds a column to data decribing if point is within mask or not
        0 means not within mask
        1 means within mask

        :param maskPath:
        :param maskType:
        :return:
        '''

        if self.hasData():
            self.logger.info("Read %s mask file... " % maskType)
            if os.path.exists(maskPath):
                mask = gp.read_file(maskPath)

                self.stats['%sMaskArea'%maskType] = float(mask['area'].sum())
                # drop all columns except geometry
                mask.drop(mask.columns.difference(['geometry']), 1, inplace=True)

                self.logger.info("Apply %s mask (adds a column to points describing if they within mask)... " % maskType)
                masked = gp.sjoin(self.data, mask, how='left', op='within')

                # assign 1 to points that are inside debris mask and 0 to points that are not inside
                masked['within%s' % maskType] = (masked['index_right'] >= 0).astype(int)
                masked.drop('index_right', axis=1, inplace=True)

                # @ TODO: not dependent on 'x' column
                count = masked.loc[(masked['within%s' % maskType] == 1), 'x'].count()

                self.stats['pointsWithin%sMask' % maskType] = float(count)
                self.logger.info("Points within %s mask: count [%d]" % (maskType, count))
                self.data = masked
            else:
                self.logger.error('Error: File path=%s of type=s% is invalid' % (maskPath, maskType))





if __name__ ==  '__main__':
    logging.basicConfig(level=logging.INFO)
    fp = '/data/puma1/scratch/v2/malard/export/mtngla_tdx_1556569735.nc'
    dataSet = 'tdx'
    projection = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=85 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    glacierMask = '/data/puma1/scratch/malard/mask/mtngla/static/RGIv60/Glacier/HMA/cell_x400000_y0_s100000/mask_Glacier_x400000_y0_s100000.gpkg'
    debrisMask = '/data/puma1/scratch/malard/mask/mtngla/static/SDCv10/Debris/HMA/cell_x400000_y0_s100000/mask_Debris_x400000_y0_s100000.gpkg'

    ds = DataSet(fp, dataSet, projection)
    #print(ds.data.head())
    geoDs = ds.asGeoDataSet()
    #print(geoDs.data.head())
    print(geoDs.data.x.count())
    geoDs.applyMask(glacierMask, 'Glacier')
    print(geoDs.stats)
    print(geoDs.data.x.count())

    geoDs.withinMask(debrisMask, 'Debris')
    print(geoDs.stats)
    print(geoDs.data.x.count())
    print(geoDs.data.head())