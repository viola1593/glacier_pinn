"""Expands the OGGM workflow with additional tasks for preprocessing and data handling. All functions are adapted from OGGM code."""
from oggm.utils import ncDataset
from oggm import entity_task
import logging
from scipy import optimize as optimization
import salem
import numpy as np
import xarray as xr
from astropy import convolution
import pandas as pd

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio import MemoryFile
    try:
        # rasterio V > 1.0
        from rasterio.merge import merge as merge_tool
    except ImportError:
        from rasterio.tools.merge import merge as merge_tool
except ImportError:
    pass

from packaging.version import Version
 

# OGGM
import oggm
import oggm.cfg as cfg
from oggm import utils
from oggm.shop import its_live
from oggm.core.gis import gaussian_blur
from oggm.core.massbalance import ConstantMassBalance
from oggm.exceptions import InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

def search_RGI_ID_in_gdirs(id, gdirs):
    """Searches through the list of GlacierDirectories for a glacier RGI ID and returns GlacierDirectory's index in the list of GlacierDirectories and the GlacierDirectory itself.
    Args: 
        id: str, RGI ID of the glacier
        gdirs: list of GlacierDirectories
    Returns:
        i: int, index of the GlacierDirectory in the list of GlacierDirectories
        gdir: GlacierDirectory, the GlacierDirectory with the RGI ID
    """
    for i, gdir in enumerate(gdirs):
        if gdir.rgi_id==id:
            return i, gdir
        
def get_gridded_nc(gdir):
    """Reads the gridded data file of a given GlacierDirectory and returns it as xarray dataset."""
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()  
    return ds


def gaussian_astro_blur(in_array, size):
    """Applies a Gaussian filter to a 2d array using astropys convolve function. 
    It is able to handle nan values, projects them to 0 if there is too many. 

    Parameters
    ----------
    in_array : numpy.array
        The array to smooth.
    size : intpy
        The half size of the smoothing window.

    Returns
    -------
    a smoothed numpy.array
    """

    #exchange nans for np.nans 
    nan_mask = np.isnan(in_array)
    in_array=np.where(nan_mask,np.nan,in_array)

    # build kernel
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)

    # do the Gaussian blur
    return convolution.convolve_fft(in_array, g)



# smooth hugonnet dhdt
@entity_task(log, writes=['gridded_data'])
def smooth_hugonnet_dhdt(gdir):
    """Reads hugonnet dhdt data and smoothes it if cfg.PARAMS['smooth_window'] is greater than 0. 
    The data is then written to `gridded_data.nc`.
    This function is originally used to smooth the topo variable in the gridded_nc file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data"""
    if cfg.PARAMS['smooth_window'] > 0.:
            
        with ncDataset(gdir.get_filepath('gridded_data')) as nc:
            hugo = nc.variables['hugonnet_dhdt'][:]
        gsize = np.rint(cfg.PARAMS['smooth_window'] / gdir.grid.dx)
        smoothed_hugo = gaussian_astro_blur(hugo, int(gsize))

    
    # Save to file
        with ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
            vn = 'dhdt_smoothed'
            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x', ))
            v.units = 'm'
            v.long_name = 'dhdt (2000-2020) from Hugonnet et al. 2021, smoothed with radius: {:.1} m'.format(cfg.PARAMS['smooth_window'])
            v.description = ('original data at /home/data/download/cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb_maps/dhdt/N77E014_2000-01-01_2020-01-01_dhdt.tif /home/data/download/cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb_maps/dhdt/N77E015_2000-01-01_2020-01-01_dhdt.tif')
            v[:] = smoothed_hugo


# smooth hugonnet dhdt
@entity_task(log, writes=['gridded_data'])
def smooth_hugonnet_dhdt2014(gdir):
    """Reads hugonnet dhdt data and smoothes it if cfg.PARAMS['smooth_window'] is greater than 0. 
    The data is then written to `gridded_data.nc`.
    This function is originally used to smooth the topo variable in the gridded_nc file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data"""
    if cfg.PARAMS['smooth_window'] > 0.:
            
        with ncDataset(gdir.get_filepath('gridded_data')) as nc:
            hugo = nc.variables['hugonnet_dhdt2014'][:]
        gsize = np.rint(cfg.PARAMS['smooth_window'] / gdir.grid.dx)
        smoothed_hugo = gaussian_astro_blur(hugo, int(gsize))
   
    # Save to file
        with ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
            vn = 'dhdt2014_smoothed'
            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x', ))
            v.units = 'm'
            v.long_name = 'dhdt (2015-2020) from Hugonnet et al. 2021, smoothed with radius: {:.1} m'.format(cfg.PARAMS['smooth_window'])
            v.description = ('')
            v[:] = smoothed_hugo

@entity_task(log, writes=['gridded_data'])
def smooth_millan(gdir):
    """Reads millan vx and vy data and smoothes it if cfg.PARAMS['smooth_window'] is greater than 0. 
    The data is then written to `gridded_data.nc`.
    This function is originally used to smooth the topo variable in the gridded_nc file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data"""
    if cfg.PARAMS['smooth_window'] > 0.:
        for vel in ['millan_vx', 'millan_vy', 'millan_v']:  
            with ncDataset(gdir.get_filepath('gridded_data')) as nc:
                millanv = nc.variables[vel][:]
            gsize = np.rint(cfg.PARAMS['smooth_window'] / gdir.grid.dx)
            smoothed_millanv = gaussian_astro_blur(millanv, int(gsize))

        # Save to file
            with ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:


                vn = vel+'_smoothed'
                if vn in nc.variables:
                    v = nc.variables[vn]
                else:
                    v = nc.createVariable(vn, 'f4', ('y', 'x', ))
                v.units = 'm/a'
                v.long_name = vel +'Millan et al. 2022, smoothed with radius: {:.1} m'.format(cfg.PARAMS['smooth_window'])
                v.description = ('RGI-7.1_2021July01')
                v[:] = smoothed_millanv





@entity_task(log)
def gridded_oggm_mb_v16(gdir, y0=2017, halfsize=1):
    """Adds mass balance related attributes to the gridded data file.

    This could be useful for distributed ice thickness models.
    The raster data are added to the gridded_data file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    # Get the input data
    with ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo_2d = nc.variables['topo_smoothed'][:]
        glacier_mask_2d = nc.variables['glacier_mask'][:]
        glacier_mask_2d = glacier_mask_2d == 1


    topo = topo_2d[glacier_mask_2d]

    # Prepare the distributed mass balance data
    rho = cfg.PARAMS['ice_density']
    dx2 = gdir.grid.dx ** 2

    # Normal OGGM (a bit tweaked)
    def to_minimize(temp_bias):
        mbmod = ConstantMassBalance(gdir, temp_bias=temp_bias, y0=y0, halfsize=halfsize,
                                    check_calib_params=False)
        smb = mbmod.get_annual_mb(heights=topo)
        return np.sum(smb)**2
    opt = optimization.minimize(to_minimize, [0.], method='Powell')
    mbmod = ConstantMassBalance(gdir, temp_bias=float(opt['x']), y0=y0, halfsize=halfsize,
    
                                check_calib_params=False)
    oggm_mb_on_z = mbmod.get_annual_mb(heights=topo) * cfg.SEC_IN_YEAR #* rho # to have it in m/year (coming from volume/area/year) same unit as velocity 
    if not np.isclose(np.sum(oggm_mb_on_z), 0, atol=10):
        raise RuntimeError('Spec mass balance should be zero but is: {}'
                           .format(np.sum(oggm_mb_on_z)))


    # Make 2D again
    def _fill_2d_like(data):
        out = topo_2d * np.NaN
        out[glacier_mask_2d] = data
        return out

    
    oggm_mb_on_z = _fill_2d_like(oggm_mb_on_z)


    # Save to file
    with ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'oggm_mb_on_z'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',))
        v.units = 'kg/year'
        v.long_name = 'MB above point from OGGM MB model, without catchments'
        v.description = ('Mass balance cumulated above the altitude of the'
                         'point, hence in unit of flux. Note that it is '
                         'a coarse approximation of the real flux. '
                         'The mass balance model is a calibrated temperature '
                         'index model like OGGM.')
        v[:] = oggm_mb_on_z






@utils.entity_task(log, writes=['gridded_data'])
def get_gridded_yearly_velocity(gdir, year=None, add_error=False):
    
    '''Downloads the ITS_LIVE velocity files from the base url which leads to the velocity files for separate years
    Changed the _reproject_and_scale function to take another argument of different region files
    
    '''
    
    region_files = {}
    d = {}
    if year is not None:
        base_url = ('http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/'
            'velocity_mosaic/landsat/v00.0/annual/cog/')
        
        if year <1985:
            year =1985 #earliest velocities are from 1985
        print(year)
        # get region file for given year
        for var in ['vx', 'vy', 'vy_err', 'vx_err']:        
            d[var] = base_url + 'SRA_G0240_{}_{}.tif'.format(year, var)
        region_files['SRA'] = d
        if not gdir.has_file('gridded_data'):
            raise InvalidWorkflowError('Please run `glacier_masks` before running '
                                    'this task')
        try:
            its_live._reproject_and_scale(gdir, do_error=False, region_files=region_files) # modified function in oggm.shop.itslive
            if add_error:
                its_live._reproject_and_scale(gdir, do_error=True, region_files=region_files)#add error as well


        except OSError:
            print(str(gdir.rgi_id)+' could not get yearly velocity')

    else: 
        base_url = ('http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/'
            'velocity_mosaic/landsat/v00.0/static/cog/')
        

        region_files = {}
        
        d = {}
        for var in ['vx', 'vy', 'vy_err', 'vx_err']:
            d[var] = base_url + 'SRA_G0120_0000_{}.tif'.format(var)
        region_files['SRA'] = d

        
        
        if not gdir.has_file('gridded_data'):
            raise InvalidWorkflowError('Please run `glacier_masks` before running '
                                   'this task')

        its_live._reproject_and_scale(gdir, do_error=False,region_files=region_files)
        if add_error:
            its_live._reproject_and_scale(gdir, do_error=True,region_files=region_files)



# consensus estimate to gdir (only needed for older version of OGGM)
default_base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/icevol/composite/'

@utils.entity_task(log, writes=['gridded_data'])
def add_consensus_thickness(gdir, base_url=None):
    """Add the consensus thickness estimate to the gridded_data file. 
    Copied from https://docs.oggm.org/en/v1.5.3/_modules/oggm/shop/bedtopo.html because the bedtopo module is apparently missing in my version of oggm.shop

    varname: consensus_ice_thickness

    Parameters
    ----------
    gdir ::py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    base_url : str
        where to find the thickness data. Default is
        https://cluster.klima.uni-bremen.de/~fmaussion/icevol/composite
    """

    if base_url is None:
        base_url = default_base_url
    if not base_url.endswith('/'):
        base_url += '/'

    rgi_str = gdir.rgi_id
    rgi_reg_str = rgi_str[:8]

    url = base_url + rgi_reg_str + '/' + rgi_str + '_thickness.tif'
    input_file = utils.file_downloader(url)

    dsb = salem.GeoTiff(input_file)
    thick = utils.clip_min(dsb.get_vardata(), 0)
    in_volume = thick.sum() * dsb.grid.dx ** 2
    thick = gdir.grid.map_gridded_data(thick, dsb.grid, interp='linear')

    # Correct for volume
    thick = utils.clip_min(thick.filled(0), 0)
    out_volume = thick.sum() * gdir.grid.dx ** 2
    if out_volume > 0:
        thick *= in_volume / out_volume

    # We mask zero ice as nodata
    thick = np.where(thick == 0, np.NaN, thick)

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'consensus_ice_thickness'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm'
        ln = 'Ice thickness from the consensus estimate'
        v.long_name = ln
        v.base_url = base_url
        v[:] = thick
        

# hugonnet dhdt to gdir, downloads the product for 2014-2019      
_lookup_csv = None

# url to the hugonnet dhdt data
dhdt_url= 'data/dhdt2014_2019'
def _get_lookup_csv():
    global _lookup_csv
    if _lookup_csv is None:
        fname = '../'+dhdt_url + '/hugonnet_dhdt_lookup_csv_20230129.csv'
        _lookup_csv = pd.read_csv(fname, index_col=0) # lookup table to find the correct file for a given lon/lat, adapted from the OGGM lookup table to work with the files for 2015-2020
    return _lookup_csv


@utils.entity_task(log, writes=['gridded_data'])
def hugonnet2014_to_gdir(gdir, add_error=False):
    """Add the Hugonnet 21 dhdt maps to this glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_error : bool
        add the error data or not
    """

    if add_error:
        raise NotImplementedError('Not yet')

    # Find out which file(s) we need
    df = _get_lookup_csv()
    lon_ex, lat_ex = gdir.extent_ll

    # adding small buffer for unlikely case where one lon/lat_ex == xx.0
    lons = np.arange(np.floor(lon_ex[0] - 1e-9), np.ceil(lon_ex[1] + 1e-9))
    lats = np.arange(np.floor(lat_ex[0] - 1e-9), np.ceil(lat_ex[1] + 1e-9))

    flist = []
    for lat in lats:
        # north or south?
        ns = 'S' if lat < 0 else 'N'
        for lon in lons:
            # east or west?
            ew = 'W' if lon < 0 else 'E'
            ll_str = f'{ns}{abs(lat):02.0f}{ew}{abs(lon):03.0f}'
            try:
                filename = df.loc[(df['file_id'] == ll_str)]['dhdt'].iloc[0]
            except IndexError:
                # We can maybe be on the edge (unlikely but hey
                pass
            file_local = '../'+dhdt_url +'/'+ filename
            if file_local is not None:
                flist.append(file_local)

    # A glacier area can cover more than one tile:
    if len(flist) == 1:
        dem_dss = [rasterio.open(flist[0])]  # if one tile, just open it
        file_crs = dem_dss[0].crs
        dem_data = rasterio.band(dem_dss[0], 1)
        if Version(rasterio.__version__) >= Version('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
        nodata = dem_dss[0].meta.get('nodata', None)
    else:
        dem_dss = [rasterio.open(s) for s in flist]  # list of rasters
        # make sure all files have the same crs and reproject if needed;
        # defining the target crs to the one most commonly used, minimizing
        # the number of files for reprojection
        crs_list = np.array([dem_ds.crs.to_string() for dem_ds in dem_dss])
        unique_crs, crs_counts = np.unique(crs_list, return_counts=True)
        file_crs = rasterio.crs.CRS.from_string(
            unique_crs[np.argmax(crs_counts)])

        if len(unique_crs) != 1:
            # more than one crs, we need to do reprojection
            memory_files = []
            for i, src in enumerate(dem_dss):
                if file_crs != src.crs:
                    transform, width, height = calculate_default_transform(
                        src.crs, file_crs, src.width, src.height, *src.bounds)
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': file_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })

                    reprojected_array = np.empty(shape=(src.count, height, width),
                                                 dtype=src.dtypes[0])
                    # just for completeness; even the data only has one band
                    for band in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, band),
                                  destination=reprojected_array[band - 1],
                                  src_transform=src.transform,
                                  src_crs=src.crs,
                                  dst_transform=transform,
                                  dst_crs=file_crs,
                                  resampling=Resampling.nearest)

                    memfile = MemoryFile()
                    with memfile.open(**kwargs) as mem_dst:
                        mem_dst.write(reprojected_array)
                    memory_files.append(memfile)
                else:
                    memfile = MemoryFile()
                    with memfile.open(**src.meta) as mem_src:
                        mem_src.write(src.read())
                    memory_files.append(memfile)

            with rasterio.Env():
                datasets_to_merge = [memfile.open() for memfile in memory_files]
                nodata = datasets_to_merge[0].meta.get('nodata', None)
                dem_data, src_transform = merge_tool(datasets_to_merge,
                                                     nodata=nodata)
        else:
            # only one single crs occurring, no reprojection needed
            nodata = dem_dss[0].meta.get('nodata', None)
            dem_data, src_transform = merge_tool(dem_dss, nodata=nodata)

    # Set up profile for writing output
    with rasterio.open(gdir.get_filepath('dem')) as dem_ds:
        dst_array = dem_ds.read().astype(np.float32)
        dst_array[:] = np.NaN
        profile = dem_ds.profile
        transform = dem_ds.transform
        dst_crs = dem_ds.crs

    # Set up profile for writing output
    profile.update({
        'nodata': np.NaN,
    })

    resampling = Resampling.bilinear

    with MemoryFile() as dest:
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=file_crs,
            src_transform=src_transform,
            src_nodata=nodata,
            # Destination parameters
            destination=dst_array,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.NaN,
            # Configuration
            resampling=resampling)
        dest.write(dst_array)

    for dem_ds in dem_dss:
        dem_ds.close()

    # Write to the gridded data file
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:

        vn = 'hugonnet_dhdt2014'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'm'
        ln = 'dhdt (2014-2019) from Hugonnet et al. 2021'
        v.long_name = ln
        data_str = ' '.join(flist) if len(flist) > 1 else flist[0]
        v.data_source = data_str
        v[:] = np.squeeze(dst_array)