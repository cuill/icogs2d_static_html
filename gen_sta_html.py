# coding: utf-8


"""
Dynamic map hindcast implementation 
"""
__author__ = 'Saeed Moghimi'
__copyright__ = 'Copyright 2020, UCAR/NOAA'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'moghimis@gmail.com'

# Thu 19 Apr 2018 03:08:06 PM EDT


###############################################################
# Original development from https://github.com/ocefpaf/python_hurricane_gis_map
# # Exploring the NHC GIS Data
#
# This notebook aims to demonstrate how to create a simple interactive GIS map with the National Hurricane Center predictions [1] and CO-OPS [2] observations along the Hurricane's path.
#
#
# 1. http://www.nhc.noaa.gov/gis/
# 2. https://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/
#
#
# NHC codes storms are coded with 8 letter names:
# - 2 char for region `al` &rarr; Atlantic
# - 2 char for number `11` is Irma
# - and 4 char for year, `2017`
#
# Browse http://www.nhc.noaa.gov/gis/archive_wsurge.php?year=2017 to find other hurricanes code.
###############################################################


#
from datetime import datetime, timedelta
import os
import sys

import arrow
from branca.element import *
#
import folium
from folium.plugins import Fullscreen, MarkerCluster, MousePosition
import matplotlib as mpl
import netCDF4
from shapely.geometry import LineString

mpl.use('Agg')

import matplotlib.tri as Tri
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.models import Title
from bokeh.embed import file_html
from bokeh.models import Range1d, HoverTool
from glob import glob

from geopandas import GeoDataFrame

from pyschism.mesh.hgrid import Hgrid
# from ioos_tools.ioos import get_coordinates
# import pickle
# import string
# import fiona


try:
    for filename in glob('./**/hurricane_funcs*.pyc'):
        os.remove(filename)
except:
    pass
if 'hurricane_funcs' in sys.modules:
    del sys.modules['hurricane_funcs']
from hurricane_funcs import *

############################
from matplotlib.colors import LinearSegmentedColormap

cdict = {
    'red': (
        (0.0, 1, 1),
        (0.05, 1, 1),
        (0.11, 0, 0),
        (0.66, 1, 1),
        (0.89, 1, 1),
        (1, 0.5, 0.5),
    ),
    'green': (
        (0.0, 1, 1),
        (0.05, 1, 1),
        (0.11, 0, 0),
        (0.375, 1, 1),
        (0.64, 1, 1),
        (0.91, 0, 0),
        (1, 0, 0),
    ),
    'blue': ((0.0, 1, 1), (0.05, 1, 1), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0)),
}

jetMinWi = LinearSegmentedColormap('my_colormap', cdict, 256)
my_cmap = plt.cm.jet

####################################
######## process command line args
arg = sys.argv
if len(arg) < 3:
    print('#######################################################')
    print('Usage: Pass line arguments as: ')
    print('python gen_sta_html.py <RUNdir> <ModelDomain>')
    print('<RUNdir> : path to run folder ')
    print('<ModelDomain> : icogs2d')
    print('######################################################')
    sys.exit()
# Get args
RUNdir = arg[1]
print(f'RUNdir {RUNdir}')
ModelDomain = arg[2]
print(f'ModelDomain {ModelDomain}')


###############################################################
# Functions


def ceil_dt(dt=datetime.now(), delta=timedelta(minutes=30)):
    """
    now = datetime.now()
    print(now)    
    print(ceil_dt(now,timedelta(minutes=30) ))

    """

    return dt + (datetime.min - dt) % delta


######################################
# Let's create a color code for the point track.
colors_hurricane_condition = {
    'subtropical depression': '#ffff99',
    'tropical depression': '#ffff66',
    'tropical storm': '#ffcc99',
    'subtropical storm': '#ffcc66',
    'hurricane': 'red',
    'major hurricane': 'crimson',
}
#######################################
############################################################
# plot ssh to pop up when click on obs locations
##

tools = 'pan,box_zoom,reset'
width, height = 750, 250


def make_plot_1line(obs, label=None):
    # TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    TOOLS = 'crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,save,'

    p = figure(
        toolbar_location='above',
        x_axis_type='datetime',
        width=width,
        height=height,
        tools=TOOLS,
    )

    p.add_layout(
        Title(text=f"Station: {obs._metadata['station_code']}", text_font_style='italic'),
        'above',
    )
    p.add_layout(Title(text=obs._metadata['station_name'], text_font_size='10pt'), 'above')

    p.yaxis.axis_label = label

    obs_val = obs.values.squeeze()

    l1 = p.line(
        x=obs.index,
        y=obs_val,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend_label='model',
        color='#0000ff',
        alpha=0.7,
    )

    minx = obs.index.min()
    maxx = obs.index.max()

    p.x_range = Range1d(start=minx, end=maxx)

    p.legend.location = 'top_left'

    p.add_tools(HoverTool(tooltips=[('model', '@y'), ], renderers=[l1], ), )
    return p


# def make_plot(obs, model = None,label,remove_mean_diff=False,bbox_bias=None):
def make_plot_2line(obs, model=None, label=None, remove_mean_diff=False, bbox_bias=0.0):
    # TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    TOOLS = 'crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,save,'

    p = figure(
        toolbar_location='above',
        x_axis_type='datetime',
        width=width,
        height=height,
        tools=TOOLS,
    )

    p.add_layout(
        Title(text='Station: ' + obs._metadata['station_code'], text_font_style='italic'),
        'above',
    )
    p.add_layout(Title(text=obs._metadata['station_name'], text_font_size='10pt'), 'above')

    p.yaxis.axis_label = label

    obs_val = obs.values.squeeze()

    l1 = p.line(
        x=obs.index,
        y=obs_val,
        line_width=5,
        line_cap='round',
        line_join='round',
        legend_label='obs.',
        color='#0000ff',
        alpha=0.7,
    )

    if model is not None:
        mod_val = model.values.squeeze()

        if ('SSH' in label) and remove_mean_diff:
            mod_val = mod_val + obs_val.mean() - mod_val.mean()

        if ('SSH' in label) and bbox_bias is not None:
            mod_val = mod_val + bbox_bias

        l0 = p.line(
            x=model.index,
            y=mod_val,
            line_width=5,
            line_cap='round',
            line_join='round',
            legend_label='model',
            color='#9900cc',
            alpha=0.7,
        )

        minx = max(model.index.min(), obs.index.min())
        maxx = min(model.index.max(), obs.index.max())

        minx = model.index.min()
        maxx = model.index.max()
    else:
        minx = obs.index.min()
        maxx = obs.index.max()

    p.x_range = Range1d(start=minx, end=maxx)

    p.legend.location = 'top_left'

    p.add_tools(
        HoverTool(tooltips=[('model', '@y'), ], renderers=[l0], ),
        HoverTool(tooltips=[('obs.', '@y'), ], renderers=[l1], ),
    )

    return p


#################
def make_marker(p, location, fname, color='green', icon='stats'):
    html = file_html(p, CDN, fname)
    # iframe = IFrame(html , width=width+45+height, height=height+80)
    iframe = IFrame(html, width=width * 1.1, height=height * 1.2)
    # popup = folium.Popup(iframe, max_width=2650+height)
    popup = folium.Popup(iframe)
    iconm = folium.Icon(color=color, icon=icon)
    marker = folium.Marker(location=location, popup=popup, icon=iconm)
    return marker


###############################
###try to add countor to map


def Read_maxele_return_plot_obj(fgrd='hgrid.gr3', felev='maxelev.gr3'):
    
    hgrid=Hgrid.open(fgrd, crs='EPSG:4326')
    h=-hgrid.values
    bbox=hgrid.get_bbox('EPSG:4326', output_type='bbox')

    elev=Hgrid.open(felev, crs='EPSG:4326')
    mzeta=-elev.values
    D=mzeta

    #Mask dry nodes
    NP=len(mzeta)
    idxs=np.where(h < 0)
    D[idxs]=np.maximum(0, mzeta[idxs]+h[idxs])

    idry=np.zeros(NP)
    idxs=np.where(mzeta+h <= 1e-6)
    idry[idxs]=1

    MinVal = np.min(mzeta)
    MaxVal = np.max(mzeta)
    NumLevels = 21

    if True:
        MinVal = max(MinVal, 0.0)
        MaxVal = min(MaxVal, 2.4)
        NumLevels = 12
    print(f'MinVal is {MinVal}')
    print(f'MaxVal is {MaxVal}')

    step = 0.2  # m
    #step = 1  # m
    #levels = np.linspace(MinVal, MaxVal, num=NumLevels)
    levels = np.arange(MinVal, MaxVal + step, step=step)
    print(f'levels is {levels}')

    fig=plt.figure()
    ax=fig.add_subplot()
    tri = elev.triangulation
    mask=np.any(np.where(idry[tri.triangles], True, False), axis=1)
    tri.set_mask(mask)

    contour = ax.tricontourf(tri, mzeta, vmin=MinVal, vmax=MaxVal,levels=levels, cmap=my_cmap, extend='max')
    return contour, MinVal, MaxVal, levels


#############################################################
def collec_to_gdf(collec_poly):
    """Transform a `matplotlib.contour.QuadContourSet` to a GeoDataFrame"""
    polygons, colors = [], []
    for i, polygon in enumerate(collec_poly.collections):
        mpoly = []
        for path in polygon.get_paths():
            try:
                path.should_simplify = False
                poly = path.to_polygons()
                # Each polygon should contain an exterior ring + maybe hole(s):
                exterior, holes = [], []
                if len(poly) > 0 and len(poly[0]) > 3:
                    # The first of the list is the exterior ring :
                    exterior = poly[0]
                    # Other(s) are hole(s):
                    if len(poly) > 1:
                        holes = [h for h in poly[1:] if len(h) > 3]
                mpoly.append(Polygon(exterior, holes))
            except:
                print('Warning: Geometry error when making polygon #{}'.format(i))
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
            colors.append(polygon.get_facecolor().tolist()[0])
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
            colors.append(polygon.get_facecolor().tolist()[0])
    return GeoDataFrame(geometry=polygons, data={'RGBA': colors}, crs={'init': 'epsg:4326'})


#################
def convert_to_hex(rgba_color):
    red = str(hex(int(rgba_color[0] * 255)))[2:].capitalize()
    green = str(hex(int(rgba_color[1] * 255)))[2:].capitalize()
    blue = str(hex(int(rgba_color[2] * 255)))[2:].capitalize()

    if blue == '0':
        blue = '00'
    if red == '0':
        red = '00'
    if green == '0':
        green = '00'

    return '#' + red + green + blue


#################
def get_station_ssh(fort61, staName):
    """
        Read model ssh
    """
    #nc0 = netCDF4.Dataset(fort61)
    #ncv0 = nc0.variables
    #sta_lon = ncv0['x'][:]
    #sta_lat = ncv0['y'][:]
    #sta_nam = ncv0['station_name'][:].squeeze()
    #sta_zeta = ncv0['zeta'][:].squeeze()
    #sta_date = netCDF4.num2date(ncv0['time'][:], ncv0['time'].units)

    print(f'getstationssh Date is {Date}')
    year=Date[:4]
    month=Date[4:6]
    day=Date[6:]
    hour=FCT
    startdate=datetime(int(year),int(month),int(day),int(hour))
    #enddate=startdate+timedelta(days=3)
    #sta_date = np.arange(startdate, enddate,np.timedelta64(30,'m'), dtype='datetime64')

    df=pd.read_csv(staName, index_col=[0])
    stationIDs = df['ID']
    sta_lon=df['lon']
    sta_lat=df['lat']
    station_names = df['Name']


    #Read model output
    sta_data=np.loadtxt(fort61)
    time=sta_data[:,0]
    nt=len(time)
    nstation=len(sta_lat)
    sta_zeta=np.ndarray(shape=(nt,nstation), dtype=float)
    sta_zeta=sta_data[:,1:]
    units=f'seconds since {startdate.year}-{startdate.month}-{startdate.day} 00:00:00 UTC'
    sta_date=netCDF4.num2date(time,units)
    print(f'sta_date: startdate is {sta_date[0]}, enddate is {sta_date[-1]}')

    #print(len(sta_zeta[:,1]))
    mod = []
    print(type(sta_date))
    print(type(sta_zeta))
    ind = np.arange(len(sta_lat))
    print(ind)
    for ista in ind:
        mod_tmp = pd.DataFrame(
            data=np.c_[sta_date, sta_zeta[:, ista]], columns=['date_time', 'ssh']
        ).set_index('date_time')
        mod_tmp._metadata = stationIDs
        mod.append(mod_tmp)

    stationIDs = np.array(stationIDs)
    station_names = np.array(station_names)

    mod_table = pd.DataFrame(
        data=np.c_[ind, stationIDs, station_names, sta_lon, sta_lat],
        columns=['ind', 'station_code', 'station_name', 'lon', 'lat'],
    )

    return mod, mod_table


#############################################################
def make_map(bbox, **kw):
    """
    Creates a folium map instance.

    Examples
    --------
    >>> from folium import Map
    >>> bbox = [-87.40, 24.25, -74.70, 36.70]
    >>> m = make_map(bbox)
    >>> isinstance(m, Map)
    True

    """
    import folium

    line = kw.pop('line', True)
    layers = kw.pop('layers', True)
    zoom_start = kw.pop('zoom_start', 5)

    lon, lat = np.array(bbox).reshape(2, 2).mean(axis=0)
    #
    m = folium.Map(width='100%', height='100%', location=[lat, lon], zoom_start=zoom_start)

    if layers:
        add = 'MapServer/tile/{z}/{y}/{x}'
        base = 'http://services.arcgisonline.com/arcgis/rest/services'
        ESRI = dict(
            Imagery='World_Imagery/MapServer',
            # Ocean_Base='Ocean/World_Ocean_Base',
            # Topo_Map='World_Topo_Map/MapServer',
            # Physical_Map='World_Physical_Map/MapServer',
            # Terrain_Base='World_Terrain_Base/MapServer',
            # NatGeo_World_Map='NatGeo_World_Map/MapServer',
            # Shaded_Relief='World_Shaded_Relief/MapServer',
            # Ocean_Reference='Ocean/World_Ocean_Reference',
            # Navigation_Charts='Specialty/World_Navigation_Charts',
            # Street_Map='World_Street_Map/MapServer'
        )

        for name, url in ESRI.items():
            url = '{}/{}/{}'.format(base, url, add)

            w = folium.TileLayer(tiles=url, name=name, attr='ESRI', overlay=False)
            w.add_to(m)

    if line:  # Create the map and add the bounding box line.
        p = folium.PolyLine(
            get_coordinates(bbox), color='#FF0000', weight=2, opacity=0.5, latlon=True
        )
        p.add_to(m)

    folium.LayerControl().add_to(m)
    return m


########################################
####       MAIN CODE from HERE     #####
########################################

# Default values
#STORM = None  # 'DELTA_2020'    #format strom_year e.g. 'SANDY_2012'
STORM = 'ELSA_2021'  # 'DELTA_2020'    #format strom_year e.g. 'SANDY_2012'
# NcMeshfile   = os.path.join(RUNdir,'../inp/hsofs_atl_mesh.nc')
BBOX = '-99.0,5.0,-52.8,46.3'  # format  lon_min,lat_min,lon_max,lat_max x1,y1,x2,y2  #hsofs
WlDatum = 'MSL'
#################################################
# update from config file

conf_name = os.path.join(RUNdir, 'config.ini')
if os.path.exists(conf_name):
    fconf = open(conf_name, 'r')
    for line in fconf:
        words = line.split()
        if 'STORM' in line:
            STORM = words[1]
        if 'NcMeshfile' in line:
            NcMeshfile = words[1]
        if 'BBOX' in line:
            BBOX = words[1]
        if 'WlDatum' in line:
            WlDatum = words[1]
    fconf.close()
else:
    # prepare conf file
    fconf = os.path.join(conf_name)
    f = open(fconf, 'w')
    f.write('STORM         ' + str(STORM) + ' \n')
    f.write('NcMeshfile    ' + NcMeshfile + ' \n')
    f.write('BBOX          ' + BBOX + ' \n')
    f.write('WlDatum       ' + WlDatum + ' \n')
    f.close()

if STORM == 'None':
    STORM = None

#########################################################
new = os.path.join(RUNdir, 'new.ini')
if os.path.exists(new):
    fnew = open(new)
else:
    sys.exit('New forecast is not arrived yet ....   !')

for line in fnew:
    words = line.split()
    if 'ModelDomain' in line:
        ModelDomain = words[1]
    if 'Date' in line:
        Date = words[1]
    if 'FCT' in line:
        FCT = words[1]
    if 'RUNdir' in line:
        RUNdir = words[1]
fnew.close()

old = os.path.join(RUNdir, Date, FCT, 'old.ini')
os.rename(new, old)
####################################################

# Date = '20201007'

print('Date:', Date)
apply_bbox_bias = False

work_dir = os.path.join(RUNdir, Date, FCT)
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

print('work_dir:', work_dir)

fort61 = glob(work_dir + '/*staout*')[0]
#print(fort61)
if not os.path.exists(fort61):
    sys.exit('ERROR: Points time series is not exist !!!!')

felev = glob(work_dir + '/*maxelev*')[0]
if not os.path.exists(felev):
    sys.exit('ERROR: Maxelev file is not exist !!!!')

fgrd = os.path.join(NcMeshfile)
if not os.path.exists(fgrd):
    sys.exit(
        'ERROR: If NcMeshfile is not in RUNdir provide full path to NcMeshfile in config.ini'
    )

freq = '30min'

######
if STORM is not None:
    name = STORM.split('_')[0]
    year = STORM.split('_')[1]
    #
    print('\n\n\n\n\n\n********************************************************')
    print('*****  Storm name ', name, '      Year ', year, '    *********')
    print('******************************************************** \n\n\n\n\n\n')

    print(' > Read NHC information ... ')
    al_code, hurricane_gis_files = get_nhc_storm_info(year, name)
    print(f'hurricane_gis_files: {hurricane_gis_files}')

    # donload gis zip files
    base = download_nhc_gis_files(hurricane_gis_files, work_dir)
    print(f'base: {base}')

    # get advisory cones and track points
    cones, points, pts = read_advisory_cones_info(hurricane_gis_files, base, year, al_code)

    # Find the bounding box to search the data.
    bbox_from_track = True
    if bbox_from_track:
        last_cone = cones[-1]['geometry'].iloc[0]
        track = LineString([point['geometry'] for point in points])
        track_lons = track.coords.xy[0]
        track_lats = track.coords.xy[1]
        bbox = (
            min(track_lons) - 2,
            min(track_lats) - 2,
            max(track_lons) + 2,
            max(track_lats) + 2,
        )
    else:
        bounds = np.array([last_cone.buffer(2).bounds, track.buffer(2).bounds]).reshape(4, 2)
        lons, lats = bounds[:, 0], bounds[:, 1]
        bbox = lons.min(), lats.min(), lons.max(), lats.max()

    #####################################
    # Now we can get all the information we need from those GIS files. Let's start with the event dates.
    ######################################

    # We are ignoring the timezone, like AST (Atlantic Time Standar) b/c
    # those are not a unique identifiers and we cannot disambiguate.

    if 'FLDATELBL' in points[0].keys():
        start = points[0]['FLDATELBL']
        end = points[-1]['FLDATELBL']
        date_key = 'FLDATELBL'

        start_dt = arrow.get(start, 'YYYY-MM-DD h:mm A ddd').datetime
        end_dt = arrow.get(end, 'YYYY-MM-DD h:mm A ddd').datetime
    elif 'ADVDATE' in points[0].keys():
        # older versions (e.g. IKE)
        start = points[0]['ADVDATE']
        end = points[-1]['ADVDATE']
        date_key = 'ADVDATE'

        start_dt = arrow.get(start, 'YYMMDD/hhmm').datetime
        end_dt = arrow.get(end, 'YYMMDD/hhmm').datetime
    else:
        print('Check for correct time stamp and adapt the code !')
        sys.exit('ERROR')

    strbbox = ', '.join(format(v, '.2f') for v in bbox)
    print(' > bbox: {}\nstart: {}\n  end: {}'.format(strbbox, start, end))
    bbox = strbbox
else:
    name = None
    year = None
    bbox = BBOX

ssh = ssh_from_model = None

############################################################
########### Read and process SSH from model
staName='./inpdir/stations_noaa-coops_150.csv'
mod, mod_table = get_station_ssh(fort61, staName)

try:
    start_dt = pd.to_datetime(mod[0].index[0])
    end_dt = pd.to_datetime(mod[0].index[-1])
    # Ceil dt to 30 minutes
    start_dt = ceil_dt(start_dt)
    end_dt = ceil_dt(end_dt)
except:
    st = mod[0].index[0]
    et = mod[0].index[-1]

    # Ceil dt to 30 minutes
    start_dt = ceil_dt(datetime(st.year, st.month, st.day, st.hour, st.minute, 0))
    end_dt = ceil_dt(datetime(et.year, et.month, et.day, et.hour, et.minute, 0))
    # start_dt = datetime(stt.year # end_dt   = arrow.get(end,   'YY-MM-DDThh:mm:ss').datetime 

index = pd.date_range(
    start=start_dt.replace(tzinfo=None), end=end_dt.replace(tzinfo=None), freq=freq
)

# model
ssh_from_model = []
for im in range(len(mod)):
    series = mod[im]
    mod0 = series.reindex(index=index, limit=1, method='nearest')
    mod0._metadata = mod_table.iloc[im].to_dict()
    mod0['ssh'][np.abs(mod0['ssh']) > 10] = np.nan
    mod0.dropna(inplace=True)
    ssh_from_model.append(mod0)

#######################################################
print('  > Put together the final map')

bboxa = np.array(eval(bbox))
# Here is the final result. Explore the map by clicking on the map features plotted!
lon = 0.5 * (bboxa[0] + bboxa[2])
lat = 0.5 * (bboxa[1] + bboxa[3])
############################################################
# if  'FLDATELBL' in points[0].keys():
##
m = folium.Map(location=[lat, lon], tiles='OpenStreetMap', zoom_start=4, control_scale=True)
Fullscreen(position='topright', force_separate_button=True).add_to(m)

add = 'MapServer/tile/{z}/{y}/{x}'
base = 'http://services.arcgisonline.com/arcgis/rest/services'
ESRI = dict(Imagery='World_Imagery/MapServer')

for name1, url in ESRI.items():
    url = '{}/{}/{}'.format(base, url, add)

    w = folium.TileLayer(tiles=url, name=name1, attr='ESRI', overlay=False)
    w.add_to(m)

#################################################################
print('     > Plot max water elev ..')
contour, MinVal, MaxVal, levels = Read_maxele_return_plot_obj(fgrd=fgrd, felev=felev)
gdf = collec_to_gdf(contour)  # From link above
plt.close('all')

## Get colors in Hex
colors_elev = []
for i in range(len(gdf)):
    color = my_cmap(i / len(gdf))
    colors_elev.append(mpl.colors.to_hex(color))

# assign to geopandas obj
gdf['RGBA'] = colors_elev
#
# plot geopandas obj
maxele = folium.GeoJson(
    gdf,
    name='Disturbance [m]',
    style_function=lambda feature: {
        'fillColor': feature['properties']['RGBA'],
        'color': feature['properties']['RGBA'],
        'weight': 1.0,
        'fillOpacity': 0.6,
        'line_opacity': 0.6,
    },
)

maxele.add_to(m)

# Add colorbar
color_scale = folium.StepColormap(
    colors_elev,
    # index=color_domain,
    vmin=MinVal,
    vmax=MaxVal,
    caption='Disturbance [m]',
)
m.add_child(color_scale)
#################################################################
# try:
print('     > Plot SSH stations ..')
# marker_cluster_estofs_ssh = MarkerCluster(name='CO-OPS SSH observations')
marker_cluster_estofs_ssh = MarkerCluster(name='ESTOFS SSH [m above MSL]')
marker_cluster_estofs_ssh.add_to(m)

print('      > plot model only')
for ssh1 in ssh_from_model:
    fname = ssh1._metadata['station_code']
    #station=noaa.Station(int(fname))
    #print(f'station is {station}')
    location = ssh1._metadata['lat'], ssh1._metadata['lon']
    p = make_plot_1line(ssh1, label='SSH [m above MSL]')
#    # p = make_plot(ssh1, ssh1)
    marker = make_marker(p, location=location, fname=fname)
    marker.add_to(marker_cluster_estofs_ssh)

###################
if False:
    ## Plotting bounding box
    # folium.LayerControl().add_to(m)
    p = folium.PolyLine(get_coordinates(bboxa), color='#009933', weight=2, opacity=0.6)

    p.add_to(m)
#####################################################

if STORM is not None:
    plot_cones = True
    track_radius = 5
    if plot_cones:
        print('     > Plot NHC cone predictions')
        marker_cluster1 = MarkerCluster(name='NHC cone predictions')
        marker_cluster1.add_to(m)

        def style_function_latest_cone(feature):
            return {
                'fillOpacity': 0.1,
                'color': 'red',
                'stroke': 1,
                'weight': 1.5,
                'opacity': 0.8,
            }

        def style_function_cones(feature):
            return {
                'fillOpacity': 0,
                'color': 'lightblue',
                'stroke': 1,
                'weight': 0.3,
                'opacity': 0.3,
            }

        if True:
            # Latest cone prediction.
            latest = cones[-1]
            ###
            if 'FLDATELBL' in points[0].keys():  # Newer storms have this information
                names3 = 'Cone prediction as of {}'.format(latest['ADVDATE'].values[0])
            else:
                names3 = 'Cone prediction'
            ###
            folium.GeoJson(
                data=latest.__geo_interface__,
                name=names3,
                style_function=style_function_latest_cone,
            ).add_to(m)
        ###
        if False:
            if 'FLDATELBL' not in points[0].keys():  # Newer storms have this information
                names3 = 'Cone prediction'

                # Past cone predictions.
            for cone in cones[:-1]:
                folium.GeoJson(
                    data=cone.__geo_interface__, style_function=style_function_cones,
                ).add_to(marker_cluster1)

            # Latest points prediction.
            for k, row in pts.iterrows():

                if 'FLDATELBL' in points[0].keys():  # Newer storms have this information
                    date = row[date_key]
                    hclass = row['TCDVLP']
                    popup = '{}<br>{}'.format(date, hclass)
                    if 'tropical' in hclass.lower():
                        hclass = 'tropical depression'

                    color = colors_hurricane_condition[hclass.lower()]
                else:
                    popup = '{}<br>{}'.format(name, year)
                    color = colors_hurricane_condition['hurricane']

                location = row['LAT'], row['LON']
                folium.CircleMarker(
                    location=location,
                    radius=track_radius,
                    fill=True,
                    color=color,
                    popup=popup,
                ).add_to(m)
    ####################################################

    print('     > Plot points along the final track ..')
    # marker_cluster3 = MarkerCluster(name='Track')
    # marker_cluster3.add_to(m)

    for point in points:
        if 'FLDATELBL' in points[0].keys():  # Newer storms have this information
            date = point[date_key]
            hclass = point['TCDVLP']
            popup = """
                <div style="width: 200px; height: 90px" </div>
                <h5> {} condition</h5> <br>
                'Date:      {} <br> 
                 Condition: {} <br>
                """.format(
                name, date, hclass
            )

            if 'tropical' in hclass.lower():
                hclass = 'tropical depression'

            color = colors_hurricane_condition[hclass.lower()]
        else:
            popup = '{}<br>{}'.format(name, year)
            color = colors_hurricane_condition['hurricane']

        location = point['LAT'], point['LON']
        folium.CircleMarker(
            location=location, radius=track_radius, fill=True, color=color, popup=popup,
        ).add_to(m)

    # m = make_map(bbox)
    ####################################################
print('     > Add disclaimer and storm name ..')
noaa_logo = 'https://www.nauticalcharts.noaa.gov/images/noaa-logo-no-ring-70.png'
# FloatImage(noaa_logo, bottom=90, left=5).add_to(m)    #in percent

if STORM is not None:
    storm_info_html = """
                <div style="position: fixed; 
                            bottom: 50px; left: 5px; width: 140px; height: 45px; 
                            border:2px solid grey; z-index:9999; font-size:14px;background-color: lightgray;opacity: 0.9;
                            ">&nbsp; Storm: {} <br>
                              &nbsp; Year:  {}  &nbsp; <br>
                </div>
                """.format(
        name, year
    )

    m.get_root().html.add_child(folium.Element(storm_info_html))
###############################################
fct_info_html = """
            <div style="position: fixed; 
                        bottom: 100px; left: 5px; width: 170px; height: 45px; 
                        border:2px solid grey; z-index:9999; font-size:14px;background-color: lightgray;opacity: 0.9;
                        ">&nbsp; Date:  {}UTC <br>
                          &nbsp; FCT : t{}z  &nbsp; <br>
            </div>
            """.format(
    Date, FCT
)

m.get_root().html.add_child(folium.Element(fct_info_html))

############################################################################
Disclaimer_html = """
                <div style="position: fixed; 
                            bottom: 5px; left: 250px; width: 520px; height: px; 
                            border:2px solid grey; z-index:9999; font-size:12px; background-color: lightblue;opacity: 0.6;
                            ">&nbsp; Hurricane Explorer;  
                            <a href="https://nauticalcharts.noaa.gov/" target="_blank" >         NOAA/NOS/OCS</a> <br>
                              &nbsp; Contact: Saeed.Moghimi@noaa.gov &nbsp; <br>
                              &nbsp; Disclaimer: Experimental product. All configurations and results are pre-decisional.<br>
                </div>
  
                """

m.get_root().html.add_child(folium.Element(Disclaimer_html))
###################################################
folium.LayerControl().add_to(m)
MousePosition().add_to(m)

print('     > Save file ...')

out_dir = os.path.join(RUNdir, 'html_out', 'data')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fname = os.path.join(out_dir, '{}_{}_{}.html'.format(ModelDomain, Date, FCT))
#print(fname)
m.save(fname)
