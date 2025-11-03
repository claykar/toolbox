#################################################
# Clay Karnis
##############
# this should be a starting point for smoothing ridged line and polygon features
# this right now is not geodata, but np arrays
#################################################
'''
#################################################
#################################################
# explain that uses arrays
import scipy.interpolate as si
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([0.0, 0.0, 4.5, 4.5,
               0.3, 1.5, 2.3, 3.8, 3.7, 2.3,
               1.5, 2.2, 2.8, 2.2,
               2.1, 2.2, 2.3])
ys = np.array([0.0, 3.0, 3.0, 0.0,
               1.1, 2.3, 2.5, 2.3, 1.1, 0.5,
               1.1, 2.1, 1.1, 0.8,
               1.1, 1.3, 1.1])
zs = np.array([0,   0,   0,   0,
               1,   1,   1,   1,   1,   1,
               2,   2,   2,   2,
               3,   3,   3])
pts = np.array([xs, ys]).transpose()

# set up a grid for us to resample onto
nx, ny = (100, 100)
xrange = np.linspace(np.min(xs[zs!=0])-0.1, np.max(xs[zs!=0])+0.1, nx)
yrange = np.linspace(np.min(ys[zs!=0])-0.1, np.max(ys[zs!=0])+0.1, ny)
xv, yv = np.meshgrid(xrange, yrange)
ptv = np.array([xv, yv]).transpose()

# interpolate over the grid
out = si.griddata(pts, zs, ptv, method='cubic').transpose()

def close(vals):
    return np.concatenate((vals, [vals[0]]))

# plot the results
levels = [1, 2, 3]
plt.plot(close(xs[zs==1]), close(ys[zs==1]))
plt.plot(close(xs[zs==2]), close(ys[zs==2]))
plt.plot(close(xs[zs==3]), close(ys[zs==3]))
plt.contour(xrange, yrange, out, levels)
plt.show()

###################################################
###################################################
###################################################

import geopandas as gpd

gdf = gpd.read_file("input.shp")

# assume polygons or lines
geoms = gdf.geometry
import numpy as np
from shapely.geometry import Polygon, LineString

def geom_to_xy(geom):
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
    elif geom.geom_type == "LineString":
        x, y = geom.xy
    else:
        raise ValueError("Must be Polygon/LineString")
    return np.array(x), np.array(y)


from scipy.interpolate import splprep, splev

def smooth_xy(x, y, smoothness=0.001):
    # parametric B-spline fit
    pts = np.vstack([x, y])
    tck, u = splprep(pts, s=smoothness)
    # dense sampling
    unew = np.linspace(0, 1, 200)
    out = splev(unew, tck)
    return np.array(out[0]), np.array(out[1])


def smooth_geom(geom):
    x, y = geom_to_xy(geom)
    xs, ys = smooth_xy(x, y)

    if geom.geom_type == "Polygon":
        new_geom = Polygon(zip(xs, ys))
    else:
        new_geom = LineString(zip(xs, ys))
    return new_geom

gdf["geometry"] = gdf.geometry.apply(smooth_geom)
gdf.to_file("smoothed.shp")

########################################################
########################################################
########################################################

import numpy as np
from shapely.ops import unary_union, polygonize
import geopandas as gpd
from scipy.interpolate import griddata
from skimage import measure

# --- 1) Extract points + z values
xs = []
ys = []
zs = []
for idx, row in gdf.iterrows():
    geom = row.geometry
    z = row["z"]   # attribute value
    x, y = geom.exterior.xy
    xs.extend(x)
    ys.extend(y)
    zs.extend([z]*len(x))

pts = np.vstack([xs, ys]).T

# --- 2) Make grid
xrange = np.linspace(min(xs), max(xs), 100)
yrange = np.linspace(min(ys), max(ys), 100)
xv, yv = np.meshgrid(xrange, yrange)
grid = griddata(pts, zs, (xv, yv), method="cubic")

# --- 3) Create contour
levels = [1,2,3]
contours = measure.find_contours(grid, level=1)

polys = []
for c in contours:
    cxy = np.column_stack([
        np.interp(c[:, 1], np.arange(len(xrange)), xrange),
        np.interp(c[:, 0], np.arange(len(yrange)), yrange),
    ])
    polys.append(Polygon(cxy))

gdf_out = gpd.GeoDataFrame(geometry=polys)
gdf_out.to_file("contours.shp")

################################################
################################################
################################################
# chaikin
from shapely.geometry import LineString

def chaikin(coords, refinements=2):
    for _ in range(refinements):
        new = []
        for i in range(len(coords) - 1):
            p = coords[i]
            q = coords[i+1]
            new.append((0.75*p[0] + 0.25*q[0], 0.75*p[1] + 0.25*q[1]))
            new.append((0.25*p[0] + 0.75*q[0], 0.25*p[1] + 0.75*q[1]))
        coords = new
    return coords
################################################
'''