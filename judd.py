import os
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# def king_county_csv_loader(data_dir,output_file): # no need for an output file!
def king_county_csv_loader(data_dir): 
    '''
    Load malformed data from the King County website
    '''
    
    dfs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and filename.startswith("Hydrology"):
            print(filename)            
            filepath = os.path.join(data_dir, filename)

            with open(filepath) as f:
                lines = [line.rstrip(',\n') for line in f]

            # Split all lines into lists of values
            split_lines = [line.split(',') for line in lines]

            # Drop the last column from the header (which is just a description)
            header = split_lines[0][:3] + ['Flag1', 'Flag2', 'Flag3']
            data_lines = split_lines[1:]

            # Pad all rows to 6 columns
            max_cols = 6
            padded_rows = [row + [''] * (max_cols - len(row)) for row in data_lines]

            # Create dataframe
            df = pd.DataFrame(padded_rows, columns=header)
            dfs.append(df)
            print(len(dfs))

    # Merge all files
    merged_df = pd.concat(dfs, ignore_index=True)
#     merged_df.to_csv(output_file, index=False)

#     print(f"✅ Merged {len(dfs)} files into {output_file}")


    # Convert date column to datetime
    merged_df['Collect Date (local)'] = pd.to_datetime(merged_df['Collect Date (local)'], format="%m/%Y", errors='coerce')
    merged_df['Precipitation (inches)'] = pd.to_numeric(merged_df['Precipitation (inches)'], errors='coerce')
    
    return merged_df


import numpy as np
from shapely.geometry import Polygon, LineString
from scipy.spatial import Voronoi
import geopandas as gpd

def make_voronoi_gdf(merged_2019, geojson_path="data/map.geojson"):
    """
    Build a clipped Voronoi GeoDataFrame from site locations and precipitation data.
    
    Parameters:
        merged_2019: pd.DataFrame with columns ['x', 'y', 'Site_Code', 'Avg Precip (inches)']
        geojson_path: path to boundary GeoJSON file (default: 'data/map.geojson')
    
    Returns:
        GeoDataFrame with Voronoi polygons clipped to the boundary and area columns.
    """
    from shapely.geometry import Polygon
    from scipy.spatial import Voronoi

    # Extract inputs
    points = merged_2019[['x', 'y']].values
    site_codes = merged_2019['Site_Code'].values
    precips = merged_2019['Avg Precip (inches)'].values

    # Voronoi computation
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Load and project boundary
    boundary = gpd.read_file(geojson_path).to_crs('EPSG:2926')
    boundary_poly = boundary.unary_union

    # Clip and store polygons
    polygons, codes, vals = [], [], []
    for i, region in enumerate(regions):
        poly = Polygon(vertices[region])
        if not poly.is_valid:
            continue
        clipped = poly.intersection(boundary_poly)
        if not clipped.is_empty:
            polygons.append(clipped)
            codes.append(site_codes[i])
            vals.append(precips[i])

    # Build GeoDataFrame
    gdf_voronoi = gpd.GeoDataFrame({
        'Site_Code': codes,
        'Avg Precip (inches)': vals,
        'geometry': polygons
    }, crs='EPSG:2926')

    # Compute area
    gdf_voronoi['Area (m²)'] = gdf_voronoi.geometry.area
    gdf_voronoi['Area (km²)'] = gdf_voronoi['Area (m²)'] / 1e6

    return gdf_voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite polygons.
    Source: https://gist.github.com/pv/8036995
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 100  # Big enough to enclose outer areas

    # Construct a map from ridge points to ridge vertices
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Infinite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue  # finite ridge

            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # Reorder region's vertices
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)
