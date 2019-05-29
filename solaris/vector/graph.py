import os
import numpy as np
import geopandas as gpd
from ..utils.geo import get_subgraph
import shapely
from shapely.geometry import Point, LineString
import networkx as nx
import rasterio as rio
import fiona
import pickle
from multiprocessing import Pool


class Node(object):
    """An object to hold node attributes.

    Attributes
    ----------
    idx : int
        The numerical index of the node. Used as a unique identifier
        when the nodes are added to the graph.
    x : `int` or `float`
        Numeric x location of the node, in either a geographic CRS or in pixel
        coordinates.
    y : `int` or `float`
        Numeric y location of the node, in either a geographic CRS or in pixel
        coordinates.

    """

    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Node {} at ({}, {})'.format(self.idx, self.x, self.y)


class Edge(object):
    """An object to hold edge attributes.

    Attributes
    ----------
    nodes : 2-`tuple` of :class:`Node` s
        :class:`Node` instances connected by the edge.
    weight : int or float
        The weight of the edge.

    """

    def __init__(self, nodes, edge_weight=None):
        self.nodes = nodes
        self.weight = edge_weight

    def __repr__(self):
        return 'Edge between {} and {} with weight {}'.format(self.nodes[0],
                                                              self.nodes[1],
                                                              self.weight)

    def set_edge_weight(self, normalize_factor=None, inverse=False):
        """Get the edge weight based on Euclidean distance between nodes.

        Note
        ----
        This method does not account for spherical deformation (i.e. does not
        use the Haversine equation). It is a simple linear distance.

        Arguments
        ---------
        normalize_factor : `int` or `float`, optional
            a number to multiply (or divide, if
            ``inverse=True``) the Euclidean distance by. Defaults to ``None``
            (no normalization)
        inverse : bool, optional
            if ``True``, the Euclidean distance weight will be divided by
            ``normalize_factor`` instead of multiplied by it.
        """
        weight = np.linalg.norm(
            np.array((self.nodes[0].x, self.nodes[0].y)) -
            np.array((self.nodes[1].x, self.nodes[1].y)))

        if normalize_factor is not None:
            if inverse:
                weight = weight/normalize_factor
            else:
                weight = weight*normalize_factor
        self.weight = weight

    def get_node_idxs(self):
        """Return the Node.idx for the nodes in the edge."""
        return (self.nodes[0].idx, self.nodes[1].idx)


class Path(object):
    """An object to hold :class:`Edge` s with common properties.

    Attributes
    ----------
    edges : `list` of :class:`Edge` s
        A `list` of :class:`Edge` s
    properties : dict
        A dictionary of property: value pairs that provide relevant metadata
        about edges along the path (e.g. road type, speed limit, etc.)

    """

    def __init__(self, edges=None, properties=None):
        self.edges = edges
        if properties is None:
            properties = {}
        self.properties = properties

    def __repr__(self):
        return 'Path including {}'.format([e for e in self.edges])

    def add_edge(self, edge):
        """Add an edge to the path."""
        self.edges.append(edge)

    def set_edge_weights(self, data_key=None, inverse=False, overwrite=True):
        """Calculate edge weights for all edges in the Path."""
        for edge in self.edges:
            if not overwrite and edge.weight is not None:
                continue
            if data_key is not None:
                edge.set_edge_weight(
                    normalize_factor=self.properties[data_key],
                    inverse=inverse)
            else:
                edge.set_edge_weight()

    def add_data(self, property, value):
        """Add a property: value pair to the Path.properties attribute."""
        self.properties[property] = value

    def __iter__(self):
        """Iterate through edges in the path."""
        yield from self.edges


def geojson_to_graph(geojson, graph_name=None, retain_all=True,
                     valid_road_types=None, road_type_field='type', edge_idx=0,
                     first_node_idx=0, weight_norm_field=None, inverse=False,
                     workers=1, verbose=False, output_path=None):
    """Convert a geojson of path strings to a network graph.

    Arguments
    ---------
    geojson : str
        Path to a geojson file (or any other OGR-compatible vector file) to
        load network edges and nodes from.
    graph_name : str, optional
        Name of the graph. If not provided, graph will be named ``'unnamed'`` .
    retain_all : bool, optional
        If ``True`` , the entire graph will be returned even if some parts are
        not connected. Defaults to ``True``.
    valid_road_types : :class:`list` of :class:`int` s, optional
        The road types to permit in the graph. If not provided, it's assumed
        that all road types are permitted. The possible values are integers
        ``1``-``7``, which map as follows::

            1: Motorway
            2: Primary
            3: Secondary
            4: Tertiary
            5: Residential
            6: Unclassified
            7: Cart track

    road_type_field : str, optional
        The name of the property in the vector data that delineates road type.
        Defaults to ``'type'`` .
    edge_idx : int, optional
        The first index to use for an edge. This can be set to a higher value
        so that a graph's edge indices don't overlap with existing values in
        another graph.
    first_node_idx : int, optional
        The first index to use for a node. This can be set to a higher value
        so that a graph's node indices don't overlap with existing values in
        another graph.
    weight_norm_field : str, optional
        The name of a field in `geojson` to pass to argument ``data_key`` in
        :func:`Path.set_edge_weights`. Defaults to ``None``, in which case
        no weighting is performed (weights calculated solely using Euclidean
        distance.)
    workers : int, optional
        Number of parallel processes to run for parallelization. Defaults to 1.
        Should not be greater than the number of CPUs available.
    verbose : bool, optional
        Verbose print output. Defaults to ``False`` .
    output_path : str, optional
        Path to a pickle file to save the output graph to. Nothing will be
        saved to disk if not provided.

    Returns
    -------
    G : :class:`networkx.MultiDiGraph`
        A :class:`networkx.MultiDiGraph` containing all of the nodes and edges
        from the geojson (or only the largest connected component if
        `retain_all` = ``False``). Edge lengths are weighted based on
        geographic distance.

    """
    with fiona.open(geojson, 'r') as f:
        crs = f.crs
        f.close()
    # due to an annoying feature of loading these graphs, the numeric road
    # type identifiers are presented as string versions. we therefore reformat
    # the valid_road_types list as strings.
    if valid_road_types is not None:
        valid_road_types = [str(i) for i in valid_road_types]

    # create the graph as a MultiGraph and set the original CRS to EPSG 4326

    # extract nodes and paths
    nodes, paths = get_nodes_paths(geojson,
                                   valid_road_types=valid_road_types,
                                   first_node_idx=first_node_idx,
                                   road_type_field=road_type_field,
                                   workers=workers, verbose=verbose)
    # nodes is a dict of node_idx: node_params (e.g. location, metadata)
    # pairs.
    # paths is a dict of path dicts. the path key is the path_idx.
    # each path dict has a list of node_idxs as well as properties metadata.

    # initialize the graph object
    G = nx.MultiDiGraph(name=graph_name, crs=crs)
    if not nodes:  # if there are no nodes in the graph
        return G
    if verbose:
        print("nodes:", nodes)
        print("paths:", paths)
    # add each osm node to the graph
    for node in nodes:
        G.add_node(node.idx, **{'x': node.x, 'y': node.y})
    # add each path to the graph
    for path in paths:
        # calculate edge length using euclidean distance and a weighting term
        path.set_edge_weights(data_key=weight_norm_field, inverse=inverse)
        edges = [(*[node.idx for node in edge.nodes],
                  edge.weight) for edge in path]
        if verbose:
            print(edges)
        G.add_weighted_edges_from(edges)
    if not retain_all:
        # keep only largest connected component of graph unless retain_all
        # code modified from osmnx.core.get_largest_component & induce_subgraph
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = get_subgraph(G, largest_cc)

    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
            f.close()

    return G


def get_nodes_paths(vector_file, first_node_idx=0, node_gdf=gpd.GeoDataFrame(),
                    valid_road_types=None, road_type_field='type', workers=1,
                    verbose=False):
    """
    Extract nodes and paths from a vector file.

    Arguments
    ---------
    vector_file : str
        Path to an OGR-compatible vector file containing line segments (e.g.,
        JSON response from from the Overpass API, or a SpaceNet GeoJSON).
    first_path_idx : int, optional
        The first index to use for a path. This can be set to a higher value
        so that a graph's path indices don't overlap with existing values in
        another graph.
    first_node_idx : int, optional
        The first index to use for a node. This can be set to a higher value
        so that a graph's node indices don't overlap with existing values in
        another graph.
    node_gdf : :class:`geopandas.GeoDataFrame` , optional
        A :class:`geopandas.GeoDataFrame` containing nodes to add to the graph.
        New nodes will be added to this object incrementally during the
        function call.
    valid_road_types : :class:`list` of :class:`int` s, optional
        The road types to permit in the graph. If not provided, it's assumed
        that all road types are permitted. The possible values are integers
        ``1``-``7``, which map as follows::

            1: Motorway
            2: Primary
            3: Secondary
            4: Tertiary
            5: Residential
            6: Unclassified
            7: Cart track

    road_type_field : str, optional
        The name of the attribute containing road type information in
        `vector_file`. Defaults to ``'type'``.
    workers : int, optional
        Number of worker processes to use for parallelization. Defaults to 1.
        Should not exceed the number of CPUs available.
    verbose : bool, optional
        Verbose print output. Defaults to ``False``.

    Returns
    -------
    nodes, paths : `tuple` of `dict` s
        nodes : list
            A `list` of :class:`Node` s to be added to the graph.
        paths : list
            A list of :class:`Path` s containing the :class:`Edge` s and
            :class:`Node` s to be added to the graph.

    """
    if valid_road_types is None:
        valid_road_types = ['1', '2', '3', '4', '5', '6', '7']

    with fiona.open(vector_file, 'r') as source:

        with Pool(processes=workers) as pool:
            node_list = pool.map(_get_all_nodes, source,
                                 chunksize=10)
            pool.close()
        source.close()

    # convert to geoseries and drop duplicates (have to flatten first)
    node_series = gpd.GeoSeries([i for sublist in node_list for i in sublist])
    # NOTE: It is ESSENTIAL to use keep='last' in the line below; otherwise, it
    # misses a duplicate if it includes the first element of the series.
    node_series = node_series.drop_duplicates(keep='last')
    node_series = node_series.reset_index(drop=True)
    node_series.name = 'geometry'
    node_series.index.name = 'node_idx'
    node_gdf = gpd.GeoDataFrame(node_series.reset_index())
    node_gdf['node'] = node_gdf.apply(
        lambda p: Node(p['node_idx'], p['geometry'].x, p['geometry'].y),
        axis=1)

    # create another parallelized operation to iterate through edges
    # _init_worker passes the node_series to every process in the pool
    with fiona.open(vector_file, 'r') as source:
        with Pool(
                processes=workers, initializer=_init_worker,
                initargs=(node_gdf, valid_road_types, road_type_field)
                ) as pool:
            zipped_edges_properties = pool.map(parallel_linestring_to_path,
                                               source, chunksize=10)
        pool.close()
    source.close()

    nodes = node_gdf['node'].tolist()
    paths = []
    # it would've been better to do this within the multiprocessing pool but
    # it's REALLY hard to share objects in memory across processes without
    # copies being made (and therefore nodes being duplicated)
    for edges, properties in zipped_edges_properties:
        path = Path(
            edges=[Edge((nodes[edge[0]], nodes[edge[1]])) for edge in edges],
            properties=properties
            )
        paths.append(path)
    return nodes, paths


def parallel_linestring_to_path(feature):
    """Read in a feature line from a fiona-opened shapefile and get the edges.

    Arguments
    ---------
    feature : dict
        An item from a :class:`fiona.open` iterable with the key ``'geometry'``
        containing :class:`shapely.geometry.line.LineString` s or
        :class:`shapely.geometry.line.MultiLineString` s.

    Returns
    -------
    A list of :class:`Path` s containing all edges in the LineString or
    MultiLineString.

    Notes
    -----
    This function depends on ``node_series`` and ``valid_road_types``, which
    are passed by an initializer as items in ``var_dict``.

    """

    properties = feature['properties']
    # TODO: create more adjustable filter
    if var_dict['road_type_field'] in properties:
        road_type = properties[var_dict['road_type_field']]
    elif 'highway' in properties:
        road_type = properties['highway']
    elif 'road_type' in properties:
        road_type = properties['road_type']
    else:
        road_type = 'None'

    geom = feature['geometry']
    if geom['type'] == 'LineString' or \
            geom['type'] == 'MultiLineString':
        if road_type not in var_dict['valid_road_types'] or \
                'LINESTRING EMPTY' in properties.values():
            return

    if geom['type'] == 'LineString':
        linestring = shapely.geometry.shape(geom)
        edges = linestring_to_edges(linestring, var_dict['node_gdf'])

    elif geom['type'] == 'MultiLineString':
        # do the same thing as above, but do it for each piece
        edges = []
        for linestring in shapely.geometry.shape(geom):
            edge_set, node_idx, node_gdf = linestring_to_edges(
                linestring, var_dict['node_gdf'])
            edges.extend(edge_set)

    return edges, properties


def linestring_to_edges(linestring, node_gdf):
    """Collect nodes in a linestring and add them to an edge.

    Arguments
    ---------
    linestring : :class:`shapely.geometry.LineString`
        A :class:`shapely.geometry.LineString` object to extract nodes and
        edges from.
    node_series : :class:`geopandas.GeoSeries`
        A :class:`geopandas.GeoSeries` containing a
        :class:`shapely.geometry.point.Point` for every node to be added to the
        graph.

    Returns
    -------
    edges : list
        A list of :class:`Edge` s from ``linestring``.

    """
    edges = []
    nodes = []

    for point in linestring.coords:
        point_shp = shapely.geometry.shape(Point(point))
        nodes.append(
            node_gdf.node_idx[node_gdf.distance(point_shp) == 0.0].values[0]
            )
        if len(nodes) > 1:
            edges.append(nodes[-2:])

    return edges


def graph_to_geojson(G, output_path, encoding='utf-8', overwrite=False,
                     verbose=False):
    """
    Save graph to two geojsons: one containing nodes, the other edges.
    Arguments
    ---------
    G : :class:`networkx.MultiDiGraph`
        A graph object to save to geojson files.
    output_path : str
        Path to save the geojsons to. ``'_nodes.geojson'`` and
        ``'_edges.geojson'`` will be appended to ``output_path`` (after
        stripping the extension).
    encoding : str, optional
        The character encoding for the saved files.
    overwrite : bool, optional
        Should files at ``output_path`` be overwritten? Defaults to no
        (``False``).
    verbose : bool, optional
        Switch to print relevant values.  Defaults to no (``False``).

    Notes
    -----
    This function is based on ``osmnx.save_load.save_graph_shapefile``, with
    tweaks to make it work with our graph objects. It will save two geojsons:
    a file containing all of the nodes and a file containing all of the edges.
    When writing to geojson, must convert the coordinate reference system
    (crs) to string if it's a dict, otherwise no crs will be appended to the
    geojson.

    Returns
    -------
    None
    """

    # convert directed graph G to an undirected graph for saving as a shapefile
    G_to_save = G.copy().to_undirected()
    # create GeoDataFrame containing all of the nodes
    nodes, data = zip(*G_to_save.nodes(data=True))
    gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes)

    # get coordinate reference system
    g_crs = G_to_save.graph['crs']
    if type(g_crs) == dict:
        # convert from dict
        g_crs = rio.crs.CRS.from_dict(g_crs)
    gdf_nodes.crs = g_crs
    if verbose:
        print("crs:", g_crs)

    gdf_nodes['geometry'] = gdf_nodes.apply(
        lambda row: Point(row['x'], row['y']), axis=1
        )
    gdf_nodes = gdf_nodes.drop(['x', 'y'], axis=1)
    # gdf_nodes['node_idx'] = gdf_nodes['node_idx'].astype(np.int32)

    # # make everything but geometry column a string
    # for col in [c for c in gdf_nodes.columns if not c == 'geometry']:
    #    gdf_nodes[col] = gdf_nodes[col].fillna('').map(make_str)

    # create GeoDataFrame containing all of the edges
    edges = []
    for u, v, key, data in G_to_save.edges(keys=True, data=True):
        edge = {'key': key}
        for attr_key in data:
            edge[attr_key] = data[attr_key]
        if 'geometry' not in data:
            point_u = Point((G_to_save.nodes[u]['x'], G_to_save.nodes[u]['y']))
            point_v = Point((G_to_save.nodes[v]['x'], G_to_save.nodes[v]['y']))
            edge['geometry'] = LineString([point_u, point_v])
        edges.append(edge)

    gdf_edges = gpd.GeoDataFrame(edges)
    gdf_edges.crs = g_crs

    for col in [c for c in gdf_nodes.columns if c != 'geometry']:
        gdf_nodes[col] = gdf_nodes[col].fillna('').apply(str)
    for col in [c for c in gdf_edges.columns if c != 'geometry']:
        gdf_edges[col] = gdf_edges[col].fillna('').apply(str)

    # make directory structure
    if not os.path.exists(os.path.split(output_path)[0]):
        os.makedirs(os.path.split(output_path)[0])

    edges_path = os.path.splitext(output_path)[0] + '_edges.geojson'
    nodes_path = os.path.splitext(output_path)[0] + '_nodes.geojson'
    if overwrite:
        if os.path.exists(edges_path):
            os.remove(edges_path)
        if os.path.exists(nodes_path):
            os.remove(nodes_path)

    gdf_edges.to_file(edges_path, encoding=encoding, driver='GeoJSON')
    gdf_nodes.to_file(nodes_path, encoding=encoding, driver='GeoJSON')


def _get_all_nodes(feature):
    """Create a list of node geometries from a geojson of (multi)linestrings.

    Note
    ----
    This function is intended to be used with pool.imap_unordered for
    parallelization.

    Returns
    -------
    A list of :class:`shapely.geometry.Point` instances. DUPLICATES CAN EXIST.
    """
    points = []
    geom = feature['geometry']
    if geom['type'] == 'LineString':
        linestring = shapely.geometry.shape(geom)
        points.extend(_get_linestring_points(linestring))
    elif geom['type'] == 'MultiLineString':
        for linestring in shapely.geometry.shape(geom):
            points.extend(_get_linestring_points(linestring))

    return points


def _get_linestring_points(linestring):
    points = []
    for point in linestring.coords:
        points.append(shapely.geometry.shape(Point(point)))
    return points


def _init_worker(node_gdf, valid_road_types, road_type_field):
    the_dict = {'node_gdf': node_gdf,
                'valid_road_types': valid_road_types,
                'road_type_field': road_type_field}
    global var_dict
    var_dict = the_dict
