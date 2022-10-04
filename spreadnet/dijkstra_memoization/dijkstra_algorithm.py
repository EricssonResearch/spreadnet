"""
Shortest path algorithms for weighted graphs.
"""

from collections import deque
from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors

__all__ = [
    "dijkstra_path",
    "dijkstra_path_length",
    "bidirectional_dijkstra",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
]


def _weight_function(G, weight):
    """Returns a function that returns the weight of an edge.

    The returned function is specifically suitable for input to
    functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.

    Parameters
    ----------
    G : NetworkX graph.

    weight : string or function
        If it is callable, `weight` itself is returned. If it is a string,
        it is assumed to be the name of the edge attribute that represents
        the weight of an edge. In that case, a function is returned that
        gets the edge weight according to the specified edge attribute.

    Returns
    -------
    function
        This function returns a callable that accepts exactly three inputs:
        a node, an node adjacent to the first one, and the edge attribute
        dictionary for the eedge joining those nodes. That function returns
        a number representing the weight of an edge.

    If `G` is a multigraph, and `weight` is not callable, the
    minimum edge weight over all parallel edges is returned. If any edge
    does not have an attribute with key `weight`, it is assumed to
    have weight one.

    """
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def dijkstra_path(G, source, target, weight="weight"):
    """Returns the shortest weighted path from source to target in G.

    Uses Dijkstra's Method to compute the shortest weighted path
    between two nodes in a graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node

    target : node
        Ending node

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    path : list
        List of nodes in a shortest path.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.dijkstra_path(G, 0, 4))
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    The weight function can be used to include node weights.

    >>> def func(u, v, d):
    ...     node_u_wt = G.nodes[u].get("node_weight", 1)
    ...     node_v_wt = G.nodes[v].get("node_weight", 1)
    ...     edge_wt = d.get("weight", 1)
    ...     return node_u_wt / 2 + node_v_wt / 2 + edge_wt

    In this example we take the average of start and end node
    weights of an edge and add it to the weight of the edge.

    The function :func:`single_source_dijkstra` computes both
    path and length-of-path if you need both, use that.
    """
    (length, path) = single_source_dijkstra(G, source, target=target, weight=weight)
    return path


def dijkstra_path_length(G, source, target, weight="weight"):
    """Returns the shortest weighted path length in G from source to target.

    Uses Dijkstra's Method to compute the shortest weighted path length
    between two nodes in a graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        starting node for path

    target : node label
        ending node for path

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    length : number
        Shortest path length.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.dijkstra_path_length(G, 0, 4)
    4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    The function :func:`single_source_dijkstra` computes both
    path and length-of-path if you need both, use that.

    See Also
    --------
    bidirectional_dijkstra
    bellman_ford_path_length
    single_source_dijkstra

    """
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} not found in graph")
    if source == target:
        return 0
    weight = _weight_function(G, weight)
    length = _dijkstra(G, source, weight, target=target)
    try:
        return length[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}") from err


def single_source_dijkstra_path(G, source, cutoff=None, weight="weight"):
    """Find shortest weighted paths in G from a source node.

    Compute shortest path between source and all other reachable
    nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for path.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    paths : dictionary
        Dictionary of shortest path lengths keyed by target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.single_source_dijkstra_path(G, 0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    single_source_dijkstra, single_source_bellman_ford

    """
    return multi_source_dijkstra_path(G, {source}, cutoff=cutoff, weight=weight)


def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    """Find shortest weighted path lengths in G from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    length : dict
        Dict keyed by node to shortest path length from source.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = nx.single_source_dijkstra_path_length(G, 0)
    >>> length[4]
    4
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    single_source_dijkstra, single_source_bellman_ford_path_length

    """
    return multi_source_dijkstra_path_length(G, {source}, cutoff=cutoff, weight=weight)


def single_source_dijkstra(G, source, target=None, cutoff=None, weight="weight"):
    """Find shortest weighted paths and lengths from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Uses Dijkstra's algorithm to compute shortest paths and lengths
    between a source and all other reachable nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    target : node label, optional
        Ending node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.


    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list.
        If target is None, paths and lengths to all nodes are computed.
        The return value is a tuple of two dictionaries keyed by target nodes.
        The first dictionary stores distance to each target node.
        The second stores the path to each target node.
        If target is not None, returns a tuple (distance, path), where
        distance is the distance from source to target and path is a list
        representing the path from source to target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.single_source_dijkstra(G, 0)
    >>> length[4]
    4
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4
    >>> path[4]
    [0, 1, 2, 3, 4]
    >>> length, path = nx.single_source_dijkstra(G, 0, 1)
    >>> length
    1
    >>> path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    https://code.activestate.com/recipes/119466/

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path
    single_source_dijkstra_path_length
    single_source_bellman_ford
    """
    return multi_source_dijkstra(
        G, {source}, cutoff=cutoff, target=target, weight=weight
    )


def multi_source_dijkstra_path(G, sources, cutoff=None, weight="weight"):
    """Find shortest weighted paths in G from a given set of source
    nodes.

    Compute shortest path between any of the source nodes and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    paths : dictionary
        Dictionary of shortest paths keyed by target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.multi_source_dijkstra_path(G, {0, 4})
    >>> path[1]
    [0, 1]
    >>> path[3]
    [4, 3]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    multi_source_dijkstra, multi_source_bellman_ford

    """
    length, path = multi_source_dijkstra(G, sources, cutoff=cutoff, weight=weight)
    return path


def multi_source_dijkstra_path_length(G, sources, cutoff=None, weight="weight"):
    """Find shortest weighted path lengths in G from a given set of
    source nodes.

    Compute the shortest path length between any of the source nodes and
    all other reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    length : dict
        Dict keyed by node to shortest path length to nearest source.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = nx.multi_source_dijkstra_path_length(G, {0, 4})
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 1
    4: 0

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    multi_source_dijkstra

    """
    if not sources:
        raise ValueError("sources must not be empty")
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    weight = _weight_function(G, weight)
    return _dijkstra_multisource(G, sources, weight, cutoff=cutoff)


def multi_source_dijkstra(G, sources, target=None, cutoff=None, weight="weight"):
    """Find shortest weighted paths and lengths from a given set of
    source nodes.

    Uses Dijkstra's algorithm to compute the shortest paths and lengths
    between one of the source nodes and the given `target`, or all other
    reachable nodes if not specified, for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
        single node, then all paths computed by this function will start
        from that node. If there are two or more nodes in the set, the
        computed paths may begin from any one of the start nodes.

    target : node label, optional
        Ending node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list
        If target is None, returns a tuple of two dictionaries keyed by node.
        The first dictionary stores distance from one of the source nodes.
        The second stores the path from one of the sources to that node.
        If target is not None, returns a tuple of (distance, path) where
        distance is the distance from source to target and path is a list
        representing the path from source to target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.multi_source_dijkstra(G, {0, 4})
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 1
    4: 0
    >>> path[1]
    [0, 1]
    >>> path[3]
    [4, 3]

    >>> length, path = nx.multi_source_dijkstra(G, {0, 4}, 1)
    >>> length
    1
    >>> path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    https://code.activestate.com/recipes/119466/

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    Raises
    ------
    ValueError
        If `sources` is empty.
    NodeNotFound
        If any of `sources` is not in `G`.

    See Also
    --------
    multi_source_dijkstra_path
    multi_source_dijkstra_path_length

    """
    if not sources:
        raise ValueError("sources must not be empty")
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    if target in sources:
        return (0, [target])
    weight = _weight_function(G, weight)
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _dijkstra_multisource(
        G, sources, weight, paths=paths, cutoff=cutoff, target=target
    )
    #TODO: Change this. just return entire dictionary of paths
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError as err:
        raise nx.NetworkXNoPath(f"No path to {target}.") from err


def _dijkstra(G, source, weight, pred=None, paths=None, cutoff=None, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths from a
    single source.

    This is a convenience function for :func:`_dijkstra_multisource`
    with all the arguments the same, except the keyword argument
    `sources` set to ``[source]``.

    """
    return _dijkstra_multisource(
        G, [source], weight, pred=pred, paths=paths, cutoff=cutoff, target=target
    )


def _dijkstra_multisource(
    G, sources, weight, pred=None, paths=None, cutoff=None, target=None
):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.

    weight: function
        Function with (u, v, data) input that returns that edges weight

    pred: dict of lists, optional(default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored.

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node.
        If None, paths are not stored.

    target : node label, optional
        Ending node for path. Search is halted when target is found.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    Returns
    -------
    distance : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.

    Raises
    ------
    NodeNotFound
        If any of `sources` is not in `G`.

    Notes
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

    """
    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        seen[source] = 0
        push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            cost = weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
                elif pred is not None and vu_dist == u_dist:
                    pred[u].append(v)
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    #TODO add dist to memoization
    return dist


def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight="weight"):
    """Compute weighted shortest path length and predecessors.

    Uses Dijkstra's Method to obtain the shortest weighted paths
    and return dictionaries of predecessors for each node and
    distance for each node from the `source`.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    pred, distance : dictionaries
        Returns two dictionaries representing a list of predecessors
        of a node and the distance to each node.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The list of predecessors contains more than one element only when
    there are more than one shortest paths to the key node.

    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)
    >>> sorted(pred.items())
    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> pred, dist = nx.dijkstra_predecessor_and_distance(G, 0, 1)
    >>> sorted(pred.items())
    [(0, []), (1, [0])]
    >>> sorted(dist.items())
    [(0, 0), (1, 1)]
    """
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} is not found in the graph")
    weight = _weight_function(G, weight)
    pred = {source: []}  # dictionary of predecessors
    return (pred, _dijkstra(G, source, weight, pred=pred, cutoff=cutoff))


def all_pairs_dijkstra(G, cutoff=None, weight="weight"):
    """Find shortest weighted paths and lengths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edge[u][v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Yields
    ------
    (node, (distance, path)) : (node obj, (dict, dict))
        Each source node has two associated dicts. The first holds distance
        keyed by target and the second holds paths keyed by target.
        (See single_source_dijkstra for the source/target node terminology.)
        If desired you can apply `dict()` to this function to create a dict
        keyed by source node to the two dicts.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> len_path = dict(nx.all_pairs_dijkstra(G))
    >>> len_path[3][0][1]
    2
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"3 - {node}: {len_path[3][0][node]}")
    3 - 0: 3
    3 - 1: 2
    3 - 2: 1
    3 - 3: 0
    3 - 4: 1
    >>> len_path[3][1][1]
    [3, 2, 1]
    >>> for n, (dist, path) in nx.all_pairs_dijkstra(G):
    ...     print(path[1])
    [0, 1]
    [1]
    [2, 1]
    [3, 2, 1]
    [4, 3, 2, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The yielded dicts only have keys for reachable nodes.
    """
    for n in G:
        dist, path = single_source_dijkstra(G, n, cutoff=cutoff, weight=weight)
        yield (n, (dist, path))


def all_pairs_dijkstra_path_length(G, cutoff=None, weight="weight"):
    """Compute shortest path lengths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    distance : iterator
        (source, dictionary) iterator with dictionary keyed by target and
        shortest path length as the key value.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = dict(nx.all_pairs_dijkstra_path_length(G))
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"1 - {node}: {length[1][node]}")
    1 - 0: 1
    1 - 1: 0
    1 - 2: 1
    1 - 3: 2
    1 - 4: 3
    >>> length[3][2]
    1
    >>> length[2][2]
    0

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionary returned only has keys for reachable node pairs.
    """
    length = single_source_dijkstra_path_length
    for n in G:
        yield (n, length(G, n, cutoff=cutoff, weight=weight))


def all_pairs_dijkstra_path(G, cutoff=None, weight="weight"):
    """Compute shortest paths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    distance : dictionary
        Dictionary, keyed by source and target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = dict(nx.all_pairs_dijkstra_path(G))
    >>> path[0][4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    floyd_warshall, all_pairs_bellman_ford_path

    """
    path = single_source_dijkstra_path
    # TODO This can be trivially parallelized.
    for n in G:
        yield (n, path(G, n, cutoff=cutoff, weight=weight))


def bidirectional_dijkstra(G, source, target, weight="weight"):
    r"""Dijkstra's algorithm for shortest paths using bidirectional search.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node.

    target : node
        Ending node.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    length, path : number and list
        length is the distance from source to target.
        path is a list of nodes on a path from source to target.

    Raises
    ------
    NodeNotFound
        If either `source` or `target` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.bidirectional_dijkstra(G, 0, 4)
    >>> print(length)
    4
    >>> print(path)
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    In practice  bidirectional Dijkstra is much more than twice as fast as
    ordinary Dijkstra.

    Ordinary Dijkstra expands nodes in a sphere-like manner from the
    source. The radius of this sphere will eventually be the length
    of the shortest path. Bidirectional Dijkstra will expand nodes
    from both the source and the target, making two spheres of half
    this radius. Volume of the first sphere is `\pi*r*r` while the
    others are `2*\pi*r/2*r/2`, making up half the volume.

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    shortest_path
    shortest_path_length
    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if source == target:
        return (0, [source])

    weight = _weight_function(G, weight)
    push = heappush
    pop = heappop
    # Init:  [Forward, Backward]
    dists = [{}, {}]  # dictionary of final distances
    paths = [{source: [source]}, {target: [target]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) for choosing node to expand
    seen = [{source: 0}, {target: 0}]  # dict of distances to seen nodes
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    # neighs for extracting correct neighbor information
    if G.is_directed():
        neighs = [G._succ, G._pred]
    else:
        neighs = [G._adj, G._adj]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w, d in neighs[dir][v].items():
            # weight(v, w, d) for forward and weight(w, v, d) for back direction
            cost = weight(v, w, d) if dir == 0 else weight(w, v, d)
            if cost is None:
                continue
            vwLength = dists[dir][v] + cost
            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
