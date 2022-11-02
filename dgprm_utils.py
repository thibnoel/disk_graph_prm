import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import expon
from geometry_utilities.occ_env import *
from geometry_utilities.pointsSegUtils import *

def displayBubbles(graph, color='blue', edges_color='black', alpha=1, lw=2):
    """Displays the graph in matplotlib"""
    bubbles_color = color  # '#9ac3e8'
    #lw = 2

    ax = plt.gca()
    nodes_list = [graph.nodes[nid]["node_obj"] for nid in graph]
    edges_list = list(graph.edges)
    # nodes_pos = np.array([n.pos for n in nodes_list])
    circles_back = []
    circles_front = []
    # Nodes visualization
    for i, ni in enumerate(nodes_list):
        # plt.scatter(ni.pos[0]/scale, ni.pos[1]/scale, color=color, zorder=9)
        plt.plot(ni.pos[0], ni.pos[1], color=edges_color,
                 zorder=9, ls='', marker='.')

        bpos = ni.pos
        brad = ni.bubble_rad
        circ_back = plt.Circle(bpos, brad, color=[1,1,1,0], ec=edges_color, lw=lw)
        circ_front = plt.Circle(bpos, brad - lw, color=bubbles_color, alpha=alpha)
        circles_back.append(circ_back)
        circles_front.append(circ_front)
    for c in circles_back:
        ax.add_patch(c)
    for c in circles_front:
        ax.add_patch(c)
    # Edgs visualization
    for e in edges_list:
        ni = graph.nodes[e[0]]
        nj = graph.nodes[e[1]]
        line = np.array([ni["node_obj"].pos, nj["node_obj"].pos])
        plt.plot(line[:, 0], line[:, 1], color=edges_color, lw=lw)

def circularMask(envDim, center, radius):
    """Returns a circular boolean mask

    Args:
        envDim: list or array of size 2, describing the size of the grid environment
        center: list or array of size 2 representing the position of the circle center in the grid
        radius: float, radius of the wanted circular mask

    Returns:
        A 2D boolean-valued numpy array, with 1 inside the circle, 0 outside 
    """
    X, Y = np.ogrid[:envDim[0], :envDim[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


def bubblesGraphMask(envDim, bubbles_pos, bubbles_rad, radInflationMult=1.0, radInflationAdd=0.0):
    """Returns the boolean mask associated with a given disk-graph
    
    Args:
        envDim: list or array of size 2, describing the size of the grid environment
        bubbles_pos: array of size (N,2) representing the position of the bubbles centers in the grid
        bubbles_rad: list of float, radii of the given bubbles
        radInflationMult: multiplier for the bubbles radii
        radInflationAdd: float to add to the bubbles radii

    Returns:
        A 2D boolean-valued numpy array, with 1 inside the circle, 0 outside 
    """
    mask = np.zeros(envDim)
    for k, bpos in enumerate(bubbles_pos):
        mask = np.logical_or(mask, circularMask(
            envDim, bpos, bubbles_rad[k]*radInflationMult + radInflationAdd))
    return mask


def circDistMap(envDim, center, radius):
    """Returns the euclidean distance field to the specified circle

    Args:
        envDim: list or array of size 2, describing the size of the grid environment
        center: list or array of size 2 representing the position of the circle center in the grid
        radius: float, radius of the circle the distance is computed to

    Returns:
        A 2D real-valued numpy array, representing the distance to the surface of the circle (i.e. 0 inside and on the perimeter)
    """
    X, Y = np.ogrid[:envDim[0], :envDim[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    dist_map = (dist_from_center - radius)*(1.-(dist_from_center <= radius))
    return dist_map


def exponDistribMap(distMap, scale_factor):
    """Returns an exponential probability distribution map from a euclidean distance map (probability decaying with the distance)

    Args:
        distMap: 2D array, containing distance values (euclidean distance to obstacles)
        scale_factor: scale to apply to the exponential distribution 

    Returns:
        A 2D real-valued probability map (all elements sum to 1)
    """
    dist_zeros = np.where(distMap == 0)
    expon_distrib = expon(scale=scale_factor)
    probaMap = expon_distrib.pdf(distMap)
    probaMap[dist_zeros] = 0
    return probaMap/probaMap.sum()


def sampleInProbaMap(probaMap, prng, n=None):
    """Samples a random grid configuration in a grid probability map (sample probability proportional to grid cell value)

    Args:
        probaMap: 2D array, containing probability values (i.e. the sum over probaMap elements is 1)
        prng: pseudo-random number generator
        n: int or None. If not None, number of samples to return

    Returns:  
        A numpy array of shape (2,n) containing grid positions sampled using the probability map
    """
    i = prng.choice(np.arange(probaMap.size), p=probaMap.ravel(), size=n)
    return np.unravel_index(i, probaMap.shape)


class BubbleNode:
    """Class to represent a graph node as a free-space bubble

    Attributes:
        pos: array of size 2, 2D position of the node
        id: int, id of the node
        bubble_rad: float, free-space radius of the node
    """

    def __init__(self, pos, id):
        """Inits BubbleNode with pos and id"""
        self.pos = pos
        self.id = id
        self.bubble_rad = 0

    def __lt__(self, other):
        """Compares bubbles radii"""
        return self.bubble_rad < other.bubble_rad

    def setBubbleRad(self, bubble_rad):
        """Sets the bubble_rad attribute to a known value"""
        self.bubble_rad = bubble_rad

    def computeBubbleRad(self, env, robot_rad=0.1):
        """Computes the bubble radius from the environment and sets it"""
        self.setBubbleRad(env.queryDistAt(self.pos) - robot_rad)

    def edgeValidityTo(self, other_node, min_rad):
        """Checks the strict edge validity condition to another node

        Args:
            other_node: BubbleNode, other node to which the edge is verified
            min_rad: float, minimal bubble radius

        Returns:
            A boolean, True if the edge is indeed valid
        """
        # TODO : ensure min_rad takes into account the correct distance (for now the definition is flawed a bit)
        return (self.bubble_rad + other_node.bubble_rad) > np.linalg.norm(self.pos - other_node.pos)

    def validEdgesFromList(self, others, env, min_rad):
        """Checks edges validity for a list of other nodes

        Arg:
            others: list of BubbleNodes
            env: OccupancyEnv containing the occupancy grid map to check collisions
            min_rad: float, minimal bubble radius

        Returns:
            A list of int pairs, each representing an edge (as [node_id_0, node_id_1] )
        """
        validity_list = []
        for node in others:
            if self.edgeValidityTo(node, min_rad) and env.checkSegCollisions(self.pos, node.pos, seg_res=0.2):
                if node.id != self.id:
                    validity_list.append(node.id)
        valid_edges = [[self.id, valid_i] for valid_i in validity_list]
        return valid_edges

    def isInsideOthers(self, others, rad_mult=1, rad_add=0):
        """Checks if the center of this node is inside other nodes' bubbles

        Arg:
            others: list of BubbleNodes, other nodes to check

        Returns:
            A boolean, True if this node is inside one of the others
        """
        others_rad = [o.bubble_rad for o in others]
        others_dist = distancesPointToSet(
            self.pos, np.array([o.pos for o in others]))
        return not (np.array(others_dist) > rad_add + rad_mult*np.array(others_rad)).all()

    def computeMissingBubblesTo(self, other_node, env, min_rad):
        """Computes a list of bubbles to link two existing nodes in the sense of strict edge validity

        Args:
            other_node: BubbleNode, other node to compute missing bubbles towards
            env: OccupancyEnv for collisions check
            min_rad: minimal bubble radius

        Returns:
            A list of BubbleNodes, empty if no valid overlapping bubbles can be found, 
            otherwise ensuring a chain of bubbles between this node and other_node and
            verifying strict edge validity 
        """
        br0 = self.bubble_rad
        br1 = other_node.bubble_rad
        valid = self.edgeValidityTo(other_node, min_rad)
        if valid:
            return [self, other_node]
        else:
            t_b0 = br0/np.linalg.norm(other_node.pos - self.pos)
            t_b1 = br1/np.linalg.norm(other_node.pos - self.pos)
            t_mid = 0.5*(1 + t_b0 - t_b1)

            weighted_mid_pos = self.pos + t_mid*(other_node.pos - self.pos)
            weighted_mid_node = BubbleNode(weighted_mid_pos, -1)
            weighted_mid_node.computeBubbleRad(env)

            if weighted_mid_node.bubble_rad < min_rad:
                return []

            l0 = self.computeMissingBubblesTo(weighted_mid_node, env, min_rad)
            l1 = weighted_mid_node.computeMissingBubblesTo(
                other_node, env, min_rad)

            if (len(l0) and len(l1)):
                return l0 + l1[1:]
            return []

    def getBubbleRelaxedEdgeValidityTo(self, other_node, env, min_rad):
        """
        Checks if two bubbles can be linked with intermediary valid bubbles,
        i.e. relaxed edge validity condition

        Args:
            other_node: BubbleNode, other node to compute missing bubbles towards
            env: OccupancyEnv for collisions check
            min_rad: minimal bubble radius

        Returns:
            A boolean, True if missing bubbles can be found to other_node
        """
        linkBubbles = self.computeMissingBubblesTo(other_node, env, min_rad)
        if not len(linkBubbles):
            return False, None
        else:
            return True, linkBubbles

    def filterPosListInd(self, pos_list, inner_range_mult=0, outer_range_mult=1):
        """Filters a list of positions to extract the ones in specified range

        Args:
            pos_list: list or array of shape (N,2), representing 2D positions
            inner_range_mult: radius multiplier defining the inner radius of the validity zone 
            outer_range_mult: radius multiplier defining the outer radius of the validity zone

        Returns:
            The indices of the valid positions in the list/array
        """
        pos_kdtree = KDTree(pos_list)
        inside_inner_ind = pos_kdtree.query_ball_point(
            self.pos, inner_range_mult*self.bubble_rad)
        inside_outer_ind = pos_kdtree.query_ball_point(
            self.pos, outer_range_mult*self.bubble_rad)
        return list(set(inside_outer_ind) - set(inside_inner_ind))
