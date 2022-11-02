from dgprm_utils import *
from pqdict import pqdict


class DiskGraphPRM:
    """Disk Graph Probabilistic Roadmap planner
    Holds the state of the disk graph and provides all necessary methods for
    sampling candidate nodes, expanding the graph, updating it on map updates
    and planning paths in accessible space

    Attributes:
        graph: networkx.Graph, current state of the roadmap
        nodes_ids: list of int, list of the nodes IDs currently in the graph
        prng: np.random.RandomState, pseudo-random number generator for sampling
        samples_count: int, total number of candidate positions sampled since initialization

        sampling_cutoff_dist: float, parameter controlling the exponential decay of the probability map for the sampling step
        n_knn: int, parameter controlling the number of neighbors considered for graph operations
        robot_rad: float, parameter controlling the minimal size of the graph nodes bubbles
        max_rad: float, parameter controlling the maxmimal size of the graph nodes bubbles
        closing_rad_mult: float, parameter controlling how the samples are moved to the closed set during expansion
        opening_rad_mult: float, parameter controlling how the samples are moved to the open set during expansion
    """

    def __init__(self, seed=None):
        """Initialize state attributes"""
        self.graph = nx.Graph()
        self.nodes_ids = []
        self.prng = np.random.RandomState(seed=seed)  # init. prng
        self.samples_count = 0

    def initFromHyperparamsDict(self, hyperparams_dict):
        """Initialize parameters attributes from a dictionary"""
        self.sampling_cutoff_dist = hyperparams_dict["sampling_cutoff_dist"]
        self.n_knn = hyperparams_dict["n_knn"]
        self.robot_rad = hyperparams_dict["robot_rad"]
        self.max_rad = hyperparams_dict["max_rad"]
        self.closing_rad_mult = hyperparams_dict["closing_rad_mult"]
        self.opening_rad_mult = hyperparams_dict["opening_rad_mult"]

    # Utilities
    def getNewId(self):
        """Returns an unused (int) node ID"""
        if not len(self.nodes_ids):
            return 0
        return max(self.nodes_ids) + 1

    def addBubble(self, bubble_node):
        """Adds a new BubbleNode to the graph

        Args:
            bubble_node: BubbleNode, the node to add

        Returns:
            The ID of the added node, newly assigned when it gets added
        """
        new_id = self.getNewId()
        bubble_node.id = new_id
        self.graph.add_node(bubble_node.id, node_obj=bubble_node)
        if not (bubble_node.id in self.nodes_ids):
            self.nodes_ids.append(bubble_node.id)
        return bubble_node.id

    def addBubbleAndEdges(self, bubble_node, env, forbidden_neighbors=None):
        """Adds a new BubbleNode to the graph and compute the edges to its graph neighbors.

        Args:
            bubble_node: BubbleNode, the node to add
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions

        Returns:
            The ID of the added node
        """
        if forbidden_neighbors is None:
            forbidden_neighbors=[]
        new_id = self.addBubble(bubble_node)
        neighbors_ids = self.getClosestNodes(bubble_node.pos, k=self.n_knn+1+len(forbidden_neighbors))
        neighbors = [self.graph.nodes[ni]["node_obj"]
                     for ni in neighbors_ids if (ni != bubble_node.id) and (not ni in forbidden_neighbors)]
        edges = bubble_node.validEdgesFromList(neighbors, env, self.robot_rad)
        for e in edges:
            self.graph.add_edge(e[0], e[1], length=1)
        return new_id

    def recomputeEdgesLengths(self):
        for n1, n2, e in self.graph.edges(data=True):
            ni = self.graph.nodes[n1]["node_obj"]
            nj = self.graph.nodes[n2]["node_obj"]
            e["length"] = np.linalg.norm(ni.pos - nj.pos)
            # print(e["length"])

    def tryAddBubble(self, bubble_node, env, forbidden_neighbors=None):
        """Checks the validity of a node and adds it to the graph if possible.
        The validity here is determined by the radius of the node bubble which should be in range [self.robot_rad, self.max_rad],
        and on the condition that the new node can not be inside an existing one.
        If those conditions are valid, the addBubbleAndEdges method is called on the node.

        Args:
            bubble_node: BubbleNode, the node to check and add
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions

        Returns:
            None if the node does not get added, its ID otherwise
        """
        if (bubble_node.bubble_rad < self.robot_rad) or (bubble_node.bubble_rad > self.max_rad):
            return None
        current_nodes = [self.graph.nodes[i]["node_obj"]
                         for i in self.nodes_ids]
        if (bubble_node.isInsideOthers(current_nodes, rad_mult=0.8)) :
            return None
        new_id = self.addBubbleAndEdges(bubble_node, env, forbidden_neighbors=forbidden_neighbors)
        return new_id

    def removeBubble(self, bubble_ind):
        """Removes a node from the graph.
        Its index also gets removed from the self.nodes_ids list.

        Args:
            bubble_ind: int, the ID of the node to remove
        """
        self.graph.remove_node(bubble_ind)
        self.nodes_ids.remove(bubble_ind)

    def addBubblesList(self, nodes_list, env):
        """
        Adds a list of nodes to the graph.
        To avoid "overpopulating" the graph, nodes are added in radius decreasing order
        using the TryAddBubble method.

        Args:
            nodes_list: list of BubbleNode, nodes to try and add to the graph
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions

        Returns:
            A list of added nodes IDs
        """
        sorted_nodes = sorted(nodes_list, reverse=True)
        new_ids = []
        while len(sorted_nodes):
            curr_node = sorted_nodes.pop(0)
            new_node_id = self.tryAddBubble(curr_node, env)#, forbidden_neighbors=new_ids[:-1])
            if new_node_id is not None:
                new_ids.append(new_node_id)
        return new_ids

    def getNodesKDTree(self):
        """Gets the scipy.KDTree derived from the current graph nodes"""
        nodes_pos = [self.graph.nodes[nid]
                     ["node_obj"].pos for nid in self.nodes_ids]
        return KDTree(nodes_pos)

    def getClosestNodes(self, pos, k=1):
        """Gets the k closest nodes, in the current graph, to a given position

        Args:
            pos: np.array of size 2, 2D position to compute the closest nodes to
            k: int, number of closest nodes to return

        Returns:
            A numpy arraay containing the k closest nodes IDs
        """
        if len(self.graph.nodes) == 0:
            return None
        k = min(k, len(self.graph.nodes))
        nodes_tree = self.getNodesKDTree()
        dist, ids = nodes_tree.query(pos, k=k)
        if k == 1:
            return [np.array(self.nodes_ids)[ids]]
        return np.array(self.nodes_ids)[ids]

    def getFrontiersNodes(self, env):
        """Gets the current frontier nodes of the graph, in the sense of occupancy.
        These nodes are defined as the ones overlapping unknown space in the occupancy map.

        Args:
            env: OccupancyEnv containing the occupancy map to check

        Returns:
            A list of BubbleNode containing the frontier nodes
        """
        RAD_MULTIPLIER = 0.9
        MIN_UNKNOWN_CELLS = 5

        nodes = [self.graph.nodes[nid]["node_obj"] for nid in self.nodes_ids]
        frontiers = []
        for n in nodes:
            nmask = circularMask(env.map.shape, n.pos,
                                 RAD_MULTIPLIER*n.bubble_rad)
            unknown_in_vbubble = np.array(nmask*(env.map == 0.5), float)
            if np.sum(unknown_in_vbubble) > MIN_UNKNOWN_CELLS:
                frontiers.append(n)
        return frontiers

    def computeCurrentMapCoverage(self, env):
        """TODO: Docstring"""
        nodes_list = [self.graph.nodes[nid]["node_obj"]
                      for nid in self.graph]
        bubbles_mask = bubblesGraphMask(
            env.map_dim, [n.pos for n in nodes_list], [n.bubble_rad for n in nodes_list])
        covered_map = np.logical_and(bubbles_mask, (env.map < 0.4))
        available_map_surface = np.sum(env.map < 0.4)
        covered_map_surface = np.sum(covered_map == 1)

        if available_map_surface == 0:
            return None
        return 1.*covered_map_surface/available_map_surface

    # Sampling
    def getBubblesDistMap(self, env, nodes_ids=None, radInflationMult=1, radInflationAdd=0, maskObst=True, maskOccupancyAbove=0.4, maskOthers=True):
        """Gets the distance map to the current graph bubbles, using the circDistMap function.

        Args:
            env: OccupancyEnv, containing the occupancy map to use as a source for the distance map
            nodes_ids: list of int, the nodes IDs to consider for the distance map computation. Defaults to all nodes IDs if None
            radInflationMult: float (default 1), inflation multiplicator to apply to the graph nodes radii
            radInflationAdd: float (default 0), inflation addition to apply to the graph nodes radii
            maskObst: boolean (default False), determines if the distance map should be 0 where the occupancy map contains obstacles

        Returns:
            A 2D np.ndarray matching the shape of the occupancy map in env, containing the distance values to the current graph bubbles
        """
        if nodes_ids is None:
            nodes_ids = self.nodes_ids
        nodes_list = [self.graph.nodes[nid]["node_obj"]
                      for nid in nodes_ids]
        bubbles_dist_maps = [circDistMap(
            env.map_dim, n.pos, n.bubble_rad*radInflationMult + radInflationAdd) for n in nodes_list]

        # obst_mask = np.ones(env.map.shape)
        # if maskObst:
        obst_mask = np.array(env.map > maskOccupancyAbove)

        others_mask = np.ones(env.map.shape)
        if maskOthers and len(nodes_ids) < len(self.nodes_ids):
            others_ids = list(set(self.nodes_ids) - set(nodes_ids))
            others_list = [self.graph.nodes[nid]["node_obj"]
                           for nid in others_ids]
            others_dist_maps = [circDistMap(
                env.map_dim, n.pos, n.bubble_rad*radInflationMult + radInflationAdd) for n in others_list]
            others_mask = (np.minimum.reduce(others_dist_maps) > 0)
        return others_mask*(1-obst_mask)*np.minimum.reduce(bubbles_dist_maps)

    def sampleFromExponDistrib(self, env, n, sampling_dist, nodes_ids=None, radInflationMult=1, radInflationAdd=0, maskOccupancyAbove=0.4, show=False):
        """Samples the environment with a decaying exponential distribution
        derived from the distance to the current graph bubbles.

        Args:
            env: OccupancyEnv, containing the occupancy map to use as a source for the probability map
            n: number of sample positions to return
            sampling_dist: parameter controlling the scale factor of the exponential distribution
            nodes_ids: list of int, the nodes IDs to consider for the sampling. Defaults to all nodes IDs if None
            radInflationMult: float (default 1), inflation multiplicator to apply to the graph nodes radii
            radInflationAdd: float (default 0), inflation addition to apply to the graph nodes radii
            show: boolean (defaulf False), displays a matplotlib visualization of the sampling

        Returns:
            A numpy array of shape (n,2) containing the positions sampled in the probability map
        """
        bubbles_dmap = self.getBubblesDistMap(
            env, nodes_ids=nodes_ids, radInflationMult=radInflationMult, radInflationAdd=radInflationAdd, maskOccupancyAbove=maskOccupancyAbove)
        pmap = exponDistribMap(bubbles_dmap, sampling_dist)
        samples = sampleInProbaMap(pmap, self.prng, n=n)
        samples = np.array(samples).T
        if show:
            env.display()
            plt.imshow(pmap.T, cmap=plt.cm.hot, alpha=0.8)
            plt.plot(samples[:, 0], samples[:, 1], color='dodgerblue',
                     ls='', marker='+', zorder=110, alpha=0.8, ms=10)
            #plt.show()
        return samples

    # Expansion
    def processCandidate(self, cand_node, env, show=False, save=None):
        """Process one candidate node, checking it using the relaxed edge validity condition, and add the relevant bubbles to the graph

        Args:
            cand_node: BubbleNode, the node to check and try adding to the graph
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions

        Returns:
            new_ids: a list of int containing the IDs of the nodes added to the graph with this candidate, empty if not added
            true_validities: for each new id, True if the link is a true validity condition, False if relaxed
        """        
        new_ids = []
        current_nodes = [self.graph.nodes[i]["node_obj"]
                         for i in self.nodes_ids]
        if (cand_node.bubble_rad < self.robot_rad): # or (cand_node.isInsideOthers(current_nodes, rad_add=-self.robot_rad)):
            return new_ids
        # Display
        if show:
            env.display()
            displayBubbles(self.graph, color='lightgrey')
            int_circles = []
            cand_color=None
            
        closest_ids = self.getClosestNodes(cand_node.pos, k=self.n_knn)
        for i in closest_ids:
            graph_node = self.graph.nodes[i]["node_obj"]
            relaxedValidity, linkBubbles = cand_node.getBubbleRelaxedEdgeValidityTo(
                graph_node, env, self.robot_rad)
            if not relaxedValidity:
                # Display
                if show:
                    plt.scatter(graph_node.pos[0], graph_node.pos[1], color='grey')
                    plt.plot([cand_node.pos[0],graph_node.pos[0]],[cand_node.pos[1],graph_node.pos[1]], color='grey', marker='', lw=2, zorder=100, ls='--', label='Invalid edge')
                    if not (cand_color=='green' or cand_color=='orange'):
                        cand_color='red'
                    
                continue
            if len(linkBubbles) == 2:
                new_id = self.tryAddBubble(cand_node, env)
                if new_id is not None:
                    new_ids.append(new_id)
                # Display
                if show:      
                    plt.scatter(linkBubbles[1].pos[0],linkBubbles[1].pos[1], color='green', marker='X', s=300, zorder=100)
                    plt.plot([linkBubbles[0].pos[0],linkBubbles[1].pos[0]],[linkBubbles[0].pos[1],linkBubbles[1].pos[1]], color='green', marker='', lw=6, zorder=100, label='Valid edge')
                    cand_color='green'
                                        
            elif len(linkBubbles) > 2:
                prev_len = len(new_ids)
                invalid = False
                for lb in linkBubbles[:-1] :
                    if lb.isInsideOthers(current_nodes, rad_add=-self.robot_rad):
                        invalid = True
                        break
                if invalid :
                    continue
                new_ids.extend(self.addBubblesList(linkBubbles[:-1], env))
                # Display
                if show:
                    lb_pos = np.array([lb.pos for lb in linkBubbles[1:]])
                    plt.scatter(lb_pos[:,0],lb_pos[:,1], color='orange', marker='P', s=250, zorder=100)
                    if not cand_color=='green':
                        cand_color='orange'
                    for lb in linkBubbles[1:]:
                        plt.plot([linkBubbles[0].pos[0],lb.pos[0]],[linkBubbles[0].pos[1],lb.pos[1]], color='orange', marker='', lw=4, zorder=100, label='Relaxed valid edge')
                        circ = plt.Circle(lb.pos, lb.bubble_rad, ec='orange', fc=[1,1,1,0], color=None, lw=4, ls='--')
                        int_circles.append(circ)
        
        if show:
            plt.scatter(cand_node.pos[0],cand_node.pos[1], color='orange', marker='X', s=480, zorder=110, label="Curr. candidate")
            cand_circ = plt.Circle(cand_node.pos, cand_node.bubble_rad, ec=cand_color, fc=[1,1,1,0], color=None, lw=6)
            plt.gca().add_patch(cand_circ)
            for c in int_circles:
                plt.gca().add_patch(c)
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(),loc='upper right', bbox_to_anchor=(1.1, 1.1))
            
            if save is not None:
                plt.savefig(save + ".png", facecolor=[1,1,1,1])
            plt.show()
      
        return new_ids

    def expand(self, env, init_samples, max_iter, show=False, save=None):
        """Expands the graph from a set of sample positions

        Args:
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions
            init_samples: a numpy array of shape (N,2) containing the sample positions to expand from
            max_iter: the maximum allowed number of iterations to terminate the expansion loop

        Returns:
            A numpy array of shape (N-A,2) containing the sample positions left unvisited
        """

        def closeSamplesOnIDs(bubbles_ids, samples, osamp, usamp, csamp):
            """Samples closing method"""
            closed_ind = []
            for id in bubbles_ids:
                node = self.graph.nodes[id]["node_obj"]
                closed = node.filterPosListInd(
                    samples, inner_range_mult=0, outer_range_mult=self.closing_rad_mult)
                closed_ind = list(set(closed_ind).union(set(closed)))
            for ci in closed_ind:
                csamp.append(ci)
                if ci in osamp:
                    osamp.pop(ci)
                if ci in usamp:
                    usamp.pop(ci)

        def openSamplesOnIDs(bubbles_ids, samples, sample_nodes, osamp, usamp):
            """Samples opening method"""
            open_ind = []
            for id in bubbles_ids:
                node = self.graph.nodes[id]["node_obj"]
                open = node.filterPosListInd(
                    samples, inner_range_mult=self.closing_rad_mult, outer_range_mult=self.opening_rad_mult)
                open_ind = list(set(open_ind).union(set(open)))
            for oi in open_ind:
                if oi in usamp:
                    usamp.pop(oi)
                if not oi in osamp:
                    osamp.additem(oi, -sample_nodes[oi].bubble_rad)
        
        # Initialization
        open_samp = pqdict()
        closed_samp = list()
        unvisited_samp = pqdict()
        
        # Display var.
        current_candidate = None
        current_valid = None
        current_invalid = None
        current_intermediate = None

        max_id = max(self.nodes_ids)
        sample_nodes = [BubbleNode(pos, k+1+max_id)
                        for k, pos in enumerate(init_samples)]
        sample_nodes_ids = [sn.id for sn in sample_nodes]
        for n in sample_nodes:
            n.computeBubbleRad(env)
        # shuffle(sample_nodes)
        for k, n in enumerate(sample_nodes):
            unvisited_samp.additem(k, -n.bubble_rad)

        # Initial search : univisited set as open set
        init_node_found = False
        init_ind = 0
        while (not init_node_found) and len(unvisited_samp):
            init_ind = unvisited_samp.top()
            unvisited_samp.pop(init_ind)
            cand_init = sample_nodes[init_ind]
            # init_node_found = tryAddCandToGraph(cand_init)
            if save is not None :
                added_ids = self.processCandidate(cand_init, env, show=show, save=save + "_{:03d}".format(len(self.nodes_ids)))
            else:
                added_ids = self.processCandidate(cand_init, env, show=show, save=save)
            #print(added_ids)
            init_node_found = len(added_ids)
            init_ind += 1
            
        if not init_node_found:
            print("Could not init. - resampling")
            return [], True
        # Get samples in opening/closing range and update the relevant sets accordingly
        # closeSamplesOnIDs(added_ids, init_samples, open_samp,
        #                  unvisited_samp, closed_samp)
        openSamplesOnIDs(added_ids, init_samples,
                         sample_nodes, open_samp, unvisited_samp)
        

        # Core loop
        for k in range(max_iter):
            if not(len(open_samp)):
                return [], True
            # Get open sample with max. radius
            curr_op = open_samp.top()
            cand_node = sample_nodes[curr_op]
            # Check its validity and add it to graph
            if save is not None :
                added_ids = self.processCandidate(cand_node, env, show=show, save=save + "_{:03d}".format(k+len(self.nodes_ids)))
            else:
                added_ids = self.processCandidate(cand_node, env, show=show, save=save)
            closeSamplesOnIDs(added_ids, init_samples,
                              open_samp, unvisited_samp, closed_samp)
            openSamplesOnIDs(added_ids, init_samples,
                             sample_nodes, open_samp, unvisited_samp)
            # Close the current candidate
            if curr_op in open_samp:
                open_samp.pop(curr_op)
            if curr_op in unvisited_samp:
                unvisited_samp.pop(curr_op)
            # closed_samp.append(curr_op)

        return init_samples[unvisited_samp], True

    def expandNSamples(self, env, n_samples, nodes_ids=None, maskOccupancyAbove=0.4):
        """Samples N new samples using the current graph nodes and expands the graph from this set.

        Args:
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions
            n_samples: int, the number of new positions to sample
            nodes_ids: list of int, the nodes IDs to consider for the sampling. Defaults to all nodes IDs if None

        Returns:
            The number of nodes added to the graph as an int
        """
        MAX_ITER = 1000

        samples = self.sampleFromExponDistrib(
            env, n_samples, self.sampling_cutoff_dist, nodes_ids=nodes_ids, maskOccupancyAbove=maskOccupancyAbove)
        init_count = len(self.graph.nodes)
        print("Sampling {} new samples - Init. vertices in graph : {}".format(n_samples, init_count))
        self.samples_count += len(samples)
        samples = np.array(samples)
        _, expand_success = self.expand(env, samples, MAX_ITER)
        print("Finished processing samples - {}/{} added".format(
            len(self.graph.nodes) - init_count, n_samples))
        return max(0, len(self.graph.nodes) - init_count)

    # Update
    def recomputeNodesRadii(self, env):
        """Recomputes the radii of current graph nodes and returns the invalid and modified ones

        Args:
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions

        Returns:
            invalid_nodes: list of int, containing the IDs of the newly invalid nodes in terms of radius size
            modified_nodes: list of int, containing the IDS of the nodes which radius was modified but remained valid
        """
        modified_nodes = []
        invalid_nodes = []
        nodes = [self.graph.nodes[nid]["node_obj"] for nid in self.nodes_ids]
        for n in nodes:
            old_rad = n.bubble_rad
            n.computeBubbleRad(env)
            if n.bubble_rad < self.robot_rad:
                invalid_nodes.append(n.id)
            elif n.bubble_rad != old_rad:
                modified_nodes.append(n.id)
        return invalid_nodes, modified_nodes

    def updateEdges(self, env, modified_nodes):
        """Removes edges if they are invalid.
        Validity is checked on 2 conditions, collision with obstacles and strict edge validity to neighbor nodes

        Args:
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions
            modified_nodes: list of int, containing the IDs of modified nodes (radius modified), which may now have invalid edges
        """
        invalid = []
        mod_edges = []
        for e in self.graph.edges:
            if (e[0] in modified_nodes) or (e[1] in modified_nodes):
                mod_edges.append(e)

        for e in mod_edges:
            n0, n1 = self.graph.nodes[e[0]]["node_obj"],\
                self.graph.nodes[e[1]]["node_obj"]
            if not env.checkSegCollisions(n0.pos, n1.pos, seg_res=0.2):
                invalid.append([e[0], e[1]])

        for mod in modified_nodes:
            neighbors = self.graph.neighbors(mod)
            for ngb in neighbors:
                if not self.graph.nodes[ngb]["node_obj"].edgeValidityTo(self.graph.nodes[mod]["node_obj"], self.robot_rad):
                    invalid.append([mod, ngb])
        self.graph.remove_edges_from(invalid)

    def removeSmallComponents(self, ref_pos, min_comp_size):
        """Removes all edges inside connected components under a minimal size to allow for nodes removal

        Args:
            ref_pos: array of size 2, a reference 2D position. The connected component closest to this position is unaffected.
            min_comp_size: int, minimal number of nodes in a connected component to avoid edges removal
        """
        invalid_edges = []
        ref_node = self.getClosestNodes(ref_pos)[0]
        for component in list(nx.connected_components(self.graph)):
            # if (len(component) < min_comp_size) and not (ref_node in component):
            if not ref_node in component:
                for node in component:
                    invalid_edges.extend(self.graph.edges(node))
        self.graph.remove_edges_from(invalid_edges)

    def getIsolatedNodes(self):
        """Returns IDs of all current graph nodes with no neighbors."""
        isolated = []
        for nid in self.graph.nodes:
            if not(len(self.graph.edges(nid))):
                isolated.append(nid)
        return isolated

    def updateOnMapUpdate(self, env, invalid_nodes, modified_nodes, ref_pos, min_comp_size):
        """Updates the graph on updated occupancy map data.
        Knowing the nodes radii have been updated first, this method removes all invalid edges
        and small connected components edges, tries to add back all isolated nodes as candidates,
        and removes those which could not be relinked to the graph.

        Args:
            env: OccupancyEnv, holding the occupancy map to check for obstacles collisions
            modified_nodes: list of int, containing the IDs of modified nodes (radius modified), which may now have invalid edges
            ref_pos: array of size 2, a reference 2D position. The connected component closest to this position is unaffected.
            min_comp_size: int, minimal number of nodes in a connected component to avoid edges removal
        """
        self.graph.remove_nodes_from(invalid_nodes)
        self.nodes_ids = list(set(self.nodes_ids) - set(invalid_nodes))
        self.updateEdges(env, modified_nodes)
        self.recomputeEdgesLengths()
        self.removeSmallComponents(ref_pos, min_comp_size)
        ref_node = self.getClosestNodes(ref_pos)[0]
        isolated = self.getIsolatedNodes()
        if ref_node in isolated:
            isolated.remove(ref_node)
        isol_bubbles = [self.graph.nodes[n]["node_obj"] for n in isolated]
        for n in isolated:
            self.removeBubble(n)
        for isb in isol_bubbles:
            #isol_bubble = 
            # self.removeBubble(n)
            self.processCandidate(isb, env)
        isolated = self.getIsolatedNodes()
        if ref_node in isolated:
            isolated.remove(ref_node)
        self.graph.remove_nodes_from(isolated)
        self.nodes_ids = list(set(self.nodes_ids) - set(isolated))

    # Path planning
    def isReachable(self, pos):
        """Returns True if the given pos is inside one of the current graph bubbles, and the ID of the corresponding node"""
        closest_ind = self.getClosestNodes(pos, k=3*self.n_knn)
        closest = [self.graph.nodes[i]["node_obj"] for i in closest_ind]
        brads = np.array(
            [self.graph.nodes[i]["node_obj"].bubble_rad for i in closest_ind])
        for k, cl in enumerate(closest):
            if np.linalg.norm(cl.pos - pos) < brads[k]:
                return True, cl.id
        return False, -1

    def planPath(self, start_pos, goal_pos):
        """If currently available, returns a path between the given start and goal positions

        Args:
            start_pos: numpy array of size 2, start position of the path
            goal_pos: numpy array of size 2, goal position of the path

        Returns:
            waypoints: the path as a list of waypoints (the nodes positions), None if no path is available
            radii: the bubble radius associated to each node in the path, 0 for start and goal
        """
        # Check path existence conditions
        start_reachable, start_id = self.isReachable(start_pos)
        goal_reachable, goal_id = self.isReachable(goal_pos)
        if not (start_reachable and goal_reachable):
            print("Start reachable :{}\nGoal reachable :{}\nEither start or goal is unreachable in current graph".format(
                start_reachable, goal_reachable))
            return None, None
        if not nx.has_path(self.graph, start_id, goal_id):
            print("No path exists between start {} and goal {} in the current graph".format(
                start_id, goal_id))
            return None, None
        # Query networkx for the shortest path in the graph
        self.recomputeEdgesLengths()
        path_indices = nx.shortest_path(
            self.graph, start_id, goal_id, weight="length")
        path_disk_waypoints = [self.graph.nodes[pi]
                               ["node_obj"].pos for pi in path_indices]
        path_disk_radii = radii = [self.graph.nodes[pi]
                                   ["node_obj"].bubble_rad for pi in path_indices]

        if len(path_indices) == 1:
            path_waypoints = np.concatenate([
                start_pos.reshape(1, 2),
                goal_pos.reshape(1, 2)
            ])
            path_radii = [0, 0]
            return np.array(path_waypoints), np.array(path_radii)

        # If path has more than 2 waypoints, ensure we dont go back through the bubbles centers for first and last node
        rad_first = self.graph.nodes[path_indices[0]
                                     ]["node_obj"].bubble_rad
        wp_first_alt = path_disk_waypoints[0] + rad_first*(
            path_disk_waypoints[1] - path_disk_waypoints[0])/np.linalg.norm(path_disk_waypoints[1] - path_disk_waypoints[0])

        if len(path_disk_waypoints) > 2:
            rad_last = self.graph.nodes[path_indices[-1]
                                        ]["node_obj"].bubble_rad
            wp_last_alt = path_disk_waypoints[-1] + rad_last*(path_disk_waypoints[-2] - path_disk_waypoints[-1])/np.linalg.norm(
                path_disk_waypoints[-2] - path_disk_waypoints[-1])

            path_waypoints = np.concatenate([
                start_pos.reshape(1, 2),
                wp_first_alt.reshape(1, 2),
                path_disk_waypoints[1:-1],
                wp_last_alt.reshape(1, 2),
                goal_pos.reshape(1, 2)
            ])
            path_radii = [0, min(path_disk_radii[0], path_disk_radii[1])] + \
                path_disk_radii[1:-1] + \
                [min(path_disk_radii[-1], path_disk_radii[-2]), 0]

        else:
            path_waypoints = np.concatenate([
                start_pos.reshape(1, 2),
                wp_first_alt.reshape(1, 2),
                goal_pos.reshape(1, 2)
            ])
            path_radii = [0, min(path_disk_radii[0], path_disk_radii[1]), 0]

        return np.array(path_waypoints), np.array(path_radii)
    
    def expandPlanPath(self, start_pos, goal_pos, env, samples_per_expand, max_samples=100):
        """Expands until a path is found"""
        # Check path existence conditions
        start_reachable, start_id = self.isReachable(start_pos)
        goal_reachable, goal_id = self.isReachable(goal_pos)
        samples_count = 0
        while (not (start_reachable and goal_reachable)) and  samples_count < max_samples:
            self.expandNSamples(env, samples_per_expand, nodes_ids=None, maskOccupancyAbove=0.4)
            samples_count += samples_per_expand
            start_reachable, start_id = self.isReachable(start_pos)
            goal_reachable, goal_id = self.isReachable(goal_pos)
        return self.planPath(start_pos, goal_pos)