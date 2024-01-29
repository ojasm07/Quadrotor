from heapq import heappush, heappop  # Recommended.
import numpy as np

from collections import defaultdict

from flightsim.world import World

from occupancy_map import OccupancyMap # Recommended.

def path_finder_helper(parent_dict, occ_map, start_index, start, path, target_voxel = None):
    #This function is used to find the path from the parent dictionary and the start and goal index
    while target_voxel is not None:
        parent_node = parent_dict[target_voxel]
        if parent_node == start_index:
            path = [start] + path
            break

        path = [occ_map.index_to_metric_center(parent_node)] + path
        target_voxel = parent_node

    return path

def update_dist_and_parentdict(neigh_itr, index_of_current_state, distance_heuristic_dictionary, distance_dictionary, parent_dict, heuristic_distance, cost):
    # This function is used to update the distance and parent dictionary (astar = False)
    distance_heuristic_dictionary[neigh_itr] = heuristic_distance
    distance_dictionary[neigh_itr] = cost
    parent_dict[neigh_itr] = index_of_current_state

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # print(occ_map.map.shape)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    # print(start_index)
    goal_index = tuple(occ_map.metric_to_index(goal))
    # print(goal_index)


    discovered_cells = set()
    parent_dict = dict()
    distance_dictionary = defaultdict(lambda: float("inf"))
    distance_heuristic_dictionary = defaultdict(lambda: float("inf"))
    boundary_cells = [(0, start_index)]
    
    distance_dictionary[start_index] = 0
    distance_heuristic_dictionary[start_index] = np.linalg.norm((np.array(goal_index)) - np.array(start_index))

    while boundary_cells:

        index_of_current_state = heappop(boundary_cells)[1]

        if index_of_current_state == goal_index:
            path = list() 
            target_voxel = goal_index
            # print(target_voxel)
            path += [goal]
            # print("path", path)
            
            #To get the path
            path = path_finder_helper(parent_dict, occ_map, start_index, start, path, target_voxel)
            # print("path", path)
            # print(type(path))

            return np.array(path), len(discovered_cells)

        neighbor_voxel = np.array([[a,b,c] for a in range(-1,2) for b in range(-1,2) for c in range(-1,2) if (a,b,c) != (0,0,0)]) 
        all_neighbors = np.array(index_of_current_state) + neighbor_voxel
        avail_neighbors = all_neighbors[(np.all(all_neighbors >= 0, axis=1)) & (np.all(all_neighbors < occ_map.map.shape, axis=1))]
        neighbors = avail_neighbors[occ_map.map[avail_neighbors[:,0], avail_neighbors[:,1], avail_neighbors[:,2]] == 0]
        discovered_cells.add(index_of_current_state)

        for i in range(len(neighbors)):
            # print(neighbors[i])
            neigh_itr = tuple(neighbors[i])
            # print(neigh_itr)
            heuristic_distance = distance_heuristic_dictionary[index_of_current_state]
            heuristic_distance += np.linalg.norm((neighbors[i] - index_of_current_state))
            # print(heuristic_distance)
            cost = heuristic_distance

            if astar == True:
                cost = heuristic_distance + np.linalg.norm(neighbors[i] - goal_index)

            if heuristic_distance < distance_heuristic_dictionary[neigh_itr]:
                # print("heuristic_distance", heuristic_distance)
                
                # Updating the distance and parent dictionary
                update_dist_and_parentdict(neigh_itr, index_of_current_state, distance_heuristic_dictionary, distance_dictionary, parent_dict, heuristic_distance, cost)
                heappush(boundary_cells, (cost, neigh_itr))

    # Return a tuple (path, nodes_expanded)
    return None, 0