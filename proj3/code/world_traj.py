import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix as llm

from .graph_search import graph_search


class WorldTraj(object):
    """
    """
    def conditional_matrix(self, mat, submatrix, i, k):
        mat[k*i + 5, k*i + 14] = -1
        mat[k*i + 6, k*i + 13] = -2
        mat[k*i + 7, k*i + 12] = -6
        mat[k*i + k, k*i + 11] = -24
        mat[k*i + 9, k*i + 10] = -120
        mat[k*i + 10, k*i + 9] = -720
        mat[k*i + 3:k*i + 11, k*i:k*i + 8] = submatrix

        return mat

    def decompose_matrix(self, t):
        t_powers = [t**i for i in range(8)]
        homogeneous_row = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        matrix = np.array([[t_powers[7], t_powers[6], t_powers[5], t_powers[4], t_powers[3], t_powers[2], t, 1],
                        [7 * t_powers[6], 6 * t_powers[5], 5 * t_powers[4], 4 * t_powers[3], 3 * t_powers[2], 2 * t, 1, 0],
                        [42 * t_powers[5], 30 * t_powers[4], 20 * t_powers[3], 12 * t_powers[2], 6 * t, 2, 0, 0],
                        [210 * t_powers[4], 120 * t_powers[3], 60 * t_powers[2], 24 * t, 6, 0, 0, 0],
                        [840 * t_powers[3], 360 * t_powers[2], 120 * t, 24, 0, 0, 0, 0],
                        [2520 * t_powers[2], 720 * t, 120, 0, 0, 0, 0, 0],
                        [5040 * t, 720, 0, 0, 0, 0, 0, 0]], dtype=object)
        matrix = np.vstack((homogeneous_row, matrix))
        return matrix
    
    def waypoint_pruning(self, pruned_path):
        # # source: https://karthaus.nl/rdp/ and Wikipedia provided by TAs
        # # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        # Ramer–Douglas–Peucker algorithm

        if len(pruned_path) < 3:
            return pruned_path

        # Calculate deflection for each point
        defl_list = []
        for i in range(1, len(pruned_path) - 1):
            defl = np.linalg.norm(np.cross((pruned_path[0] - pruned_path[i]), (pruned_path[-1] - pruned_path[0]))) / np.linalg.norm(pruned_path[-1] - pruned_path[0])
            defl_list.append(defl)

        # Find the index with the maximum deflection
        index = np.argmax(defl_list)
        dmax = defl_list[index]

        if dmax > 0.5:
            # Recursively generate sparse waypoints for left and right segments
            recResult1 = self.waypoint_pruning(pruned_path[:index+1])
            recResult2 = self.waypoint_pruning(pruned_path[index:])
            x = np.vstack((recResult1[:-1], recResult2))
            return x

        else: 
            if np.linalg.norm(pruned_path[-1] - pruned_path[0]) > 2:
                # If the segment is long enough, return waypoints at the beginning, middle, and end
                x = np.vstack((pruned_path[0], pruned_path[len(pruned_path)//2], pruned_path[-1]))
                return x
            else:
                # Otherwise, return waypoints at the beginning and end
                x = np.vstack((pruned_path[0], pruned_path[-1]))
                return x

    def compose_matrix(self, t):
        t_powers = [t**i for i in range(8)]
        matrix = np.array([[t_powers[7], t_powers[6], t_powers[5], t_powers[4], t_powers[3], t_powers[2], t, 1],
                         [7 * t_powers[6], 6 * t_powers[5], 5 * t_powers[4], 4 * t_powers[3], 3 * t_powers[2], 2 * t, 1, 0],
                         [42 * t_powers[5], 30 * t_powers[4], 20 * t_powers[3], 12 * t_powers[2], 6 * t, 2, 0, 0],
                         [210 * t_powers[4], 120 * t_powers[3], 60 * t_powers[2], 24 * t, 6, 0, 0, 0],
                         [840 * t_powers[3], 360 * t_powers[2], 120 * t, 24, 0, 0, 0, 0]])

        return matrix

        
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.
        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.
        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)
        """
        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        # print('paht: ', self.path.shape)
        
        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = self.waypoint_pruning(self.path)
        # print('points: ', self.points.shape)   
        print('Number of Waypoints: ', len(self.points))

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester

        # STUDENT CODE HERE
        self.v = 12

        self.dist_seg = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)[:, np.newaxis]
        # print('dist_seg: ', self.dist_seg.shape)
        # self.unit_dir = (self.points[1:] - self.points[:-1]) / self.dist_seg

        self.time = self.dist_seg / self.v
        self.time[0] = 2.7*self.time[0]
        self.time[-1] = 2.7*self.time[-1]
        self.time *= (np.sqrt(1.65) / np.sqrt(self.time))

        self.time_array = np.hstack((0, np.cumsum(self.time)))


    def update(self, t):
        """
        PRIMARY METHOD
        Given the present time, return the desired flat output and derivatives.
        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        l = len(self.points)
        y = self.time
        final_t = np.sum(y)

        seg = 8*(l-1)
        p , q = llm((seg,seg)) , llm((seg,3))
        # print('p: ', p.shape)
        # print('q: ', q.shape)
        k = 0
        while k < l - 1:
            q[8 * k + 3:8 * k + 5, :] = self.points[k:k+2, :]
            k += 1

        q = q.tocsc()

        l2 = len(self.time)
        zip_file = zip(self.time, range(0, l2))
        # print('zip_file: ', zip_file)
        # print('zip_file: ', type(zip_file))
        for iter_time, i in zip_file:
            sub_matrix = self.decompose_matrix(iter_time)
            p[[0, 1, 2], [6, 5, 4]] = [1, 2, 6]

            if i != len(self.time) - 1:
                p = self.conditional_matrix(p, sub_matrix, i, 8)           
            else:
                p[8*i + 3 : 8*i + 11, 8*i : 8*i + 8] = sub_matrix[:5, :]

        # print('p : ', p)        
        p = p.tocsc()
        # print('p : ', p)    
        coefficient = spsolve(p, q).toarray()
        # print('coeff: ', coefficient)    

        if t < final_t: 
            segment_no_of_quadcop = (np.where(self.time_array - t > 0))[0][0] - 1
            # segment_number = segment_number[0][0] - 1
            current_time = t - self.time_array[segment_no_of_quadcop]

            solution_matrix = self.compose_matrix(current_time)
            # print('solution_matrix: ', solution_matrix)
            # print('solution_matrix: ', solution_matrix.shape)


            segment_coeff = coefficient[8*segment_no_of_quadcop : 8*(segment_no_of_quadcop + 1)]
            # print('segment_coeff: ', segment_coeff)
            # print('segment_coeff: ', segment_coeff.shape)
            
            solution_matrix = solution_matrix @ segment_coeff
            
            x_dot = solution_matrix[1, :]
            x = solution_matrix[0, :]
           
            # print('solution_matrix: ', solution_matrix)
            # print('solution_matrix: ', solution_matrix.shape)

            x_ddot = solution_matrix[2, :]
            x_ddddot = solution_matrix[4, :]
            x_dddot = solution_matrix[3, :]

            # print('x : ', x)    
            # print('x_dot : ', x_dot)
            # print('x_ddot : ', x_ddot)    
            # print('x_dddot : ', x_dddot)                

        else:
            x = self.points[-1]
            x_dot    = np.zeros((3,))
            x_ddot   = np.zeros((3,))
            x_dddot  = np.zeros((3,))
            x_ddddot = np.zeros((3,))
            yaw = 0
            yaw_dot = 0
            

        # STUDENT CODE END
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output