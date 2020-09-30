"""
    Description:    Construct a class to model the safety and attention environment
    Author:         Michael Hibbard, Abraham Vinod
    Created:        September 2020
"""
import numpy as np
from linear_dynamics_planning import linear_dynamics_planning

class RobotSAEnvironment:

    def __init__(self, target_state, target_tolerance, gauss_lev_param, planning_horizon,
                 rob_init_pos, rob_A_mat, rob_B_mat, obs_field_of_view_rad, obs_interval,
                 rob_state_max, rob_input_max):
        """
        input: target_state: the state that the robot seeks to reach
        input: target_tolerance: the tolerance for which we say the robot has "reached"
            its target state
        input: gauss_lev_param: the probability for which we "slice" the Gaussian level sets
        input: planning_horizon: the number of times steps forward to plan the robot's
            nominal trajectory
        input: rob_init_pos: the initial position of the robot in the environment
        input: rob_A_mat: matrix mapping current robot position to x' (in x' = Ax + Bu)
        input: rob_B_mat: matrix mapping control input to x' (in x' = Ax + Bu)
        input: obs_field_of_view_rad: radius of the obstacle's field of view. It can only
            make an observation about an obstacle within this area
        input: obs_interval: after each interval, the robot can make an observation about
            an obstacle within its field-of-view radius
        input: rob_state_max: absolute value of maximum distance in any direction the robot
            can be from the origin
        input: rob_input_max: absolute value of robot control input. NOT A PHYSICAL QUANTITY AT THIS MOMENT!
        """
        self.target_state = target_state
        self.target_tolerance = target_tolerance
        self.beta = gauss_lev_param
        self.planning_horizon = planning_horizon
        self.rob_pos = rob_init_pos
        self.rob_A_mat = rob_A_mat
        self.rob_B_mat = rob_B_mat
        self.obs_field_of_view_rad = obs_field_of_view_rad
        self.obs_interval = obs_interval
        self.rob_y_pos_max = rob_state_max
        self.rob_x_pos_max = rob_state_max
        self.rob_x_input_max = rob_input_max
        self.rob_y_input_max = rob_input_max

        # Setup some robot parameters based on inputs
        self.rob_state_dim = self.rob_A_mat.shape[1]
        self.rob_input_dim = self.rob_B_mat.shape[1]

        # Initialize the parameters for linear obstacles.
        # Upon initialization, assume empty. Add these manually
        self.lin_obs_list = []
        self.num_lin_obs = 0

        # Initialize the parameters for nonlinear obstacles.
        # Upon initialization, assume empty. Add these manually
        self.nonlin_obs_list = []
        self.num_nonlin_obs = 0

        # Encode information about the trajectory, control, active dual
        # variables, etc (essentially, outputs of the optimization problem
        self.num_active_dual_vars = []
        self.nominal_trajectory = np.zeros(
            (self.rob_state_dim, self.planning_horizon))

        # Total number of obstacles, linear and nonlinear
        self.num_obs = 0

        # Overall time step in the system
        self.total_time_steps = 0

        # Based on robot state dynamics for this environment, can iteratively
        # define the matrices to obtain X over the planning horizon. Call
        # this X = Z x_0 + H u
        self.Z, self.H = self.get_concatenated_matrices()

    def add_linear_obstacle(self, init_position, A_matrix, F_matrix,
                 w_mean_vec, w_cov_mat, radius):
        """
        input: init_position: initial position of linear obstacle-to-add
            in the environment
        input: A_matrix: A matrix of linear obstacle-to-add
        input: F_matrix: F matrix of linear obstacle-to-add
        input: w_mean_vec: mean disturbance signal of linear obstacle-to-add
        input: w_cov_mat: covariance matrix of disturbance signal of linear
            obstacle-to-add
        input: radius: radius of linear obstacle-to-add
        """
        self.lin_obs_list.append(LinearObstacle(init_position, A_matrix, F_matrix,
                 w_mean_vec, w_cov_mat, radius))
        self.num_lin_obs += 1
        self.num_obs += 1
        self.num_active_dual_vars.append(0)

    def add_nonlinear_obstacle(self, init_position, A_matrix, gamma,
                 w_mean_vec, w_cov_mat, radius):
        """
        input: init_position: initial position of nonlinear obstacle-to-add in
            the environment
        input: A_matrix: A matrix of nonlinear obstacle-to-add
        input: gamma: turning radius of nonlinear obstacle-to-add
        input: w_mean_vec: mean disturbance signal of nonlinear obstacle-to-add
        input: w_cov_mat: covariance matrix of disturbance signal of nonlinear
            obstacle-to-add
        input: radius: radius of nonlinear obstacle-to-add
        """
        self.nonlin_obs_list.append(NonlinearObstacle(init_position, A_matrix, gamma,
                 w_mean_vec, w_cov_mat, radius))
        self.num_nonlin_obs += 1
        self.num_obs += 1
        self.num_active_dual_vars.append(0)

    def get_concatenated_matrices(self):
        """
        Compute the matrices Z and H such that the concatenated state vector
            X = [x_1 x_2 .... x_T]
        can be expressed in terms of the concatenated input
            U = [u_0 u_1 .... u_{T-1}].
        Specifically, we have
            X = Z x_0 + H U
        return: Z: concatenated state matrix
        return: H: concatenated control matrix
        """

        # For ease of notation in what follows.
        state_dim = self.rob_state_dim
        input_dim = self.rob_input_dim
        plan_hor = self.planning_horizon
        state_mat = self.rob_A_mat
        input_mat = self.rob_B_mat

        # Describe the dynamics as X = Z x_0 + H U (X excludes current state)
        # Z matrix [A   A^2 .... A^T]
        Z = np.zeros((plan_hor * state_dim, state_dim))
        for time_ind in range(plan_hor):
            Z[time_ind*state_dim:(time_ind+1)*state_dim, :] = \
                np.linalg.matrix_power(state_mat, time_ind+1)

        # H matrix via flipped controllability matrices
        # flipped_controllability_matrix is [A^(T-1)B, A^(T-2)B, ... , AB, B]
        flipped_controllability_matrix = \
            np.zeros((input_dim, input_dim*plan_hor))
        for time_ind in range(plan_hor):
            flip_time_index = plan_hor - 1 - time_ind
            flipped_controllability_matrix[:, flip_time_index*input_dim:
                (flip_time_index + 1)*input_dim] = \
                    np.linalg.matrix_power(state_mat, time_ind)@input_mat
        H = np.tile(flipped_controllability_matrix, (plan_hor, 1))
        for time_ind in range(plan_hor - 1):
            zeroed_indices = (plan_hor - time_ind - 1) * input_dim
            H[time_ind * state_dim:(time_ind + 1) * state_dim,
                (time_ind + 1) * input_dim:] = \
                    np.zeros((state_dim, zeroed_indices))

        return Z, H

    def solve_optim_prob_and_update(self):
        """
        return:
        """
        #TODO: finish this :)


class LinearObstacle:

    # FOLLOWS DYNAMICS: x(k+1) = Ax(k) + Fw(k), w(k)~N(w_mean,w_cov)

    def __init__(self, obs_init_position, obs_A_matrix, obs_F_matrix,
                 obs_w_mean_vec, obs_w_cov_mat, obs_radius, rob_state_dim):
        """
        input: obs_init_position: initial position of the obstacle
        input: obs_A_matrix: matrix mapping current state to next state
        input: obs_F_matrix: matrix mapping disturbance to next state
        input: obs_w_mean_vec: mean value of disturbance signal
        input: obs_w_cov: covariance matrix of disturbance signal
        input: obs_radius: radius of the obstacle
        input: rob_state_dim: dimension of the robot / obstacle environment
        """
        self.A_matrix = obs_A_matrix
        self.F_matrix = obs_F_matrix
        self.w_mean_vec = obs_w_mean_vec
        self.w_cov_mat = obs_w_cov_mat
        self.radius = obs_radius

        # Distinguish between what the actual obstacle position is and
        # what the robot believes it to be when planning its trajectory
        self.sim_position = obs_init_position
        self.act_position = obs_init_position

        # Information about the shape of the obstacle's uncertainty
        # ellipsoid at the current time step (call this the obstacle's
        # q matrix)
        self.q_mat = np.zeros((rob_state_dim, rob_state_dim))

    def update_obstacle_positions(self, robot_act_pos):
        """
        input: robot_act_pos: actual position of robot in the environment, given
            by oracle (only queried when specifically chosen to make an observation of)
        input: make_obs_about: True if we are to make an observation about this obstacle
            at the current time step (most relevant AND at interval)
        """

        # First, update the actual position of the obstacle in the environment
        self.act_position = robot_act_pos

        # Now, if we are to make an observation about the obstacle, update the
        # simulation position to the actual
        #TODO: finish this :)


class NonlinearObstacle:

    # FOLLOWS DYNAMICS OF DUBIN'S VEHICLE WITH FIXED TURNING RATE \gamma

    def __init__(self, init_position, A_matrix, gamma,
                 w_mean_vec, w_cov_mat, radius, rob_state_dim):
        """
        input: init_position: initial position of the obstacle
        input: A_matrix: matrix mapping current state to next state
        input: gamma: fixed turning rate of the nonlinear obstacle
        input: w_mean_vec: mean value of disturbance signal
        input: w_cov: covariance matrix of disturbance signal
        input: radius: radius of the obstacle
        nput: rob_state_dim: dimension of the robot / obstacle environment
        """
        self.A_matrix = A_matrix
        self.gamma = gamma
        self.w_mean_vec = w_mean_vec
        self.w_cov_mat = w_cov_mat
        self.radius = radius

        # Distinguish between what the actual obstacle position is and
        # what the robot believes it to be when planning its trajectory
        self.sim_position = init_position
        self.act_position = init_position

        # Information about the shape of the obstacle's uncertainty
        # ellipsoid at the current time step (call this the obstacle's
        # q matrix)
        self.q_mat = np.zeros((rob_state_dim, rob_state_dim))

    def update_obstacle_positions(self):
        """
        input:
        """
        # TODO: finish this :)






















