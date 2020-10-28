"""
    Description:    Construct a class to model the safety and attention environment
    Author:         Michael Hibbard, Abraham Vinod
    Created:        September 2020
"""
import numpy as np
from scipy.linalg import block_diag
from numpy import linalg as LA

# Additional scripts to import
from linear_dynamics_auxilliary_functions import \
    (ellipsoid_level_function,check_collisions,
    get_mu_sig_over_time_hor, get_Q_mats_over_time_hor,
    get_Qplus_mats_over_time_hor, propagate_linear_obstacles,
    propagate_nonlinear_obstacles, get_mu_sig_theta_over_time_hor)
from dc_based_motion_planner import \
    (create_dc_motion_planner, create_cvxpy_motion_planner_obstacle_free,
    solve_dc_motion_planning)
from dc_motion_planning_ecos import \
    (dc_motion_planning_call_ecos, stack_params_for_ecos,
    set_up_independent_ecos_params)


class RobotSaAEnvironment:

    def __init__(self, target_state, target_tolerance, gauss_lev_param, planning_horizon,
                 rob_init_pos, rob_A_mat, rob_B_mat, obs_field_of_view_rad, obs_interval,
                 rob_state_max, rob_input_max, sampling_time):
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
        input: sampling_time: The time between subsequent planning steps (k) and (k+1)
        """
        # Set the initial parameters
        self.sampling_time = sampling_time
        self.target_state = target_state
        self.target_tolerance = target_tolerance
        self.beta = gauss_lev_param
        self.planning_horizon = planning_horizon
        self.rob_pos = rob_init_pos
        self.rob_A_mat = rob_A_mat
        self.rob_B_mat = self.sampling_time*rob_B_mat
        self.obs_field_of_view_rad = obs_field_of_view_rad
        self.obs_interval = obs_interval
        self.rob_state_max = rob_state_max
        self.rob_input_max = rob_input_max

        # Sets to false once we have reached the target state
        self.continue_condition = True

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
                 w_mean_vec, w_cov_mat, radius, self.rob_state_dim, self.sampling_time))
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
                 w_mean_vec, w_cov_mat, radius, self.rob_state_dim, self.sampling_time))
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
                    np.matmul(np.linalg.matrix_power(state_mat, time_ind),input_mat)
        H = np.tile(flipped_controllability_matrix, (plan_hor, 1))
        for time_ind in range(plan_hor - 1):
            zeroed_indices = (plan_hor - time_ind - 1) * input_dim
            H[time_ind * state_dim:(time_ind + 1) * state_dim,
                (time_ind + 1) * input_dim:] = \
                    np.zeros((state_dim, zeroed_indices))

        return Z, H

    def solve_optim_prob_and_update(self):
        """
        input: robot_act_pos: the ACTUAL position of the robotic agent in
        return: rob_traj: set of waypoints for the robotic agent to follow
        return: obs_traj: next waypoint for each of the obstacles to travel to
        """

        # Update the number of time steps considered
        self.total_time_steps += 1

        # Solve the linear dynamics planning problem of the robotic
        # agent for the current time step
        self.nominal_trajectory = self.linear_dynamics_planning()

        # Now, iterate through the obstacles and update their
        # simulated and actual positions
        obs_index = 0

        # Handle updates differently if we can make an observation this
        # time step
        if self.total_time_steps % self.obs_interval == 0:
            for j in range(self.num_lin_obs):
                # Access the current obstacle
                cur_obs = self.lin_obs_list[j]
                # If this obstacle was deemed to be the most relevant, attempt to make
                # an observation about it
                if obs_index == np.argmax(self.num_active_dual_vars):
                    cur_obs.update_obstacle_position(1,self.rob_pos,self.obs_field_of_view_rad)
                    obs_index += 1
                else:
                    cur_obs.update_obstacle_position(0,self.rob_pos,self.obs_field_of_view_rad)
                    obs_index += 1
            # Repeat the process for the nonlinear obstacles
            for j in range(self.num_nonlin_obs):
                cur_obs = self.nonlin_obs_list[j]
                if obs_index == np.argmax(self.num_active_dual_vars):
                    cur_obs.update_obstacle_position(1,self.rob_pos,self.obs_field_of_view_rad)
                    obs_index += 1
                else:
                    cur_obs.update_obstacle_position(0,self.rob_pos,self.obs_field_of_view_rad)
                    obs_index += 1
        else:
            # Never give the opportunity to make an observation about an obstacle
            for j in range(self.num_nonlin_obs):
                cur_obs = self.lin_obs_list[j]
                cur_obs.update_obstacle_position(0,self.rob_pos,self.obs_field_of_view_rad)
            for j in range(self.num_nonlin_obs):
                cur_obs = self.nonlin_obs_list[j]
                cur_obs.update_obstacle_position(0,self.rob_pos,self.obs_field_of_view_rad)

        if np.linalg.norm(self.rob_pos-self.target_state) <= self.target_tolerance:
            self.continue_condition = False

        return self.nominal_trajectory

    def linear_dynamics_planning(self):
        """
        Given state of robotic agent's environment, construct a new trajectory
        to follow (i.e. generate a set of waypoints to send to the agent)
        """

        # Setup the CVXPY problem as well as the relevant parameters, variables
        # cvxpy_motion_planner - CVXPY optimization problem with
        #     cvxpy_robot_current_state: CVXPY parameter that can be updated by the
        #       receeding horizon control framework
        #     cvxpy_obstacle_free_polytope_A, cvxpy_obstacle_free_polytope_b:
        #       CVXPY parameter that indicates a subset of the
        #       free space that the robot can move without
        #       colliding into obstacles
        #     cvxpy_robot_trajectory, cvxpy_robot_input_sequence:
        #       CVXPY variable that will be updated by the solver
        cvxpy_motion_planner, cvxpy_robot_current_state, \
        cvxpy_robot_trajectory, cvxpy_robot_input_sequence = \
            create_cvxpy_motion_planner_obstacle_free(self.Z, self.H,
                                    self.planning_horizon, self.target_state,
                                    self.rob_state_max, self.rob_input_max)

        # Set up optimization problem to run in ECOS...

        # Compute matrix needed in setting up problem
        obs_time_index = 0
        for obs_index in range(self.num_obs):
            for t_step in range(self.planning_horizon):
                if obs_time_index == 0:
                    mult_matrix_stack = np.ones((1, self.rob_state_dim))
                    obs_time_index += 1
                else:
                    mult_matrix_stack = block_diag(
                        mult_matrix_stack, np.ones((1, self.rob_state_dim)))

        # Set up the problem parameters to call ECOS
        A_ecos, G_linear_partial_stack_ecos, G_quadratic_ecos, \
        h_linear_partial_stack_ecos, h_quadratic_ecos, target_tile_ecos = \
            set_up_independent_ecos_params(self.num_obs, self.planning_horizon,
                                           self.rob_state_dim, self.rob_input_dim,
                                           self.H, self.target_state,
                                           self.rob_state_max, self.rob_input_max)

        # Propagate the linear obstacles over the planning horizon

        # First, propagate one time step in the future
        linear_obs_mu_bars_t, linear_obs_sig_bars_t = \
            propagate_linear_obstacles(self.num_lin_obs, self.lin_obs_list)

        # Now, propagate the linear obstacles over the time horizon
        linear_obs_mu_bars_time_hor, linear_obs_sig_bars_time_hor = \
            get_mu_sig_over_time_hor(self.planning_horizon, self.num_lin_obs, linear_obs_mu_bars_t,
                            linear_obs_sig_bars_t, self.lin_obs_list)

        # Use this information to construct the matrix Q for each obstacle,
        # which is the ellipsoidal level set centered around the mean position
        # sliced at the level corresponding to the parameter beta
        linear_obs_Q_mat_time_hor = get_Q_mats_over_time_hor(self.planning_horizon,
            self.num_lin_obs, self.beta, self.lin_obs_list, linear_obs_sig_bars_time_hor)

        # Now, propagate the nonlinear obstacles over the planning
        # horizon (assume that the turning rate is fixed)

        # First, propagate one time step in the future
        nonlinear_obs_mu_bars_t, nonlinear_obs_theta_t, nonlinear_obs_sig_bars_t = \
            propagate_nonlinear_obstacles(self.num_nonlin_obs, self.sampling_time,
              self.nonlin_obs_list)

        # Now, propagate the nonlinear obstacles over the remaining time horizon
        nonlinear_obs_mu_bars_time_hor, nonlinear_obs_sig_bars_time_hor, \
        nonlinear_obs_theta_time_hor = get_mu_sig_theta_over_time_hor(
            self.planning_horizon, self.num_nonlin_obs, nonlinear_obs_mu_bars_t,
            nonlinear_obs_sig_bars_t, self.nonlin_obs_list, nonlinear_obs_theta_t,
            self.sampling_time)

        # Again, use this information to construct the Q matrix (in the same way as for the
        # linear obstacles)
        nonlinear_obs_Q_mat_time_hor = get_Q_mats_over_time_hor(
            self.planning_horizon, self.num_nonlin_obs, self.beta,
            self.nonlin_obs_list, nonlinear_obs_sig_bars_time_hor)

        # Concatenate the linear and nonlinear obstacle ellipsoids. From here on out,
        # they are functionally the same. For now, assume that we will always have at
        # least one linear obstacle.
        if self.num_nonlin_obs != 0:
            obs_Q_mat_time_hor = np.vstack(
                (linear_obs_Q_mat_time_hor, nonlinear_obs_Q_mat_time_hor))
            obs_mu_bars_time_hor = np.vstack(
                (linear_obs_mu_bars_time_hor, nonlinear_obs_mu_bars_time_hor))
            obs_rad_vector = np.concatenate(
                (np.array([self.lin_obs_list[j].radius for j in range(self.num_lin_obs)]),
                 np.array([self.nonlin_obs_list[j].radius for j in range(self.num_nonlin_obs)])),
                axis=0)
        else:
            obs_Q_mat_time_hor = linear_obs_Q_mat_time_hor
            obs_mu_bars_time_hor = linear_obs_mu_bars_time_hor
            obs_rad_vector = np.array([self.lin_obs_list[j].radius for j in range(self.num_lin_obs)])

        # Solve the obstacle-free optimal control problem to determine a nominal
        # path for the robot to follow.
        cvxpy_robot_current_state.value = self.rob_pos
        cvxpy_motion_planner.solve(solver='ECOS')
        # print('CVX status: {:s} | Solution time: {:1.4f} seconds'.format(
        #     cvxpy_motion_planner.status,
        #     cvxpy_motion_planner.solver_stats.solve_time))
        robot_RHC_trajectory_initial_solution = cvxpy_robot_trajectory.value
        robot_initial_trajectory_state_x_time = np.reshape(
            robot_RHC_trajectory_initial_solution, (self.planning_horizon, self.rob_state_dim)).T

        # Construct the obstacle Q+ matrix. This matrix is an outer approximation
        # of the Minkowski sum of the obstacle's uncertainty ellipsoid
        # with its rigid body shape (assumed to be a circle). The outer approximation
        # is tight in our direction of interest. We take this direction to be that of
        # the difference between the obstacle mean position and the robot's position
        obs_Qplus_mat_time_hor = get_Qplus_mats_over_time_hor(
            self.planning_horizon, self.num_obs, obs_mu_bars_time_hor, obs_Q_mat_time_hor,
            robot_initial_trajectory_state_x_time, obs_rad_vector, self.rob_state_dim)

        # Using these tight outer approximations, determine if the robot's nominal
        # trajectory intersects any of these matrices
        collision_flag_list = check_collisions(
            robot_initial_trajectory_state_x_time, obs_mu_bars_time_hor,
            obs_Qplus_mat_time_hor, self.num_obs)

        # If there were collisions, use difference-of-convex procedure to adjust
        # the nominal trajectory
        if any(collision_flag_list):

            # Set up some parameters here to subsequently feed into the ECOS solver
            obs_Qplus_stack_all, obs_mu_bar_stack_all = stack_params_for_ecos(
                self.num_obs, self.planning_horizon, obs_mu_bars_time_hor,
                obs_Qplus_mat_time_hor)

            # Solve the DC planning problem
            obs_cons_dual_variables, iteration_count, solver_times, \
            num_active_duals, traj, input = \
                dc_motion_planning_call_ecos(
                    self.num_obs, self.planning_horizon, self.rob_state_dim, self.rob_input_dim,
                    self.Z, self.rob_pos, robot_RHC_trajectory_initial_solution,
                    obs_mu_bar_stack_all, obs_Qplus_stack_all, mult_matrix_stack, A_ecos,
                    G_linear_partial_stack_ecos, G_quadratic_ecos, h_linear_partial_stack_ecos,
                    h_quadratic_ecos, target_tile_ecos, tau_max=np.array([10000]),
                    mu=10, delta=1e-3, tau=1)

            # Check to see if the updated trajectory found through DC program
            # successfully removed collisions from the trajectory
            robot_RHC_trajectory_state_x_time = np.reshape(
                traj, (self.planning_horizon, self.rob_state_dim)).T
            collision_flag_list = check_collisions(
                robot_RHC_trajectory_state_x_time, obs_mu_bars_time_hor,
                obs_Qplus_mat_time_hor, self.num_obs)
            if any(collision_flag_list):
                raise RuntimeError('\n\nWARNING!!! Collision not resolved!\n\n')
            else:
                # Use the resolved solution
                robot_RHC_trajectory = traj
                robot_input_sequence = input

            # Store information about dual variables at current time step
            obs_cons_dual_variables = np.asarray(obs_cons_dual_variables)

            # Set the dual variables to be 1 if they are nonzero
            # (above tolerance) and zero otherwise
            obs_cons_dual_variables[obs_cons_dual_variables >= 1e-4] = 1
            obs_cons_dual_variables[obs_cons_dual_variables < 1e-4] = 0

            # Reshape, sum over time horizon, add to the overall counter
            obs_cons_dual_variables = np.reshape(obs_cons_dual_variables,
                                                 (self.num_obs, self.planning_horizon))
            obs_cons_dual_vars_sum = np.sum(obs_cons_dual_variables, 1)
            self.num_active_dual_vars += obs_cons_dual_vars_sum

        else:
            # If no collisions detected, use nominal trajectory
            robot_RHC_trajectory_state_x_time = robot_initial_trajectory_state_x_time

        return robot_RHC_trajectory_state_x_time


class LinearObstacle:

    # FOLLOWS DYNAMICS: x(k+1) = Ax(k) + Fw(k), w(k)~N(w_mean,w_cov)

    def __init__(self, obs_init_position, obs_A_matrix, obs_F_matrix,
                 obs_w_mean_vec, obs_w_cov_mat, obs_radius,
                 rob_state_dim, sampling_time):
        """
        input: obs_init_position: initial position of the obstacle
        input: obs_A_matrix: matrix mapping current state to next state
        input: obs_F_matrix: matrix mapping disturbance to next state
        input: obs_w_mean_vec: mean value of disturbance signal
        input: obs_w_cov: covariance matrix of disturbance signal
        input: obs_radius: radius of the obstacle
        input: rob_state_dim: dimension of the robot / obstacle environment
        """
        self.sampling_time = sampling_time
        self.A_matrix = obs_A_matrix
        self.F_matrix = self.sampling_time*obs_F_matrix
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
        self.sig_mat = np.zeros((rob_state_dim, rob_state_dim))

    def update_obstacle_position(self,query_obs_pos,rob_act_pos,
                                 rob_field_of_view_rad):
        """
        input: robot_act_pos: actual position of robot in the environment, given
            by oracle (only queried when specifically chosen to make an observation of)
        input: query_obs_pos: True if we are to make an observation about this obstacle
            at the current time step (most relevant AND at interval)
        input: rob_act_pos: position of the robotic agent in the environment
        input: rob_field_of_view_rad: radius for which the agent can make an observation
            of an obstacle if they are within this distance
        """

        # Sample noise given system parameters
        noise_cur_t_step = np.random.normal(
            self.w_mean_vec, self.w_cov_mat, 1)

        # Push through the linear dynamics to obtain obstacle's
        # ACTUAL position
        self.act_position = self.A_matrix*self.act_position \
            + self.F_matrix*noise_cur_t_step

        # Update the robotic agent's understanding of the
        # obstacle's position (the SIMULATION position, uncertainty ellipsoid)
        # TODO: change this to be a boolean?
        # 0 / FALSE: do not query the obstacle position at this time step
        # 1 / TRUE: attempt to make an observation about the obstacle's position
        if query_obs_pos == 0:
            # If not making an observation, push current position and
            # uncertainty through the system dynamics
            A_mat = self.A_matrix
            F_mat = self.F_matrix
            self.sim_position = np.matmul(A_mat,self.sim_position) + np.matmul(F_mat,self.w_mean_vec)
            self.sig_mat = np.matmul(A_mat,self.sig_mat,np.transpose(A_mat)) \
                           + np.matmul(F_mat,self.w_cov_mat,np.transpose(F_mat))
        else:
            if np.norm(rob_act_pos - self.act_position) <= rob_field_of_view_rad:
                # Assume a perfect observation
                self.sim_position = self.act_position
                self.sig_mat = np.zeros(np.shape(self.sig_mat))
            else:
                # If the observation is unsuccessful, proceed as if we
                # had not attempted to make an observation of the obstacle
                A_mat = self.A_matrix
                F_mat = self.F_matrix
                self.sim_position = np.matmul(A_mat,self.sim_position) + np.matmul(F_mat,self.w_mean_vec)
                self.sig_mat = np.matmul(A_mat,self.sig_mat,np.transpose(A_mat)) \
                               + np.matmul(F_mat,self.w_cov_mat,np.transpose(F_mat))


class NonlinearObstacle:

    # FOLLOWS DYNAMICS OF DUBIN'S VEHICLE WITH FIXED TURNING RATE, \gamma

    def __init__(self, init_position, A_matrix, gamma, w_mean_vec,
                 w_cov_mat, radius, rob_state_dim, theta_init, sampling_time):
        """
        input: init_position: initial position of the obstacle
        input: A_matrix: matrix mapping current state to next state
        input: gamma: fixed turning rate of the nonlinear obstacle
        input: w_mean_vec: mean value of disturbance signal
        input: w_cov: covariance matrix of disturbance signal
        input: radius: radius of the obstacle
        input: theta: the initial heading angle of the nonlinear obstacle
        nput: rob_state_dim: dimension of the robot / obstacle environment
        """
        self.A_matrix = A_matrix
        self.gamma = gamma
        self.w_mean_vec = w_mean_vec
        self.w_cov_mat = w_cov_mat
        self.radius = radius
        self.theta = theta_init
        self.sampling_time = sampling_time

        # Distinguish between what the actual obstacle position is and
        # what the robot believes it to be when planning its trajectory
        self.sim_position = init_position
        self.act_position = init_position

        # Information about the shape of the obstacle's uncertainty
        # ellipsoid at the current time step (call this the obstacle's
        # q matrix)
        self.sig_mat = np.zeros((rob_state_dim, rob_state_dim))

    def update_obstacle_act_position(self,query_obs_pos,rob_act_pos,
                                     rob_field_of_view_rad):
        """
        input: query_obs_pos: True if we are to make an observation about this obstacle
            at the current time step (most relevant AND at interval)
        input: rob_act_pos: position of the robotic agent in the environment
        input: rob_field_of_view_rad: radius for which the agent can make an observation
            of an obstacle if they are within this distance
        """

        # Sample noise for turning rate given system parameters
        noise_cur_t_step = np.random.normal(
            self.w_mean_vec, self.w_cov_mat, 1)

        # For notational convenience
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # Push through the nonlinear dynamics
        B_0 = self.sampling_time*np.array([sin_theta, cos_theta])
        self.act_position = self.A_matrix*self.act_position\
                                + B_0*noise_cur_t_step
        self.theta = self.theta + self.sampling_time * self.gamma

        # Update the robotic agent's understanding of the
        # obstacle's position (the SIMULATION position, uncertainty ellipsoid)
        # 0 / FALSE: do not query the obstacle position at this time step
        # 1 / TRUE: attempt to make an observation about the obstacle's position
        if query_obs_pos == 0:
            # If not making an observation, push current position and
            # uncertainty through the system dynamics
            A_mat = self.A_matrix
            self.sim_position = np.matmul(A_mat,self.sim_position) + np.matmul(B_0,self.w_mean_vec)
            self.sig_mat = np.matmul(A_mat,self.sig_mat,np.transpose(A_mat)) \
                           + np.matmul(B_0,self.w_cov_mat,np.transpose(B_0))
        else:
            if np.norm(rob_act_pos - self.act_position) <= rob_field_of_view_rad:
                # Assume a perfect observation
                self.sim_position = self.act_position
                self.sig_mat = np.zeros(np.shape(self.sig_mat))
            else:
                # If the observation is unsuccessful, proceed as if we
                # had not attempted to make an observation of the obstacle
                A_mat = self.A_matrix
                self.sim_position = np.matmul(A_mat,self.sim_position) + np.matmul(B_0,self.w_mean_vec)
                self.sig_mat = np.matmul(A_mat,self.sig_mat,np.transpose(A_mat)) \
                               + np.matmul(B_0,self.w_cov_mat,np.transpose(B_0))
