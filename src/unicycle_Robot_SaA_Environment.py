"""
    Description:    Construct a class to model the safety and attention environment
    Author:         Michael Hibbard, Abraham Vinod
    Created:        September 2020
"""
import numpy as np
import time as time
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex"
})
from matplotlib.patches import Polygon, Circle, Ellipse
from matplotlib.collections import PatchCollection
from numpy import linalg as LA
import math

# Additional scripts to import
from linear_dynamics_auxilliary_functions import \
    (ellipsoid_level_function,check_collisions,
    get_mu_sig_over_time_hor, get_Q_mats_over_time_hor,
    get_Qplus_mats_over_time_hor, propagate_linear_obstacles,
    propagate_nonlinear_obstacles, get_mu_sig_theta_over_time_hor)
from unicycle_dc_motion_planning_ecos import \
    (ecos_unicycle_shared_cons, solve_obs_free_ecos_unicycle,
     stack_params_for_ecos, dc_motion_planning_call_ecos_unicycle)
from ecosqp_file import ecosqp

# Mathematic functions
from math import atan2

# Parallel processing
from itertools import repeat

# To unpack iterables for parallel function calls
from functools import wraps

class RobotSaAEnvironment:

    def __init__(self, target_state, target_tolerance, gauss_lev_param, planning_horizon,
                 rob_init_pos, rob_A_mat, obs_field_of_view_rad, obs_interval,
                 rob_state_x_max, rob_state_y_max, sampling_time, observation_strategy,
                 max_heading_view, rob_max_velocity, rob_max_turn_rate,
                 rob_agg_turn_rate, most_rel_obs_ind, num_turning_rates,
                 turning_rates_array, rob_heading_ang):
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
        input: rob_state_x_max: absolute value of maximum distance in the x direction the robot
            can be from the origin
        input: rob_state_y_max: absolute value of maximum distance in the y direction the robot
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
        self.obs_field_of_view_rad = obs_field_of_view_rad
        self.obs_interval = obs_interval
        self.rob_state_x_max = rob_state_x_max
        self.rob_state_y_max = rob_state_y_max
        self.obs_strat = observation_strategy
        self.max_heading_view = max_heading_view
        self.rob_max_velocity = rob_max_velocity
        self.rob_max_turn_rate = rob_max_turn_rate
        self.rob_agg_turn_rate = rob_agg_turn_rate
        self.most_rel_obs_ind = most_rel_obs_ind
        self.num_turning_rates = num_turning_rates
        self.turning_rates_array = turning_rates_array
        self.heading_angle = rob_heading_ang
        self.best_gamma_ind = None
        self.solve_times = []

        # Plot the initial map
        safe_set_polygon = Polygon(np.array([[-self.rob_state_x_max, -self.rob_state_y_max],
            [self.rob_state_x_max, -self.rob_state_y_max],[self.rob_state_x_max, self.rob_state_y_max],
            [-self.rob_state_x_max, self.rob_state_y_max]]), alpha=1, edgecolor='k', fill=False, zorder=0)
        self.fig, self.ax = self.plot_initial_map(safe_set_polygon)
        plt.draw()
        plt.pause(0.001)

        # Sets to false once we have reached the target state
        self.continue_condition = True

        # Setup some robot parameters based on inputs. Note that, in the case of unicycle
        # dynamics, we will have an input dimension of 1 (the velocity control)
        self.rob_state_dim = self.rob_A_mat.shape[1]
        self.rob_input_dim = 1

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
        self.dual_var_metric = []
        self.nominal_trajectory = np.zeros(
            (self.rob_state_dim, self.planning_horizon))
        self.heading_angle_sequence = np.zeros(self.planning_horizon)

        # Total number of obstacles, linear and nonlinear
        self.num_obs = 0

        # Overall time step in the system
        self.total_time_steps = 0

    def plot_initial_map(self,safe_set_polygon):

        # Plotting the map
        fig = plt.figure()
        ax = fig.gca()
        # Plot the safe set
        ax.add_patch(safe_set_polygon)
        # Plot the initial and target states
        # ax.scatter(self.rob_pos[0], self.rob_pos[1], 100, color='b',
        #            label='Initial state')
        # ax.scatter(self.target_state[0], self.target_state[1], 100, color='g',
        #           label='Target state')
        ax.scatter(self.target_state[0], self.target_state[1], 100, marker='*', color='y')

        sta_lim_tup_x = [-self.rob_state_x_max, self.rob_state_x_max]
        sta_lim_tup_y = [-self.rob_state_y_max, self.rob_state_y_max]
        ax.set_xlim(sta_lim_tup_x)
        ax.set_ylim(sta_lim_tup_y)
        ax.set_aspect('equal')
        ax.axis('off')
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #leg = plt.legend(bbox_to_anchor=(1.01, 0.5))
        plt.tight_layout()

        return fig, ax

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
        self.dual_var_metric.append(0)

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
        self.dual_var_metric.append(0)

    def get_concatenated_matrices(self, input_matrix_array):
        """
        Compute the matrices Z and H such that the concatenated state vector
                X = [x_1 x_2 .... x_T]
        can be expressed in terms of the concatenated input
                U = [u_0 u_1 .... u_{T-1}].
        Specifically, we have
                X = Z x_0 + H U

        This permits analysis of discrete-time systems x_{t+1} = A x_t + B u_t

        :param state_matrix: System matrix A
        :param input_matrix: Input matrix B
        :param time_horizon:
        :return: concatenated matrices Z and H
        """

        state_matrix = self.rob_A_mat
        planning_horizon = self.planning_horizon
        rob_state_dim = self.rob_state_dim
        rob_input_dim = self.rob_input_dim

        # Describe the dynamics as X = Z x_0 + H U (X excludes current state)
        # Z matrix [A   A^2 .... A^T]
        Z = np.zeros((planning_horizon * rob_state_dim, rob_state_dim))
        for time_index in range(planning_horizon):
            Z[time_index * rob_state_dim:
              (time_index + 1) * rob_state_dim, :] = \
                np.linalg.matrix_power(state_matrix, time_index + 1)

        # H matrix via flipped controllability matrices
        # flipped_controllability_matrix is [A^(T-1)B, A^(T-2)B, ... , AB, B]
        flipped_controllability_matrix = \
            np.zeros((rob_state_dim, rob_input_dim * planning_horizon))
        for time_index in range(planning_horizon):
            flip_time_index = planning_horizon - 1 - time_index
            cur_input_matrix = input_matrix_array[flip_time_index]
            flipped_controllability_matrix[:,
            flip_time_index * rob_input_dim:
            (flip_time_index + 1) * rob_input_dim] = \
                np.matmul(np.linalg.matrix_power(state_matrix, time_index), cur_input_matrix)
        H = np.tile(flipped_controllability_matrix, (planning_horizon, 1))
        for time_index in range(planning_horizon - 1):
            zeroed_indices = (planning_horizon - time_index - 1) * rob_input_dim
            H[time_index * rob_state_dim:
              (time_index + 1) * rob_state_dim,
            (time_index + 1) * rob_input_dim:] = \
                np.zeros((rob_state_dim, zeroed_indices))

        return Z, H

    def solve_optim_prob_and_update(self,pool):
        """
        input: robot_act_pos: the ACTUAL position of the robotic agent in
        return: rob_traj: set of waypoints for the robotic agent to follow
        return: obs_traj: next waypoint for each of the obstacles to travel to

        NOTE: obs_act_positions is obtained from querying the ros velocity controller
            associated with that obstacle!

        """

        # Update the number of time steps considered
        self.total_time_steps += 1

        # Solve the linear dynamics planning problem of the robotic
        # agent for the current time step
        self.nominal_trajectory, self.heading_angle_sequence, self.best_gamma_ind = self.linear_dynamics_planning(pool)

        return self.nominal_trajectory, self.heading_angle_sequence

    def update_obs_positions_and_plots(self, obs_act_positions, obs_cam_list):
        # Now, iterate through the obstacles and update their
        # simulated and actual positions
        obs_index = 0

        # Handle updates differently if we can make an observation this
        # time step
        if self.total_time_steps % self.obs_interval == 0:
            for j in range(self.num_lin_obs):
                # Access the current obstacle
                cur_obs = self.lin_obs_list[j]
                # Access the true state of the obstacle
                cur_obs_physical_state = obs_act_positions[j]

                # Make an observation about the obstacle if it is within the field
                # of view of the camera
                if j in obs_cam_list:
                    query_obs_pos = True
                    cur_obs.update_obstacle_position(query_obs_pos,self.rob_pos,self.obs_field_of_view_rad,
                                                 cur_obs_physical_state)
                else:
                    query_obs_pos = False
                    cur_obs.update_obstacle_position(query_obs_pos, self.rob_pos, self.obs_field_of_view_rad,
                                                     cur_obs_physical_state)
                obs_index += 1
            # Repeat the process for the nonlinear obstacles
            # TODO: below is deprecated, but for the current experiments we only have linear obstacles.
            for j in range(self.num_nonlin_obs):
                cur_obs = self.nonlin_obs_list[j]
                if obs_index == np.argmax(self.dual_var_metric):
                    cur_obs.update_obstacle_position(True,self.rob_pos,self.obs_field_of_view_rad)
                    obs_index += 1
                else:
                    cur_obs.update_obstacle_position(True,self.rob_pos,self.obs_field_of_view_rad)
                    obs_index += 1
        else:
            # Never give the opportunity to make an observation about an obstacle
            for j in range(self.num_lin_obs):
                cur_obs = self.lin_obs_list[j]
                cur_obs_physical_state = obs_act_positions[j]
                cur_obs.update_obstacle_position(False,self.rob_pos,self.obs_field_of_view_rad,
                                                 cur_obs_physical_state)
            for j in range(self.num_nonlin_obs):
                cur_obs = self.nonlin_obs_list[j]
                cur_obs.update_obstacle_position(False,self.rob_pos,self.obs_field_of_view_rad)

        if np.linalg.norm(self.rob_pos-self.target_state) <= self.target_tolerance:
            self.continue_condition = False

        # print('----------')
        # print(self.dual_var_metric)
        # for obs in range(2):
        #     print(self.lin_obs_list[obs].sig_mat)

        # Reset the dual variable metric counter to all zeros
        self.dual_var_metric = [0 for i in range(len(self.dual_var_metric))]

        # print(self.dual_var_metric)
        # print('----------')

        return

    def linear_dynamics_planning(self,pool):
        """
        Given state of robotic agent's environment, construct a new trajectory
        to follow (i.e. generate a set of waypoints to send to the agent)
        """

        np.random.seed(0)

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

        # Set up problem parameters that are shared when solving in ECOS
        target_tile_ecos = np.tile(np.reshape(self.target_state,
                                              (self.rob_state_dim, 1)), (self.planning_horizon, 1))
        A_shared, b_shared, H_shared, f_shared = \
            ecos_unicycle_shared_cons(self.rob_state_dim,self.rob_input_dim,self.planning_horizon,
                self.rob_state_x_max,self.rob_state_y_max,self.rob_max_velocity,target_tile_ecos)

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

        # If a relevant obstacle has been identified, change the one-step turning rate
        # to accommodate the observation
        if self.most_rel_obs_ind is not None:
            # Determine the position of the most relevant obstacle relative to the agent's current position
            # at the next time step
            most_rel_obs_pos = obs_mu_bars_time_hor[self.most_rel_obs_ind][0]
            rob_to_obs_vec = most_rel_obs_pos - self.rob_pos
            rob_to_obs_ang = atan2(rob_to_obs_vec[1], rob_to_obs_vec[0])
            ang_dif = rob_to_obs_ang - self.heading_angle

            hard_turn_ang = self.rob_max_turn_rate
            # Depending on the quadrant relative to the robot, either turn hard left, hard right, or straight forward.
            if -self.max_heading_view <= ang_dif <= self.max_heading_view:
                # If the most relevant obstacle is in front of the robot, don't need to make a "hard turn"
                init_turn_ang = 0
            elif self.max_heading_view <= ang_dif <= np.pi:
                # If the most relevant obstacle is to the back left of the robot, make a hard turn to the left
                init_turn_ang = self.sampling_time*hard_turn_ang
            else:
                init_turn_ang = -self.sampling_time*hard_turn_ang

        else:
            init_turn_ang = 0

        # rob_motion_plans_each_gamma, rob_input_sequence_each_gamma, rob_obs_func_val_each_gamma, \
        # rob_obs_func_dual_vals_each_gamma, obs_Qplus_mat_time_hor_each_gamma, rh_ang_time_hor_each_gamma = \
        #     pool.map(self.solve_motion_planning_prob,zip(self.turning_rates_array,
        #         repeat(target_tile_ecos),repeat(obs_mu_bars_time_hor),repeat(obs_Q_mat_time_hor),
        #         repeat(obs_rad_vector), repeat(mult_matrix_stack),repeat(A_shared), repeat(b_shared),
        #         repeat(H_shared), repeat(f_shared),repeat(init_turn_ang)))


        arguments = zip(self.turning_rates_array,repeat(target_tile_ecos),repeat(obs_mu_bars_time_hor),repeat(obs_Q_mat_time_hor),
                repeat(obs_rad_vector), repeat(mult_matrix_stack),repeat(A_shared), repeat(b_shared),
                repeat(H_shared), repeat(f_shared),repeat(init_turn_ang),
                ##########
                repeat(self.sampling_time),repeat(self.planning_horizon),repeat(self.heading_angle),
                repeat(self.rob_state_dim),repeat(self.rob_input_dim),repeat(self.rob_pos),
                repeat(self.num_obs),repeat(self.most_rel_obs_ind),repeat(self.rob_A_mat))

        # rob_motion_plans_each_gamma, rob_input_sequence_each_gamma, rob_obs_func_val_each_gamma, \
        # rob_obs_func_dual_vals_each_gamma, obs_Qplus_mat_time_hor_each_gamma, rh_ang_time_hor_each_gamma = \
        #     pool.map(solve_motion_planning_prob,iterable=arguments)
        
        sol_tic = time.time()
        outputs = pool.map(solve_motion_planning_prob, iterable=arguments)
        sol_toc = time.time()

        self.solve_times.append(sol_toc-sol_tic)

        # Extract the results of the parallel processes
        rob_motion_plans_each_gamma = [output[0] for output in outputs]
        rob_input_sequence_each_gamma = [output[1] for output in outputs]
        rob_obs_func_val_each_gamma = [output[2] for output in outputs]
        rob_obs_func_dual_vals_each_gamma = [output[3] for output in outputs]
        obs_Qplus_mat_time_hor_each_gamma = [output[4] for output in outputs]
        rh_ang_time_hor_each_gamma = [output[5] for output in outputs]

        # Check if no feasible trajectory was found
        if np.isnan(rob_obs_func_val_each_gamma).all():
            raise RuntimeError("\n Warning: No turning rate permits a feasible trajectory")

        # Determine the best value of the turning rate, gamma, based on the minimum
        # objective function observed
        best_gamma_ind = int(np.nanargmin(rob_obs_func_val_each_gamma))
        robot_RHC_trajectory = rob_motion_plans_each_gamma[best_gamma_ind]
        robot_RHC_trajectory_state_x_time = np.reshape(
            robot_RHC_trajectory, (self.planning_horizon, self.rob_state_dim)).T
        robot_input_sequence = rob_input_sequence_each_gamma[best_gamma_ind]
        obs_cons_dual_variables = rob_obs_func_dual_vals_each_gamma[best_gamma_ind]
        opt_gamma = self.turning_rates_array[best_gamma_ind]
        obs_Qplus_mat_time_hor = obs_Qplus_mat_time_hor_each_gamma[best_gamma_ind]
        best_rh_sequence = rh_ang_time_hor_each_gamma[best_gamma_ind]

        # Method 1: Use the total number of active dual variables
        if self.obs_strat == "bin":

            # Set the dual variables to be 1 if they are nonzero
            # (above tolerance) and zero otherwise
            obs_cons_dual_variables[obs_cons_dual_variables >= 1e-4] = 1
            obs_cons_dual_variables[obs_cons_dual_variables < 1e-4] = 0

            # Reshape, sum over time horizon, add to the overall counter
            obs_cons_dual_variables = np.reshape(obs_cons_dual_variables,
                (self.num_obs, self.planning_horizon))
            obs_cons_dual_vars_indic_sum = np.sum(obs_cons_dual_variables, 1)
            self.dual_var_metric += obs_cons_dual_vars_indic_sum

        # Method 2: Add up the sum of the dual variables,
        elif self.obs_strat == "sum":

            # Get rid of rounding errors due to numerical precision
            # DO NOT make them binary, though!
            obs_cons_dual_variables[obs_cons_dual_variables < 1e-4] = 0

            # Reshape, sum over time horizon, add to the overall counter
            obs_cons_dual_variables = np.reshape(obs_cons_dual_variables,
                (self.num_obs, self.planning_horizon))
            obs_cons_dual_vars_sum = np.sum(obs_cons_dual_variables, 1)
            self.dual_var_metric += obs_cons_dual_vars_sum

        # Currently, we only have these two methods.
        else:
            raise RuntimeError('\n\nWARNING!!! PICK A VALID OBSERVATION STRATEGY!\n\n')

        # Filter out case that all dual var metrics are zero
        if np.max(self.dual_var_metric) <= 1e-5:
            self.most_rel_obs_ind = None
        else:
            self.most_rel_obs_ind = int(np.argmax(self.dual_var_metric))

        # Update the plot ellipsoids
        obs_ellipse = []
        for j in range(self.num_obs):

            cur_obs_ell_list = self.lin_obs_list[j].ellipse_list
            cur_obs_ell_col_list = self.lin_obs_list[j].ellipse_color_list

            obs_center = obs_mu_bars_time_hor[j][0]
            obs_matrix_inv = LA.inv(obs_Qplus_mat_time_hor[j][0])
            eigs, eig_vecs = LA.eig(obs_matrix_inv)
            angle = np.rad2deg(math.atan2(eig_vecs[0, 1], eig_vecs[0, 0]))

            if len(cur_obs_ell_list) < 15:
                cur_obs_ell_list.insert(0, Ellipse(obs_center, 2 / np.sqrt(eigs[0]),
                                                   2 / np.sqrt(eigs[1]), -angle))
                if self.lin_obs_list[j].observed_last_step:
                    cur_obs_ell_col_list.insert(0, 'b')
                else:
                    cur_obs_ell_col_list.insert(0, 'r')
            else:
                cur_obs_ell_list.insert(0, Ellipse(obs_center, 2 / np.sqrt(eigs[0]),
                                                   2 / np.sqrt(eigs[1]), -angle))
                if self.lin_obs_list[j].observed_last_step:
                    cur_obs_ell_col_list.insert(0, 'b')
                else:
                    cur_obs_ell_col_list.insert(0, 'r')
                cur_obs_ell_list.pop()
                cur_obs_ell_col_list.pop()
            # Not sure if this is necessary, but resave the current obstacle's ellipse list
            self.lin_obs_list[j].ellipse_list = cur_obs_ell_list
            self.lin_obs_list[j].ellipse_color_list = cur_obs_ell_col_list

            # cur_obs_ell_collection = PatchCollection(cur_obs_ell_list,
            #                                          facecolors=cur_obs_ell_col_list, alpha=0.3, zorder=0)
            #
            # # cur_obs_ell_collection.set_alphas([0.4/(t_step+1) for t_step in range(len(cur_obs_ell_list))])
            #
            # self.lin_obs_list[j].ell_collection = cur_obs_ell_collection
            #
            # self.ax.add_collection(self.lin_obs_list[j].ell_collection)

        # Hard code the following for now...
        all_obstacle_ell_coll = []
        for t_step in range(len(self.lin_obs_list[0].ellipse_list)):
            if t_step == 0:
                cur_ell_list = [self.lin_obs_list[0].ellipse_list[t_step], self.lin_obs_list[1].ellipse_list[t_step],
                                self.lin_obs_list[2].ellipse_list[t_step]]
                cur_ell_col_list = [self.lin_obs_list[0].ellipse_color_list[t_step],
                                    self.lin_obs_list[1].ellipse_color_list[t_step],
                                    self.lin_obs_list[2].ellipse_color_list[t_step]]
                all_obstacle_ell_coll = [PatchCollection(cur_ell_list,facecolors=cur_ell_col_list,alpha=0.4/(t_step+1))]
            else:
                cur_ell_list = [self.lin_obs_list[0].ellipse_list[t_step], self.lin_obs_list[1].ellipse_list[t_step],
                                self.lin_obs_list[2].ellipse_list[t_step]]
                cur_ell_col_list = [self.lin_obs_list[0].ellipse_color_list[t_step],
                                    self.lin_obs_list[1].ellipse_color_list[t_step],
                                    self.lin_obs_list[2].ellipse_color_list[t_step]]
                cur_patch_coll = PatchCollection(cur_ell_list,facecolors=cur_ell_col_list,alpha=0.4/(t_step+1))
                all_obstacle_ell_coll.append(cur_patch_coll)
        for t_step in range(len(self.lin_obs_list[0].ellipse_list)):
            self.ax.add_collection(all_obstacle_ell_coll[t_step])

        # Update agent trajectory and nominal future trajectory
        nominal_trajectory_plot = plt.scatter(np.hstack((self.rob_pos[0],
                                                         robot_RHC_trajectory_state_x_time[0, :])),
                                              np.hstack((self.rob_pos[1],
                                                         robot_RHC_trajectory_state_x_time[1, :])), 2,
                                              marker='o', color='c')  # 'b:')
        plt.scatter(self.rob_pos[0],
                    self.rob_pos[1], 10, marker='x', color='k')

        # Update the obstacle trajectories
        obs_x = [self.lin_obs_list[i].act_position[0] for i in range(self.num_lin_obs)]
        obs_y = [self.lin_obs_list[i].act_position[1] for i in range(self.num_lin_obs)]
        plt.scatter(obs_x, obs_y, 10, marker='x', color='r')

        ellipse_collection = PatchCollection(obs_ellipse, facecolor='r', alpha=0.3, zorder=0)
        self.ax.add_collection(ellipse_collection)

        plt.draw()
        save_name = self.obs_strat + "_" + str(self.total_time_steps) + "_" + "fig.png"
        plt.savefig(save_name)
        plt.pause(0.001)

        # Remove the old ellipse collection, nominal trajectory
        # for j in range(self.num_lin_obs):
        #     self.lin_obs_list[j].ell_collection.remove()
        for t_step in range(len(self.lin_obs_list[0].ellipse_list)):
            all_obstacle_ell_coll[t_step].remove()
        nominal_trajectory_plot.remove()

        return robot_RHC_trajectory_state_x_time, best_rh_sequence, best_gamma_ind

    # def solve_motion_planning_prob(self,gamma,target_tile_ecos,obs_mu_bars_time_hor,
    #              obs_Q_mat_time_hor,obs_rad_vector,mult_matrix_stack,A_ecosqp,b_ecosqp,
    #              H_ecosqp,f_ecosqp,init_turn_ang):
    #
    #     sampling_time = self.sampling_time
    #     planning_horizon = self.planning_horizon
    #     heading_angle = self.heading_angle
    #
    #     ##########################################################
    #     # Given the current heading angle, use the fixed turning #
    #     # rate to propagate it over the time horizon, then use   #
    #     # these values to construct the corresponding B matrices #
    #     # over the time horizon. Additionally, precompute some   #
    #     # parameters for ECOS that are dependent on these values #
    #     ##########################################################
    #     if init_turn_ang == 0:
    #         rh_ang_t_hor_cur_gamma = [heading_angle + j * sampling_time * gamma for j in range(planning_horizon)]
    #     else:
    #         rh_ang_t_hor_cur_gamma = [heading_angle, heading_angle + init_turn_ang] \
    #                                  + [heading_angle + init_turn_ang + j * sampling_time * gamma for j in
    #                                     range(1, planning_horizon - 1)]
    #     B_mat_t_hor_cur_gamma = [sampling_time * np.array([[np.cos(rh_ang_t_hor_cur_gamma[j])],
    #                                 [np.sin(rh_ang_t_hor_cur_gamma[j])]]) for j in range(planning_horizon)]
    #
    #     # Store the information for the current value of the turning rate \gamma
    #     rob_motion_plans, rob_input_sequence, rob_obs_func_val, rob_obs_func_dual_vals, \
    #     obs_Qplus_mat_time_hor = self.inner_motion_planning_prob(B_mat_t_hor_cur_gamma,
    #                                 target_tile_ecos,obs_mu_bars_time_hor, obs_rad_vector,
    #                                 rh_ang_t_hor_cur_gamma,obs_Q_mat_time_hor, mult_matrix_stack,
    #                                 A_ecosqp, b_ecosqp, H_ecosqp, f_ecosqp)
    #
    #     return rob_motion_plans, rob_input_sequence, rob_obs_func_val, rob_obs_func_dual_vals, \
    #            obs_Qplus_mat_time_hor, rh_ang_t_hor_cur_gamma
    #
    # def inner_motion_planning_prob(self,B_mat_t_hor_cur_gamma,target_tile_ecos,
    #     obs_mu_bars_time_hor, obs_rad_vector,rh_ang_t_hor_cur_gamma,obs_Q_mat_time_hor,
    #     mult_matrix_stack,A_ecosqp, b_ecosqp, H_ecosqp, f_ecosqp):
    #
    #     robot_A = self.rob_A_mat
    #     planning_horizon = self.planning_horizon
    #     rob_state_dim = self.rob_state_dim
    #     rob_input_dim = self.rob_input_dim
    #     current_state = self.rob_pos
    #     rob_x_max = self.rob_state_x_max
    #     rob_y_max = self.rob_state_y_max
    #     rob_max_velocity = self.rob_max_velocity
    #     n_obstacles = self.num_obs
    #     most_rel_obs_ind = self.most_rel_obs_ind
    #
    #     Z, H = self.get_concatenated_matrices(B_mat_t_hor_cur_gamma)
    #
    #     #######################################################################
    #     # Using the computed dynamics, solve for the obstacle free trajectory #
    #     # given the fixed turning rate sequence.                              #
    #     #######################################################################
    #     robot_RHC_trajectory_initial_solution, robot_input_initial_solution, \
    #     A_eq_ecosqp, b_eq_ecosqp, obs_free_traj_func_val = \
    #         solve_obs_free_ecos_unicycle(Z, H, current_state, rob_state_dim,
    #                                      rob_input_dim, planning_horizon, A_ecosqp,
    #                                      b_ecosqp, H_ecosqp, f_ecosqp)
    #     robot_initial_trajectory_state_x_time = np.reshape(
    #         robot_RHC_trajectory_initial_solution, (planning_horizon, rob_state_dim)).T
    #
    #     ####################################################################
    #     # From the nominal trajectory, compute the Q+ matrices (outer      #
    #     # approximation to Minkowski sum of original Q and the rigid body  #
    #     # ball of each obstacle (of radius r_j). The ellipsoidal outer     #
    #     # approximation is tight in our chosen direction of interest,      #
    #     # which is the direction defined by the robot position to the mean #
    #     # position of the obstacle.                                        #
    #     ####################################################################
    #     obs_Qplus_mat_time_hor = get_Qplus_mats_over_time_hor(planning_horizon, n_obstacles,
    #                               obs_mu_bars_time_hor, obs_Q_mat_time_hor,
    #                               robot_initial_trajectory_state_x_time,
    #                               obs_rad_vector, rob_state_dim)
    #
    #     ###############################################################
    #     # STEP 2: Check for collisions between our nominal trajectory #
    #     # and the obstacle ellipsoidal outer approximations           #
    #     ###############################################################
    #     collision_flag_list = check_collisions(robot_initial_trajectory_state_x_time,
    #                                            obs_mu_bars_time_hor, obs_Qplus_mat_time_hor,
    #                                            n_obstacles)
    #
    #     # If there were collisions, use DC procedure to
    #     # (hopefully) adjust nominal trajectory to prevent them
    #     if any(collision_flag_list):
    #
    #         ####################################################################
    #         # Set up some parameters to feed into the ECOS DC solver function. #
    #         # Doing so here will help to set up the remaining problem          #
    #         # more efficiently.                                                #
    #         ####################################################################
    #         obs_Qplus_stack_all, obs_mu_bar_stack_all = stack_params_for_ecos(n_obstacles, planning_horizon,
    #                                                       obs_mu_bars_time_hor,obs_Qplus_mat_time_hor)
    #
    #         ########################################################
    #         # Use DC program to determine obstacle-free trajectory #
    #         ########################################################
    #         # print('Resolving via DC programming to avoid collisions!')
    #         obs_cons_dual_variables, traj, input, obs_func_val = \
    #             dc_motion_planning_call_ecos_unicycle(n_obstacles, planning_horizon, rob_state_dim,
    #                                                   rob_input_dim,robot_RHC_trajectory_initial_solution,
    #                                                   obs_mu_bar_stack_all, obs_Qplus_stack_all, mult_matrix_stack,
    #                                                   target_tile_ecos, obs_mu_bars_time_hor,
    #                                                   obs_Qplus_mat_time_hor, most_rel_obs_ind, rh_ang_t_hor_cur_gamma,
    #                                                   A_eq_ecosqp, b_eq_ecosqp, A_ecosqp, b_ecosqp, H_ecosqp,
    #                                                   f_ecosqp,tau_max=np.array([10000]), mu=20, delta=1e-3, tau=1)
    #
    #         ###############################################
    #         # Check to see if our updated trajectory has  #
    #         # avoided obstacle collisions                 #
    #         ###############################################
    #         robot_RHC_trajectory_state_x_time = np.reshape(
    #             traj, (planning_horizon, rob_state_dim)).T
    #         collision_flag_list = check_collisions(
    #             robot_RHC_trajectory_state_x_time, obs_mu_bars_time_hor,
    #             obs_Qplus_mat_time_hor, n_obstacles)
    #         if any(collision_flag_list):
    #             # raise RuntimeError('\n\nWARNING!!! Collision not resolved!\n\n')
    #             robot_RHC_trajectory = None
    #             robot_input_sequence = None
    #             obs_cons_dual_variables = None
    #             obs_func_val = np.nan
    #         else:
    #             # Use the resolved solution
    #             robot_RHC_trajectory = traj
    #             robot_input_sequence = input
    #
    #             # Store information about dual variables at current time step
    #             obs_cons_dual_variables = np.asarray(obs_cons_dual_variables)
    #
    #     else:
    #         #####################################################
    #         # If no collisions detected, use nominal trajectory #
    #         #####################################################
    #         robot_RHC_trajectory_state_x_time = \
    #             robot_initial_trajectory_state_x_time
    #         robot_RHC_trajectory = robot_RHC_trajectory_initial_solution
    #         robot_input_sequence = robot_input_initial_solution
    #         obs_func_val = obs_free_traj_func_val
    #         obs_cons_dual_variables = np.zeros((n_obstacles * planning_horizon, 1))
    #
    #     return robot_RHC_trajectory,robot_input_sequence,obs_func_val,\
    #        obs_cons_dual_variables,obs_Qplus_mat_time_hor

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

        # For plotting purposes
        self.ellipse_list = []
        self.ellipse_color_list = []
        self.observed_last_step = False

    def update_obstacle_position(self,query_obs_pos,rob_act_pos,
                                 rob_field_of_view_rad,current_physical_state):
        """
        input: robot_act_pos: actual position of robot in the environment, given
            by oracle (only queried when specifically chosen to make an observation of)
        input: query_obs_pos: True if we are to make an observation about this obstacle
            at the current time step (most relevant AND at interval)
        input: rob_act_pos: position of the robotic agent in the environment
        input: rob_field_of_view_rad: radius for which the agent can make an observation
            of an obstacle if they are within this distance

        NOTE: current_physical_state obtained by the ros velocity controller
        """

        # # Sample noise given system parameters
        # noise_cur_t_step = np.reshape(np.random.multivariate_normal(
        #     self.w_mean_vec, self.w_cov_mat, 1),(2,))

        # # Push through the linear dynamics to obtain obstacle's ACTUAL position
        # self.act_position = np.matmul(self.A_matrix,self.act_position) \
        #     + np.matmul(self.F_matrix,noise_cur_t_step)

        # Update the robotic agent's understanding of the
        # obstacle's position (the SIMULATION position, uncertainty ellipsoid)

        if query_obs_pos:

            # Update the true underlying state of the current obstacle, as determined by the
            # ros velocity controller
            self.act_position = current_physical_state

            # Assume a perfect observation
            self.sim_position = self.act_position
            self.sig_mat = np.zeros(np.shape(self.sig_mat))
            self.observed_last_step = True

        else:

            # Update the true underlying state of the current obstacle, as determined by the
            # ros velocity controller
            self.act_position = current_physical_state

            # If not making an observation, push current estimated position and
            # uncertainty through the system dynamics
            A_mat = self.A_matrix
            F_mat = self.F_matrix
            self.sim_position = np.matmul(A_mat, self.sim_position) + np.matmul(F_mat, self.w_mean_vec)
            self.sig_mat = np.matmul(A_mat, np.matmul(self.sig_mat, np.transpose(A_mat))) \
                           + np.matmul(F_mat, np.matmul(self.w_cov_mat, np.transpose(F_mat)))
            self.observed_last_step = False




# TODO: THIS IS CURRENTLY DEPRECATED. IF WE WANT TO INCLUDE NONLINEAR OBSTACLES, NEED TO
# TODO: CHANGE THIS CLASS AS WELL AS THAT OF THE LINEAR OBSTACLE!
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
        noise_cur_t_step = np.random.multivariate_normal(
            self.w_mean_vec, self.w_cov_mat, 1)

        # For notational convenience
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # Push through the nonlinear dynamics
        B_0 = self.sampling_time*np.array([sin_theta, cos_theta])
        self.act_position = np.matmul(self.A_matrix,self.act_position)\
                                + np.matmul(B_0,noise_cur_t_step)
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

# --------------------------------------- BEGIN FUNCTION DEFINITIONS --------------------------------------------- #


def unpack(func):
    @wraps(func)
    def wrapper(arg_tuple):
        return func(*arg_tuple)
    return wrapper

@unpack
def solve_motion_planning_prob(gamma,target_tile_ecos,obs_mu_bars_time_hor,
             obs_Q_mat_time_hor,obs_rad_vector,mult_matrix_stack,A_ecosqp,b_ecosqp,
             H_ecosqp,f_ecosqp,init_turn_ang,
             ##########
             sampling_time, planning_horizon, heading_angle,rob_state_dim,rob_input_dim,current_state,n_obstacles,
             most_rel_obs_ind,state_matrix):

    ##########################################################
    # Given the current heading angle, use the fixed turning #
    # rate to propagate it over the time horizon, then use   #
    # these values to construct the corresponding B matrices #
    # over the time horizon. Additionally, precompute some   #
    # parameters for ECOS that are dependent on these values #
    ##########################################################
    if init_turn_ang == 0:
        rh_ang_t_hor_cur_gamma = [heading_angle + j * sampling_time * gamma for j in range(planning_horizon)]
    else:
        rh_ang_t_hor_cur_gamma = [heading_angle, heading_angle + init_turn_ang] \
                                 + [heading_angle + init_turn_ang + j * sampling_time * gamma for j in
                                    range(1, planning_horizon - 1)]
    B_mat_t_hor_cur_gamma = [sampling_time * np.array([[np.cos(rh_ang_t_hor_cur_gamma[j])],
                                [np.sin(rh_ang_t_hor_cur_gamma[j])]]) for j in range(planning_horizon)]

    # Store the information for the current value of the turning rate \gamma
    rob_motion_plans, rob_input_sequence, rob_obs_func_val, rob_obs_func_dual_vals, \
    obs_Qplus_mat_time_hor = inner_motion_planning_prob(B_mat_t_hor_cur_gamma,
                                target_tile_ecos,obs_mu_bars_time_hor, obs_rad_vector,
                                rh_ang_t_hor_cur_gamma,obs_Q_mat_time_hor, mult_matrix_stack,
                                A_ecosqp, b_ecosqp, H_ecosqp, f_ecosqp,
                                ##########
                                planning_horizon,rob_state_dim,rob_input_dim,current_state,n_obstacles,most_rel_obs_ind,state_matrix)

    return rob_motion_plans, rob_input_sequence, rob_obs_func_val, rob_obs_func_dual_vals, obs_Qplus_mat_time_hor, rh_ang_t_hor_cur_gamma


def inner_motion_planning_prob(B_mat_t_hor_cur_gamma,target_tile_ecos,
    obs_mu_bars_time_hor, obs_rad_vector,rh_ang_t_hor_cur_gamma,obs_Q_mat_time_hor,
    mult_matrix_stack,A_ecosqp, b_ecosqp, H_ecosqp, f_ecosqp,
    #########
    planning_horizon,rob_state_dim,rob_input_dim,current_state,n_obstacles,most_rel_obs_ind,state_matrix):

    Z, H = get_concatenated_matrices(B_mat_t_hor_cur_gamma,
                                     ##########
                                     state_matrix,planning_horizon,rob_state_dim,rob_input_dim)

    #######################################################################
    # Using the computed dynamics, solve for the obstacle free trajectory #
    # given the fixed turning rate sequence.                              #
    #######################################################################
    robot_RHC_trajectory_initial_solution, robot_input_initial_solution, \
    A_eq_ecosqp, b_eq_ecosqp, obs_free_traj_func_val = \
        solve_obs_free_ecos_unicycle(Z, H, current_state, rob_state_dim,
                                     rob_input_dim, planning_horizon, A_ecosqp,
                                     b_ecosqp, H_ecosqp, f_ecosqp)
    robot_initial_trajectory_state_x_time = np.reshape(
        robot_RHC_trajectory_initial_solution, (planning_horizon, rob_state_dim)).T

    ####################################################################
    # From the nominal trajectory, compute the Q+ matrices (outer      #
    # approximation to Minkowski sum of original Q and the rigid body  #
    # ball of each obstacle (of radius r_j). The ellipsoidal outer     #
    # approximation is tight in our chosen direction of interest,      #
    # which is the direction defined by the robot position to the mean #
    # position of the obstacle.                                        #
    ####################################################################
    obs_Qplus_mat_time_hor = get_Qplus_mats_over_time_hor(planning_horizon, n_obstacles,
                              obs_mu_bars_time_hor, obs_Q_mat_time_hor,
                              robot_initial_trajectory_state_x_time,
                              obs_rad_vector, rob_state_dim)

    ###############################################################
    # STEP 2: Check for collisions between our nominal trajectory #
    # and the obstacle ellipsoidal outer approximations           #
    ###############################################################
    collision_flag_list = check_collisions(robot_initial_trajectory_state_x_time,
                                           obs_mu_bars_time_hor, obs_Qplus_mat_time_hor,
                                           n_obstacles)

    # If there were collisions, use DC procedure to
    # (hopefully) adjust nominal trajectory to prevent them
    if any(collision_flag_list):

        ####################################################################
        # Set up some parameters to feed into the ECOS DC solver function. #
        # Doing so here will help to set up the remaining problem          #
        # more efficiently.                                                #
        ####################################################################
        obs_Qplus_stack_all, obs_mu_bar_stack_all = stack_params_for_ecos(n_obstacles, planning_horizon,
                                                      obs_mu_bars_time_hor,obs_Qplus_mat_time_hor)

        ########################################################
        # Use DC program to determine obstacle-free trajectory #
        ########################################################
        # print('Resolving via DC programming to avoid collisions!')
        obs_cons_dual_variables, traj, input, obs_func_val = \
            dc_motion_planning_call_ecos_unicycle(n_obstacles, planning_horizon, rob_state_dim,
                                                  rob_input_dim,robot_RHC_trajectory_initial_solution,
                                                  obs_mu_bar_stack_all, obs_Qplus_stack_all, mult_matrix_stack,
                                                  target_tile_ecos, obs_mu_bars_time_hor,
                                                  obs_Qplus_mat_time_hor, most_rel_obs_ind, rh_ang_t_hor_cur_gamma,
                                                  A_eq_ecosqp, b_eq_ecosqp, A_ecosqp, b_ecosqp, H_ecosqp,
                                                  f_ecosqp,tau_max=np.array([10000]), mu=20, delta=1e-3, tau=1)

        ###############################################
        # Check to see if our updated trajectory has  #
        # avoided obstacle collisions                 #
        ###############################################
        robot_RHC_trajectory_state_x_time = np.reshape(
            traj, (planning_horizon, rob_state_dim)).T
        collision_flag_list = check_collisions(
            robot_RHC_trajectory_state_x_time, obs_mu_bars_time_hor,
            obs_Qplus_mat_time_hor, n_obstacles)
        if any(collision_flag_list):
            # raise RuntimeError('\n\nWARNING!!! Collision not resolved!\n\n')
            robot_RHC_trajectory = None
            robot_input_sequence = None
            obs_cons_dual_variables = None
            obs_func_val = np.nan
        else:
            # Use the resolved solution
            robot_RHC_trajectory = traj
            robot_input_sequence = input

            # Store information about dual variables at current time step
            obs_cons_dual_variables = np.asarray(obs_cons_dual_variables)

    else:
        #####################################################
        # If no collisions detected, use nominal trajectory #
        #####################################################
        robot_RHC_trajectory_state_x_time = \
            robot_initial_trajectory_state_x_time
        robot_RHC_trajectory = robot_RHC_trajectory_initial_solution
        robot_input_sequence = robot_input_initial_solution
        obs_func_val = obs_free_traj_func_val
        obs_cons_dual_variables = np.zeros((n_obstacles * planning_horizon, 1))

    return robot_RHC_trajectory,robot_input_sequence,obs_func_val,\
       obs_cons_dual_variables,obs_Qplus_mat_time_hor


def get_concatenated_matrices(input_matrix_array,
                              ##########
                              state_matrix,planning_horizon,rob_state_dim,rob_input_dim):
    """
    Compute the matrices Z and H such that the concatenated state vector
            X = [x_1 x_2 .... x_T]
    can be expressed in terms of the concatenated input
            U = [u_0 u_1 .... u_{T-1}].
    Specifically, we have
            X = Z x_0 + H U

    This permits analysis of discrete-time systems x_{t+1} = A x_t + B u_t

    :param state_matrix: System matrix A
    :param input_matrix: Input matrix B
    :param time_horizon:
    :return: concatenated matrices Z and H
    """

    # Describe the dynamics as X = Z x_0 + H U (X excludes current state)
    # Z matrix [A   A^2 .... A^T]
    Z = np.zeros((planning_horizon * rob_state_dim, rob_state_dim))
    for time_index in range(planning_horizon):
        Z[time_index * rob_state_dim:
          (time_index + 1) * rob_state_dim, :] = \
            np.linalg.matrix_power(state_matrix, time_index + 1)

    # H matrix via flipped controllability matrices
    # flipped_controllability_matrix is [A^(T-1)B, A^(T-2)B, ... , AB, B]
    flipped_controllability_matrix = \
        np.zeros((rob_state_dim, rob_input_dim * planning_horizon))
    for time_index in range(planning_horizon):
        flip_time_index = planning_horizon - 1 - time_index
        cur_input_matrix = input_matrix_array[flip_time_index]
        flipped_controllability_matrix[:,
        flip_time_index * rob_input_dim:
        (flip_time_index + 1) * rob_input_dim] = \
            np.matmul(np.linalg.matrix_power(state_matrix, time_index), cur_input_matrix)
    H = np.tile(flipped_controllability_matrix, (planning_horizon, 1))
    for time_index in range(planning_horizon - 1):
        zeroed_indices = (planning_horizon - time_index - 1) * rob_input_dim
        H[time_index * rob_state_dim:
          (time_index + 1) * rob_state_dim,
        (time_index + 1) * rob_input_dim:] = \
            np.zeros((rob_state_dim, zeroed_indices))

    return Z, H