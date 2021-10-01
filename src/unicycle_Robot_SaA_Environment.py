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
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex"
# })
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

# For solving MPC problems
import casadi as csi
import ecos
import osqp
from scipy.sparse import csc_matrix as csc

# Mathematic functions
from math import atan2


class RobotSaAEnvironment:

    def __init__(self, target_state, target_tolerance, gauss_lev_param, planning_horizon,
                 rob_init_pos, rob_A_mat, obs_field_of_view_rad, obs_interval,
                 rob_state_x_max, rob_state_y_max, sampling_time, observation_strategy,
                 max_heading_view, rob_min_velocity, rob_max_velocity, rob_max_turn_rate,
                 rob_agg_turn_rate, most_rel_obs_ind, num_turning_rates,
                 turning_rates_array, rob_heading_ang, discount_factor, discount_weight):
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
        self.rob_min_velocity = rob_min_velocity
        self.rob_max_velocity = rob_max_velocity
        self.rob_max_turn_rate = rob_max_turn_rate
        self.rob_agg_turn_rate = rob_agg_turn_rate
        self.most_rel_obs_ind = most_rel_obs_ind
        self.num_turning_rates = num_turning_rates
        self.turning_rates_array = turning_rates_array
        self.heading_angle = rob_heading_ang
        self.discount_factor = discount_factor
        self.discount_weight = discount_weight
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
        self.input_sequence = np.zeros(self.planning_horizon)
        self.turning_rate_sequence = np.zeros(self.planning_horizon)

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

    def solve_optim_prob_and_update(self):
        """
        input: robot_act_pos: the ACTUAL position of the robotic agent in
        return: rob_traj: set of waypoints for the robotic agent to follow
        return: obs_traj: next waypoint for each of the obstacles to travel to
        NOTE: obs_act_positions is obtained from querying the ros velocity controller
            associated with that obstacle!
        """

        # Update the number of time steps considered
        self.total_time_steps += 1

        # Take a step of the safely solver (create obstacle ellipsoids, etc.)
        self.nominal_trajectory, self.heading_angle_sequence, \
            self.input_sequence, self.turning_rate_sequence = self.dynamics_planning()

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

    def dynamics_planning(self):
        """
        Given state of robotic agent's environment, construct a new trajectory
        to follow (i.e. generate a set of waypoints to send to the agent)
        """

        np.random.seed(0)
    
        #
        # Start by propagating the linear obstacles over the planning horizon
        #
     
        # First, propagate one time step in the future
        linear_obs_mu_bars_t, linear_obs_sig_bars_t = \
            propagate_linear_obstacles(self.num_lin_obs, self.lin_obs_list)

        # Now, propagate the linear obstacles over the time horizon
        linear_obs_mu_bars_time_hor, linear_obs_sig_bars_time_hor = \
            get_mu_sig_over_time_hor(self.planning_horizon, self.num_lin_obs, linear_obs_mu_bars_t,
                            linear_obs_sig_bars_t, self.lin_obs_list)

        # Use this information to construct the matrix Q for each obstacle,
        # which is the ellipsoidal level set centered around the mean position
        # sliced according to the parameter beta
        linear_obs_Q_mat_time_hor = get_Q_mats_over_time_hor(self.planning_horizon,
            self.num_lin_obs, self.beta, self.lin_obs_list, linear_obs_sig_bars_time_hor)
    
        #
        # Now, propagate the nonlinear obstacles over the planning
        # horizon (assume that the turning rate is fixed)
        #
    
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
    
    #
        # Concatenate the linear and nonlinear obstacle ellipsoids. From here on out,
        # they are functionally the same. For now, assume that we will always have at
        # least one linear obstacle.
    #
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
        
        sol_tic = time.time()
        
        # Construct the discount factor array:
        discount_factors = np.zeros((self.planning_horizon,self.num_obs))
        if (self.most_rel_obs_ind is not None) and (not (self.most_rel_obs_ind in self.observable_obstacles_list)):
            discount_factors[:,self.most_rel_obs_ind] = \
            np.array([self.discount_weight*self.discount_factor**t_step for t_step in range(self.planning_horizon)])
        discount_factors = np.reshape(discount_factors.T,(self.planning_horizon*self.num_obs,1))

        # Solve the inner MPC function
        robot_RHC_trajectory, robot_input_sequence,robot_heading_angle_sequence, robot_turning_rate_sequence,\
            obs_func_val,obs_cons_dual_variables,obs_Qplus_mat_time_hor = solve_mpc(self.rob_pos,self.heading_angle,self.target_state,\
                self.rob_state_dim,self.rob_input_dim,self.num_obs,obs_rad_vector,obs_mu_bars_time_hor,\
                obs_Q_mat_time_hor,self.planning_horizon,\
                self.all_dv_lower_bounds_nl,
                self.all_dv_upper_bounds_nl,\
                self.constraints_dyn_lower_bounds_nl,\
                self.constraints_dyn_upper_bounds_nl,\
                self.all_constraints_lower_bounds_nl,\
                self.all_constraints_upper_bounds_nl,\
                self.all_constraints_lower_bounds_proj_nl,\
                self.all_constraints_upper_bounds_proj_nl,\
                self.safely_mpc_init_guess_solver,\
                self.safely_solver_ipopt,\
                self.safely_mpc_projection_solver,\
                discount_factors)
        
        sol_toc = time.time()

        self.solve_times.append(sol_toc-sol_tic)

        # Reshape the output trajectory to a more convenient form
        robot_RHC_trajectory_state_x_time = np.reshape(
            robot_RHC_trajectory, (self.planning_horizon, self.rob_state_dim)).T

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


        return robot_RHC_trajectory_state_x_time, robot_heading_angle_sequence, robot_input_sequence, robot_turning_rate_sequence
        
        
    def construct_mpc_solvers_and_get_bounds(self):
        """
        
        Using Casadi, construct the nonlinear mpc solvers used in the motion-planning problem
        
        """
        
        # 
        # Construct the nonlinear safely MPC problem and the initial guess
        # 

        # Declare the variables
        state_vars = csi.SX.sym('state_x',2*self.planning_horizon,1)
        state_th = csi.SX.sym('state_th',self.planning_horizon,1)
        state_v = csi.SX.sym('state_v',self.planning_horizon,1)
        state_tr = csi.SX.sym('state_tr',self.planning_horizon,1)

        # Declare what will be used as the parameters
        obs_pos_mean = csi.SX.sym('obs_pos_mean',2,1,self.planning_horizon,self.num_obs)
        obs_pos_cov = csi.SX.sym('obs_pos_cov',2,2,self.planning_horizon,self.num_obs)
        x_i = csi.SX.sym('init_state',2,1)
        th_i = csi.SX.sym('init_heading_angle')
        x_g = csi.SX.sym('goal_state',2,1)
        gamma_obs_t = csi.SX.sym('discount_factors',self.planning_horizon,self.num_obs)

        # Initialize the objective function
        objective_function_nl = 0

        # Initialize constraints for dynamics
        constraints_dyn_nl = []
        constraints_dyn_lower_bounds_nl = []
        constraints_dyn_upper_bounds_nl = []

        # Initialize the constraints for elliptical keep-out zone
        constraints_obs_nl = []
        constraints_obs_lower_bounds_nl = []
        constraints_obs_upper_bounds_nl = []

        # Initialize variable bounds
        variable_lower_bounds_state_vars_nl = []
        variable_upper_bounds_state_vars_nl = []
        variable_lower_bounds_th_nl = []
        variable_upper_bounds_th_nl = []
        variable_lower_bounds_v_nl = []
        variable_upper_bounds_v_nl = []
        variable_lower_bounds_tr_nl = []
        variable_upper_bounds_tr_nl = []

        # Loop through variables and construct the objective function
        for t_step in range(self.planning_horizon): 

            # Update the objective function component for reaching the goal state
            objective_function_nl += csi.norm_2(state_vars[2*t_step:2*(t_step+1)] - x_g)

            # Iterate through the list of obstacles
            for obs_ind in range(self.num_obs):

                # Update the objective function component for turning
                objective_function_nl -= gamma_obs_t[t_step,obs_ind] * \
                    csi.dot( obs_pos_mean[obs_ind][t_step] - state_vars[2*t_step:2*(t_step+1)],\
                             csi.vertcat(csi.cos(state_th[t_step]),csi.sin(state_th[t_step])) )

            # Append the current dynamics constraint: differentiate whether beginning
            # from the initial state or not
            if t_step == 0:

                # Initial position dynamics constraint
                constraints_dyn_nl.append(csi.vertcat(state_vars[2*t_step:2*(t_step+1)] \
                    - x_i 
                    - self.sampling_time*csi.vertcat(state_v[t_step]*csi.cos(th_i),state_v[t_step]*csi.sin(th_i))))
                constraints_dyn_lower_bounds_nl += [0.0,0.0]
                constraints_dyn_upper_bounds_nl += [0.0,0.0]

                # Initial heading angle dynamics constraint
                constraints_dyn_nl.append(state_th[t_step] - th_i - self.sampling_time*state_tr[t_step])
                constraints_dyn_lower_bounds_nl += [0.0]
                constraints_dyn_upper_bounds_nl += [0.0]

            else:

                # Intermediate / terminal position dynamics constraint
                constraints_dyn_nl.append(state_vars[2*t_step:2*(t_step+1)] \
                    - state_vars[2*(t_step-1):2*t_step] \
                    - self.sampling_time*csi.vertcat(state_v[t_step]*csi.cos(state_th[t_step-1]),state_v[t_step]*csi.sin(state_th[t_step-1])))
                constraints_dyn_lower_bounds_nl += [0.0,0.0]
                constraints_dyn_upper_bounds_nl += [0.0,0.0]

                # Intermediate / terminal heading angle constraint
                constraints_dyn_nl.append(state_th[t_step] - state_th[t_step-1] - self.sampling_time*state_tr[t_step])
                constraints_dyn_lower_bounds_nl += [0.0]
                constraints_dyn_upper_bounds_nl += [0.0]

            # Append the lower and upper bounds for each of the variables at the current time step
                
            # Position components
            variable_lower_bounds_state_vars_nl += [-self.rob_state_x_max,-self.rob_state_y_max]
            variable_upper_bounds_state_vars_nl += [self.rob_state_x_max,self.rob_state_y_max]

            # heading angle components
            variable_lower_bounds_th_nl += [-np.inf]
            variable_upper_bounds_th_nl += [np.inf]

            # velocity components
            variable_lower_bounds_v_nl += [self.rob_min_velocity]
            variable_upper_bounds_v_nl += [self.rob_max_velocity]

            # turning rate components
            variable_lower_bounds_tr_nl += [-1*self.rob_max_turn_rate]
            variable_upper_bounds_tr_nl += [self.rob_max_turn_rate]

        # Iterate through the ellipses at each time step, add constraints
        for obs_ind in range(self.num_obs):
            for t_step in range(self.planning_horizon):

                # Access the mean position and ellipse shape of the current obstacle:
                cur_obs_mean_pos = obs_pos_mean[obs_ind][t_step]
                cur_obs_pos_cov = obs_pos_cov[obs_ind][t_step]

                constraints_obs_nl.append( (state_vars[2*t_step:2*(t_step+1)] - cur_obs_mean_pos).T @ \
                    cur_obs_pos_cov @ (state_vars[2*t_step:2*(t_step+1)] - cur_obs_mean_pos) )
                constraints_obs_lower_bounds_nl += [1.0]
                constraints_obs_upper_bounds_nl += [np.inf]

        # Combine all of the constraints
        all_constraints_nl = csi.vertcat(csi.vertcat(*constraints_dyn_nl),csi.vertcat(*constraints_obs_nl))
        self.constraints_dyn_lower_bounds_nl = constraints_dyn_lower_bounds_nl
        self.constraints_dyn_upper_bounds_nl = constraints_dyn_upper_bounds_nl
        self.all_constraints_lower_bounds_nl = constraints_dyn_lower_bounds_nl + constraints_obs_lower_bounds_nl
        self.all_constraints_upper_bounds_nl = constraints_dyn_upper_bounds_nl + constraints_obs_upper_bounds_nl

        # Combine all of the variables
        all_decision_variables_nl = csi.vertcat(state_vars,state_v,state_th,state_tr)

        # Combine all of the variable bounds
        self.all_dv_lower_bounds_nl = csi.vertcat(csi.vertcat(variable_lower_bounds_state_vars_nl),csi.vertcat(variable_lower_bounds_v_nl),\
            csi.vertcat(variable_lower_bounds_th_nl),csi.vertcat(variable_lower_bounds_tr_nl))
        self.all_dv_upper_bounds_nl = csi.vertcat(csi.vertcat(variable_upper_bounds_state_vars_nl),csi.vertcat(variable_upper_bounds_v_nl),\
            csi.vertcat(variable_upper_bounds_th_nl),csi.vertcat(variable_upper_bounds_tr_nl))

        # Combine all of the parameters
        # Note that casadi uses a sparse-like representation.
        obs_pos_mean_stacked = []
        obs_pos_cov_stacked = []
        for obs_ind in range(self.num_obs):
            obs_pos_mean_stacked.append(csi.vertcat(*obs_pos_mean[obs_ind]))
            obs_pos_cov_stacked.append(csi.vertcat(*obs_pos_cov[obs_ind]))
        obs_pos_mean_stacked = csi.vertcat(*obs_pos_mean_stacked)
        obs_pos_cov_stacked = csi.vertcat(*obs_pos_cov_stacked)
        obs_pos_cov_stacked = csi.reshape(obs_pos_cov_stacked,4*self.planning_horizon*self.num_obs,1)
        gamma_obs_t_stacked = csi.reshape(gamma_obs_t,self.num_obs*self.planning_horizon,1)

        all_parameters_nl = csi.vertcat(obs_pos_mean_stacked,obs_pos_cov_stacked,x_i,th_i,x_g,gamma_obs_t_stacked)

        all_parameters_init = csi.vertcat(obs_pos_mean_stacked,obs_pos_cov_stacked,x_i,th_i,x_g,gamma_obs_t_stacked)

        # Construct the initial guess to feed into the MPC problem with dynamic obstacles
        safely_mpc_init_guess = {'x':all_decision_variables_nl,\
                            'p':all_parameters_init,'f':objective_function_nl,'g':csi.vertcat(*constraints_dyn_nl)}

        # flags = ["-O3"]
        # compiler = "gcc"
        # jit_options = {"flags": flags, "verbose": True, "compiler": compiler}
        self.safely_mpc_init_guess_solver = csi.nlpsol('solver','ipopt',safely_mpc_init_guess,\
            # {"jit": True, "compiler": "shell", "jit_options": jit_options,'print_time':False,'verbose':False,\
            #  'ipopt':{'print_level':0}})
            {'print_time':False,'verbose':False,'ipopt':{'print_level':0}})


        # Construct the Safely MPC problem
        safely_mpc = {'x':all_decision_variables_nl,'p':all_parameters_nl,'f':objective_function_nl,'g':all_constraints_nl}

        # Declare the solver

        # ipopt: tried and true :)
        # flags = ["-O3"]
        # compiler = "gcc"
        # jit_options = {"flags": flags, "verbose": True, "compiler": compiler}
        # options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        self.safely_solver_ipopt = csi.nlpsol('solver','ipopt',safely_mpc,\
            # {"jit": True, "compiler": "shell", "jit_options": jit_options,'print_time':False,'verbose':False,\
            #  'ipopt':{'print_level':0}})
            {'print_time':False,'verbose':False,'ipopt':{'print_level':0}})


        #
        # Set up the projection-based nonlinear safely MPC problem 
        #

        # Declare the variables
        state_vars = csi.SX.sym('state_x',2*self.planning_horizon,1)
        state_th = csi.SX.sym('state_th',self.planning_horizon,1)
        state_v = csi.SX.sym('state_v',self.planning_horizon,1)
        state_tr = csi.SX.sym('state_tr',self.planning_horizon,1)

        # Declare what will be used as the parameters
        obs_pos_mean = csi.SX.sym('obs_pos_mean',2,1,self.planning_horizon,self.num_obs)
        obs_pos_cov = csi.SX.sym('obs_pos_cov',2,2,self.planning_horizon,self.num_obs) 
        obs_pos_A = csi.SX.sym('obs_pos_A',self.planning_horizon,2,self.num_obs)
        obs_pos_b = csi.SX.sym('obs_pos_b',self.planning_horizon,self.num_obs)
        x_i = csi.SX.sym('init_state',2,1)
        th_i = csi.SX.sym('init_heading_angle')
        x_g = csi.SX.sym('goal_state',2,1)
        gamma_obs_t = csi.SX.sym('discount_factors',self.planning_horizon,self.num_obs)

        # Initialize the objective function
        objective_function_proj_nl = 0

        # Initialize constraints for dynamics
        constraints_dyn_proj_nl = []
        constraints_dyn_lower_bounds_proj_nl = []
        constraints_dyn_upper_bounds_proj_nl = []

        # Initialize the constraints for the projection-based linear inequalities
        constraints_obs_proj_nl = []
        constraints_obs_lower_bounds_proj_nl = []
        constraints_obs_upper_bounds_proj_nl = []

        # Initialize variable bounds
        variable_lower_bounds_state_vars_proj_nl = []
        variable_upper_bounds_state_vars_proj_nl = []
        variable_lower_bounds_th_proj_nl = []
        variable_upper_bounds_th_proj_nl = []
        variable_lower_bounds_v_proj_nl = []
        variable_upper_bounds_v_proj_nl= []
        variable_lower_bounds_tr_proj_nl = []
        variable_upper_bounds_tr_proj_nl = []

        # Loop through variables and construct the objective function, dynamics constraints
        for t_step in range(self.planning_horizon): 

            # Update the objective function
            objective_function_proj_nl += csi.norm_2(state_vars[2*t_step:2*(t_step+1)] - x_g)

            # Iterate through the list of obstacles
            for obs_ind in range(self.num_obs):

                # Update the objective function component for turning
                objective_function_proj_nl -= gamma_obs_t[t_step,obs_ind] * \
                    csi.dot( obs_pos_mean[obs_ind][t_step] - state_vars[2*t_step:2*(t_step+1)],\
                             csi.vertcat(csi.cos(state_th[t_step]),csi.sin(state_th[t_step])) )

            # Append the current dynamics constraint: differentiate whether beginning
            # from the initial state or not
            if t_step == 0:

                # Initial position dynamics constraint
                constraints_dyn_proj_nl.append(csi.vertcat(state_vars[2*t_step:2*(t_step+1)] \
                    - x_i 
                    - self.sampling_time*csi.vertcat(state_v[t_step]*csi.cos(th_i),state_v[t_step]*csi.sin(th_i))))
                constraints_dyn_lower_bounds_proj_nl += [0.0,0.0]
                constraints_dyn_upper_bounds_proj_nl += [0.0,0.0]

                # Initial heading angle dynamics constraint
                constraints_dyn_proj_nl.append(state_th[t_step] - th_i - self.sampling_time*state_tr[t_step])
                constraints_dyn_lower_bounds_proj_nl += [0.0]
                constraints_dyn_upper_bounds_proj_nl += [0.0]

            else:

                # Intermediate / terminal position dynamics constraint
                constraints_dyn_proj_nl.append(state_vars[2*t_step:2*(t_step+1)] \
                    - state_vars[2*(t_step-1):2*t_step] \
                    - self.sampling_time*csi.vertcat(state_v[t_step]*csi.cos(state_th[t_step-1]),state_v[t_step]*csi.sin(state_th[t_step-1])))
                constraints_dyn_lower_bounds_proj_nl += [0.0,0.0]
                constraints_dyn_upper_bounds_proj_nl += [0.0,0.0]

                # Intermediate / terminal heading angle constraint
                constraints_dyn_proj_nl.append(state_th[t_step] - state_th[t_step-1] - self.sampling_time*state_tr[t_step])
                constraints_dyn_lower_bounds_proj_nl += [0.0]
                constraints_dyn_upper_bounds_proj_nl += [0.0]

            # Append the lower and upper bounds for each of the variables at the current time step
            
            # Position components
            variable_lower_bounds_state_vars_proj_nl += [-self.rob_state_x_max,-self.rob_state_y_max]
            variable_upper_bounds_state_vars_proj_nl += [self.rob_state_x_max,self.rob_state_y_max]

            # heading angle components
            variable_lower_bounds_th_proj_nl += [-np.inf]
            variable_upper_bounds_th_proj_nl += [np.inf]

            # velocity components
            variable_lower_bounds_v_proj_nl += [self.rob_min_velocity]
            variable_upper_bounds_v_proj_nl += [self.rob_max_velocity]

            # turning rate components
            variable_lower_bounds_tr_proj_nl += [-self.rob_max_turn_rate]
            variable_upper_bounds_tr_proj_nl += [self.rob_max_turn_rate]

        # Iterate through the obstacles, assign the projection-based hyperplane constraints.
        obs_pos_A = csi.SX.sym('obs_pos_mean',self.planning_horizon,2,self.num_obs)
        obs_pos_b = csi.SX.sym('obs_pos_cov',self.planning_horizon,self.num_obs)
        for obs_ind in range(self.num_obs):
            for t_step in range(self.planning_horizon):
                constraints_obs_proj_nl.append( obs_pos_A[obs_ind][t_step,:] @ state_vars[2*t_step:2*(t_step+1)] - obs_pos_b[t_step,obs_ind] )
                constraints_obs_lower_bounds_proj_nl += [-np.inf]
                constraints_obs_upper_bounds_proj_nl += [0.0]

        # Combine all of the constraints
        all_constraints_proj_nl = csi.vertcat(csi.vertcat(*constraints_obs_proj_nl),csi.vertcat(*constraints_dyn_proj_nl))
        self.all_constraints_lower_bounds_proj_nl = constraints_obs_lower_bounds_proj_nl + constraints_dyn_lower_bounds_proj_nl
        self.all_constraints_upper_bounds_proj_nl = constraints_obs_upper_bounds_proj_nl + constraints_dyn_upper_bounds_proj_nl

        # Combine all of the variables
        all_decision_variables_proj_nl = csi.vertcat(state_vars,state_v,state_th,state_tr)

        # Combine all of the variable bounds
        self.all_dv_lower_bounds_proj_nl = csi.vertcat(csi.vertcat(variable_lower_bounds_state_vars_proj_nl),csi.vertcat(variable_lower_bounds_v_proj_nl),\
            csi.vertcat(variable_lower_bounds_th_proj_nl),csi.vertcat(variable_lower_bounds_tr_proj_nl))
        self.all_dv_upper_bounds_proj_nl = csi.vertcat(csi.vertcat(variable_upper_bounds_state_vars_proj_nl),csi.vertcat(variable_upper_bounds_v_proj_nl),\
            csi.vertcat(variable_upper_bounds_th_proj_nl),csi.vertcat(variable_upper_bounds_tr_proj_nl))

        # Combine all of the parameters
        # Note that casadi uses a sparse-like representation.
        obs_pos_mean_stacked = []
        obs_pos_cov_stacked = []
        for obs_ind in range(self.num_obs):
            obs_pos_mean_stacked.append(csi.vertcat(*obs_pos_mean[obs_ind]))
            obs_pos_cov_stacked.append(csi.vertcat(*obs_pos_cov[obs_ind]))
        obs_pos_mean_stacked = csi.vertcat(*obs_pos_mean_stacked)
        obs_pos_cov_stacked = csi.vertcat(*obs_pos_cov_stacked)
        obs_pos_cov_stacked = csi.reshape(obs_pos_cov_stacked,4*self.planning_horizon*self.num_obs,1)
        gamma_obs_t_stacked = csi.reshape(gamma_obs_t,self.num_obs*self.planning_horizon,1)
        obs_A_stacked = csi.reshape(csi.vertcat(*obs_pos_A),(2*self.num_obs*self.planning_horizon,1))
        obs_b_stacked = csi.reshape(obs_pos_b,(self.num_obs*self.planning_horizon,1))

        # Set up the projection-based MPC problem
        all_parameters = csi.vertcat(obs_pos_mean_stacked,obs_pos_cov_stacked,obs_A_stacked,obs_b_stacked,x_i,th_i,x_g,gamma_obs_t_stacked)
        safely_mpc_projection = {'x':all_decision_variables_proj_nl,'p':all_parameters,'f':objective_function_proj_nl,'g':all_constraints_proj_nl}
        # self.safely_mpc_projection_solver = csi.nlpsol('solver','sqpmethod',safely_mpc_projection,\
        #     {'max_iter':10,'print_time':False,'print_header':False,'verbose':False,'print_status':False,'print_iteration':False,\
        #     'qpsol':'osqp','convexify_strategy':'regularize','error_on_fail':False,'qpsol_options':{'osqp':{'verbose':False}}})
        self.safely_mpc_projection_solver = csi.nlpsol('solver','blocksqp',safely_mpc_projection,\
            {'max_iter':1,'qpsol':'qr','print_time':False,'print_header':False,'verbose':False,'conv_strategy':0})
        # self.safely_mpc_projection_solver = csi.nlpsol('solver','ipopt',safely_mpc_projection,\
        #     {'print_time':False,'verbose':False,'ipopt':{'print_level':4}})

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


def solve_mpc(current_state,current_heading_angle,goal_state,rob_state_dim,rob_input_dim,n_obstacles,\
    obs_rad_vector,obs_mu_bars_time_hor,obs_Q_mat_time_hor,planning_horizon,all_dv_lower_bounds_nl,all_dv_upper_bounds_nl,\
    constraints_dyn_lower_bounds_nl,constraints_dyn_upper_bounds_nl,all_constraints_lower_bounds_nl,\
    all_constraints_upper_bounds_nl,all_constraints_lower_bounds_proj_nl,all_constraints_upper_bounds_proj_nl,\
    safely_mpc_init_guess_solver,safely_solver_ipopt,safely_mpc_projection_solver,discount_factors):
    """
    
    Solve the nonlinear MPC problem and obtain information on the dual varaibles corresponding to obstacle avoidance
    
    Inputs:
    current_state: The current state of the ego robot
    current_heading_angle: The current heading angle of the ego robot
    goal_state: The goal state of the ego robot
    rob_state_dim: The dimension of the state space
    rob_input_dim: The dimension of the input space
    n_obstacles: The number of dynamic obstacles in the environment (linear / nonlinear)
    obs_rad_vector: List containing the radius of each hard-body obstacle
    obs_mu_bars_time_hor: The mean positions of each obstacle in the environment over the course
        of the planning horizon
    obs_Q_mat_time_hor: The initial obstacle ellipsoids over the course of the planning horizon
    planning_horizon: Planning horizon length for the ego robot
    all_dv_lower_bounds_nl: All lower bounds for the decision variables in the nonlinear problem
    all_dv_upper_bounds_nl: All upper bounds for the decision variables in the nonlinear problem
    constraints_dyn_lower_bounds_nl: All lower bounds for the dynamics constraints in the nonlinear
        problem (since these are equality, these are trivially zero)
    constraints_dyn_upper_bounds_nl: All upper bounds for the dynamics constraints in the nonlinear
        problem (Again, since these are equality, these are trivially zero)
    all_constraints_lower_bounds_nl: All lower bounds for the nonlinear mpc constraints
    all_constraints_upper_bounds_nl: All upper bounds for the nonlinear mpc constraints
    all_constraints_lower_bounds_proj_nl: All lower bounds for the projection-based nonlinear mpc constraints
    all_constraints_upper_bounds_proj_nl: All upper bounds for the projection-based nonlinear mpc constraints
    safely_mpc_init_guess_solver: Casadi-based solver for obtaining an initial guess 
    safely_solver_ipopt: Casadi-based solver for the nonlinear MPC problem
    safely_mpc_projection_solver: Casadi-based solver for the projection-based nonlinear MPC problem
    discount_factors: Discount factors applied to the heading angle term in nonlinear mpc objective function
    
    
    """
    
    #
    # Stack the initial set of parameters
    #
    
    # The Q-matrices and mean positions are structured as a list of lists, where the inner list is the
    # time index and the the outer list is the obstacle index.
    obs_Q_mat_time_stacked = []
    obs_mu_bars_time_stacked = []
    for obs_ind in range(n_obstacles):
        obs_Q_mat_time_stacked.append(np.vstack(obs_Q_mat_time_hor[obs_ind]))
        obs_mu_bars_time_stacked.append(np.vstack(obs_mu_bars_time_hor[obs_ind]))
    obs_Q_mat_stacked = np.vstack(obs_Q_mat_time_stacked)
    obs_Q_mat_stacked = np.reshape( obs_Q_mat_stacked.T,((rob_state_dim**2)*n_obstacles*planning_horizon,1) )
    obs_mu_bars_stacked = np.vstack(obs_mu_bars_time_stacked)
    
    # Set the initial guess for the first optimization problem
    init_guess_state_var = np.tile(current_state,(planning_horizon,1))
    init_guess_th = np.tile(current_heading_angle,(planning_horizon,1))
    init_guess_v = np.zeros((planning_horizon,1))
    init_guess_tr = np.zeros((planning_horizon,1))
    init_guess = np.vstack((init_guess_state_var,init_guess_v,init_guess_th,init_guess_tr))

    #
    # Compute an optimal trajectory assuming no obstacles are present (include heading-angle component)
    # Use a naive initial guess (assume it applies no control input)
    #
    params_act_init = csi.vertcat(obs_mu_bars_stacked,obs_Q_mat_stacked,current_state,current_heading_angle,goal_state,discount_factors)
    sol_init_mpc = safely_mpc_init_guess_solver(
                lbx = all_dv_lower_bounds_nl,
                ubx = all_dv_upper_bounds_nl,
                lbg = constraints_dyn_lower_bounds_nl,
                ubg = constraints_dyn_upper_bounds_nl,
                p = params_act_init,
                x0 = init_guess)
    
    # Extract parameters from the solution of the initial problem
    obs_free_traj_func_val = float(sol_init_mpc['f'])
    robot_RHC_trajectory_initial_solution = np.array(sol_init_mpc['x'])[0:2*planning_horizon]
    robot_initial_trajectory_state_x_time = np.reshape(
        robot_RHC_trajectory_initial_solution, (planning_horizon, rob_state_dim)).T
    robot_input_initial_solution = np.array(sol_init_mpc['x'])[2*planning_horizon:3*planning_horizon]
    robot_heading_angle_sequence_initial_solution = np.array(sol_init_mpc['x'])[3*planning_horizon:4*planning_horizon]
    robot_turning_rate_sequence_initial_solution = np.array(sol_init_mpc['x'])[4*planning_horizon:5*planning_horizon]

    #
    # From the nominal trajectory, compute the Q+ matrices (outer      
    # approximation to Minkowski sum of original Q and the rigid body  
    # ball of each obstacle (of radius r_j). The ellipsoidal outer     
    # approximation is tight in our chosen direction of interest,      
    # which is the direction defined by the robot position to the mean 
    # position of the obstacle.                                        
    #
    obs_Qplus_mat_time_hor, obs_Qplus_mat_time_hor_inv = get_Qplus_mats_over_time_hor(planning_horizon, n_obstacles,
                      obs_mu_bars_time_hor, obs_Q_mat_time_hor,
                      robot_initial_trajectory_state_x_time,
                      obs_rad_vector, rob_state_dim)

    #
    # Check for collisions between our nominal trajectory
    # and the obstacle ellipsoidal outer approximations
    #
    collision_flag_list = check_collisions(robot_initial_trajectory_state_x_time,
                                   obs_mu_bars_time_hor, obs_Qplus_mat_time_hor,
                                   n_obstacles)

    # If there were collisions, use DC procedure to adjust nominal trajectory to prevent them
    if any(collision_flag_list):

        #
        # Use sequence of ipopt- and sqp-based approaches to obtain the collision-free
        # trajectory
        #
        
        # Stack the obstacle Q+ matrices to put in parameter vector
        # NOTE THAT WE USE THE INVERSES OF THE QPLUS MATRICES!
        obs_Qplus_mat_time_stacked = []
        for obs_ind in range(n_obstacles):
            obs_Qplus_mat_time_stacked.append(np.vstack(obs_Qplus_mat_time_hor_inv[obs_ind]))
        obs_Qplus_mat_stacked = np.vstack(obs_Qplus_mat_time_stacked)
        obs_Qplus_mat_stacked = np.reshape( obs_Qplus_mat_stacked.T,((rob_state_dim**2)*n_obstacles*planning_horizon,1) )

        params_act_nonlinear = np.vstack((obs_mu_bars_stacked,obs_Qplus_mat_stacked,\
            current_state,current_heading_angle,goal_state,discount_factors))
        # Using ipopt to get guess
        sol_ipopt = safely_solver_ipopt(
            lbx=all_dv_lower_bounds_nl,
            ubx=all_dv_upper_bounds_nl,
            lbg=all_constraints_lower_bounds_nl,
            ubg=all_constraints_upper_bounds_nl,
            x0 = np.array(sol_init_mpc['x']),
            p=params_act_nonlinear)
            
        # Project this trajectory onto each of the obstacles, use these constraints: preserves duality-based argument.
        
        # Trajectory to use for projection
        nominal_trajectory = np.array(sol_ipopt['x'][0:2*planning_horizon])
        
        # Initialize the stack of hyperplanes
        hyperplane_A = []
        hyperplane_b = []

        tmp = 1/np.sqrt(2)
        obj_half_cur = np.linalg.cholesky(2*np.eye((rob_state_dim))).T  
        for obs_ind in range(n_obstacles):
            
            # Initialize for the current stack
            G_quad = []
            h_quad = []
            G_lin = []
            h_lin = []

            # Iterate through all time steps and append current time step variable
            for t_step in range(planning_horizon):

                # Extract the current system parameters
                ell_cov_cur = obs_Qplus_mat_time_hor[obs_ind][t_step]
                ell_cen_cur = obs_mu_bars_time_hor[obs_ind][t_step]
                ag_traj_cur = nominal_trajectory[rob_state_dim*t_step:rob_state_dim*(t_step+1)]

                # Perform necessary operations on the data
                f_obj_cur = -2*ag_traj_cur / np.sqrt(2)
                f_con_cur = -2*ell_cov_cur @ ell_cen_cur / np.sqrt(2)
                ell_cov_half_cur = np.linalg.cholesky(2*ell_cov_cur).T

                # Current linear cone elements
                G_lin_cur = np.zeros((1,4*planning_horizon))
                G_lin_cur[0,3*planning_horizon + t_step] = 1.0
                h_lin_cur = np.array([[1.0 - ell_cen_cur.T @ ell_cov_cur @ ell_cen_cur]])

                # current quadratic cone for the objective term
                G_quad1_cur = np.zeros((1,4*planning_horizon))
                G_quad1_cur[0,2*t_step:2*(t_step+1)] = f_obj_cur.T
                G_quad1_cur[0,2*planning_horizon + t_step] = -tmp
                G_quad2_cur = np.zeros((2,4*planning_horizon))
                G_quad2_cur[:,2*t_step:2*(t_step+1)] = -obj_half_cur
                G_quad3_cur = np.zeros((1,4*planning_horizon))
                G_quad3_cur[0,2*t_step:2*(t_step+1)] = -f_obj_cur.T
                G_quad3_cur[0,2*planning_horizon + t_step] = tmp

                # current quadratic cone for the current quadratic constraint
                G_quad4_cur = np.zeros((1,4*planning_horizon))
                G_quad4_cur[0,2*t_step:2*(t_step+1)] = f_con_cur.T
                G_quad4_cur[0,3*planning_horizon + t_step] = -tmp
                G_quad5_cur = np.zeros((2,4*planning_horizon))
                G_quad5_cur[:,2*t_step:2*(t_step+1)] = -ell_cov_half_cur
                G_quad6_cur = np.zeros((1,4*planning_horizon))
                G_quad6_cur[0,2*t_step:2*(t_step+1)] = -f_con_cur.T
                G_quad6_cur[0,3*planning_horizon + t_step] = tmp

                # Current right-hand-side of quadratic cone (stacked)
                h_quad_cur = np.vstack((np.array([tmp]), np.zeros((2,1)), np.array([tmp]),\
                np.array([tmp]), np.zeros((2,1)), np.array([tmp])))

                # Stack everything
                G_quad_cur = np.vstack((G_quad1_cur,G_quad2_cur,G_quad3_cur,\
                G_quad4_cur,G_quad5_cur,G_quad6_cur))
                G_quad.append(G_quad_cur)
                h_quad.append(h_quad_cur)
                G_lin.append(G_lin_cur)
                h_lin.append(h_lin_cur)

            # Put everything together across all time steps
            G_quad = np.vstack(G_quad)
            h_quad = np.vstack(h_quad)
            G_lin = np.vstack(G_lin)
            h_lin = np.reshape(np.vstack(h_lin),(planning_horizon,1))
            G = csc(np.vstack((G_lin,G_quad)))
            h = np.reshape(np.vstack((h_lin,h_quad)),(9*planning_horizon,))
            c = np.squeeze(np.vstack((np.zeros((2*planning_horizon,1)),np.ones((planning_horizon,1)),np.zeros((planning_horizon,1)))))
            dims = {'l':planning_horizon,'q':[4 for t_step in range(2*planning_horizon)]}
            sol = ecos.solve(c,G,h,dims,verbose=False)

            # Compute the hyperplanes for the current obstacle at each time step
            projection_pts = np.array(sol['x'][0:2*planning_horizon])
            projection_pts_x_vals = projection_pts[::2]
            projection_pts_y_vals = projection_pts[1::2]
            # For each time step, append the current A and b linear inequality components
            for t_step in range(planning_horizon):
                proj_pt_cur = np.array([[projection_pts_x_vals[t_step]],[projection_pts_y_vals[t_step]]])
                ell_mean_pos_cur = obs_mu_bars_time_hor[obs_ind][t_step]
                A_cur = -(proj_pt_cur - ell_mean_pos_cur).T @ obs_Qplus_mat_time_hor[obs_ind][t_step]
                b_cur = A_cur @ proj_pt_cur
                hyperplane_A.append(A_cur)
                hyperplane_b.append(b_cur)

        hyperplane_A = np.vstack(hyperplane_A)
        hyperplane_A_stacked = np.reshape(hyperplane_A.T,(2*planning_horizon*n_obstacles,1))
        hyperplane_b = np.vstack(hyperplane_b)

        # Use SQP-based approach tocheck_collisions get meaningful result
        params_act_projection_mpc = np.vstack((obs_mu_bars_stacked,obs_Qplus_mat_stacked,hyperplane_A_stacked,\
            hyperplane_b,current_state,current_heading_angle,goal_state,discount_factors))
        sol = safely_mpc_projection_solver(
            lbx=all_dv_lower_bounds_nl,
            ubx=all_dv_upper_bounds_nl,
            lbg=all_constraints_lower_bounds_proj_nl,
            ubg=all_constraints_upper_bounds_proj_nl,
            x0 = np.array(sol_ipopt['x']),
            p=params_act_projection_mpc)
        
        # Dual variables
        dual_variables_hyperplanes = np.array(sol['lam_g'][0:n_obstacles*planning_horizon])
        robot_RHC_trajectory = np.array(sol['x'])[0:rob_state_dim*planning_horizon]
        robot_input_sequence = np.array(sol['x'])[rob_state_dim*planning_horizon:(rob_state_dim+rob_input_dim)*planning_horizon]
        robot_heading_angle_sequence = np.array(sol['x'])[(rob_state_dim+rob_input_dim)*planning_horizon:(rob_state_dim+rob_input_dim+1)*planning_horizon]
        robot_turning_rate_sequence = np.array(sol['x'])[(rob_state_dim+rob_input_dim+1)*planning_horizon:(rob_state_dim+rob_input_dim+2)*planning_horizon]
        obs_func_val = float(sol['f'])

        #
        # Check to see if our updated trajectory has
        # avoided obstacle collisions                
        #
        robot_RHC_trajectory_state_x_time = np.reshape(
            robot_RHC_trajectory, (planning_horizon, rob_state_dim)).T

        collision_flag_list = check_collisions(
            robot_RHC_trajectory_state_x_time, obs_mu_bars_time_hor,
            obs_Qplus_mat_time_hor, n_obstacles)
        if any(collision_flag_list):
            # raise RuntimeError('\n\nWARNING!!! Collision not resolved!\n\n')
            # robot_RHC_trajectory = None
            # robot_input_sequence = None
            # robot_heading_angle_sequence = None
            # robot_turning_rate_sequence = None
            # obs_cons_dual_variables = None
            # obs_func_val = np.nan


            # Store information about dual variables at current time step
            obs_cons_dual_variables = np.asarray(dual_variables_hyperplanes)

        else:
        
            # Use the resolved solution

            # Store information about dual variables at current time step
            obs_cons_dual_variables = np.asarray(dual_variables_hyperplanes)

    else:
        #
        # If no collisions detected, use nominal trajectory
        #
        robot_RHC_trajectory_state_x_time = \
            robot_initial_trajectory_state_x_time
        robot_RHC_trajectory = robot_RHC_trajectory_initial_solution
        robot_input_sequence = robot_input_initial_solution
        robot_heading_angle_sequence = robot_heading_angle_sequence_initial_solution
        robot_turning_rate_sequence = robot_turning_rate_sequence_initial_solution
        obs_func_val = obs_free_traj_func_val
        obs_cons_dual_variables = np.zeros((n_obstacles * planning_horizon, 1))

    return robot_RHC_trajectory,robot_input_sequence,robot_heading_angle_sequence,\
        robot_turning_rate_sequence,obs_func_val,obs_cons_dual_variables,obs_Qplus_mat_time_hor