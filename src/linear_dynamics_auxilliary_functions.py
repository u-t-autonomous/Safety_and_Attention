import numpy as np
from numpy import linalg as LA


def ellipsoid_level_function(obs_center, obs_matrix, query_state):
    """
    Compute the level set that defines the obstacle ellipsoid
            {x| (x - obs_center)' @ obs_matrix @ (x - obs_center) <= 1}
    It evaluates h(z) = 1 - (z - obs_center)' obs_matrix (z - obs_center)

    If this evaluation is positive, then the query_state is within the
    ellipsoid.

    :param obs_center: center of the current ellipse obstacle of interest
    :param obs_matrix: quadratic form of the current ellipse obstacle of
        interest.
    :param query_state_vector:
    :return: Evaluation of h
    """

    return 1 - np.matmul((query_state-obs_center).T, np.matmul(obs_matrix, (query_state - obs_center)))


def check_collisions(future_trajectory, obs_mu_bars_time_hor, obs_Qplus_mat_time_hor,
                     n_obstacles, relaxation=1e-3):
    """
    Check for collisions with obstacles that are ellipses
    ---
    :param future_trajectory: Robot trajectory (state_dimension x time_horizon)
    :param obs_center_vector: Obstacle centers, each row is a center (2D matrix)
    :param obs_matrix_vector: Obstacle quadratic form matrix, each row is a
        matrix (3D matrix)
    :param relaxation: Relax the ellipsoidal constraint satisfaction for
        numerical stability
    ---
    :return: collision_flag_list of length time_horizon whose t-th element is
        set to True when a collision is detected for the given trajectory at
        time t
    """
    # This structure is assumed
    robot_state_dimension, time_horizon = future_trajectory.shape
    # Initialization
    collision_flag_list = [True] * time_horizon
    for time_index in range(time_horizon):
        # # Query state
        # future_trajectory_state = np.reshape(np.transpose(future_trajectory[:, time_index]),(robot_state_dimension,1))
        future_trajectory_state = np.reshape(future_trajectory[:, time_index],(robot_state_dimension,1))

        # List of ellipsoidal evaluations for all obstacles
        ellipse_equation_vals = np.zeros((n_obstacles,))
        for obs_index in range(n_obstacles):
            obs_matrix_inv = LA.inv(obs_Qplus_mat_time_hor[obs_index][time_index])
            obs_center = obs_mu_bars_time_hor[obs_index][time_index]
            ellipse_equation_vals[obs_index] = ellipsoid_level_function(
                obs_center, obs_matrix_inv, future_trajectory_state)
        # We collide if the evaluation is positive (permit some incursion by
        # considering a relaxation, a positive number)
        collision_flag_list[time_index] = \
            np.any(ellipse_equation_vals >= relaxation)
    return collision_flag_list

def continue_condition(current_state, target_state, target_tolerance):
    """
    Check if we must continue (current state is far away from the target state)

    :param current_state:
    :param target_state:
    :param target_tolerance:
    :return:
    """
    return np.linalg.norm(target_state - current_state) >= target_tolerance


def get_mu_sig_over_time_hor(planning_horizon, num_lin_obs, obs_mu_bars_t,
                             obs_sig_bars_t, lin_obs_list):
    """
    Propagate the linear obstacles' mean positions and position covariances
    over the planning horizon
    ---
    param planning_horizon: Number of time steps that the robotic agent plans into
        the future
    param num_lin_obs: Number of linear obstacles in the robotic agent's environment
    param obs_mu_bars_t: Linear obstacle mean positions at the current time step
    param obs_sig_bars_t: Linear obstacle mean position covariance at current time step
    param lin_obs_list: List of linear obstacle objects in the robotic agent's environment
    ---
    return obs_mu_bars_time_hor: List of obstacle mean positions over the planning horizon
    return obs_sig_bars_time_hor: List of obstacle position covariances over the planning horizon
    """

    # Initialize lists to store this information
    obs_mu_bars_plan_hor = [[[] for i in range(planning_horizon)] for j in range(num_lin_obs)]
    obs_sig_bars_plan_hor = [[[] for i in range(planning_horizon)] for j in range(num_lin_obs)]

    # In the first time step, simply use the current values
    for j in range(num_lin_obs):
        obs_mu_bars_plan_hor[j][0] = obs_mu_bars_t[j]
        obs_sig_bars_plan_hor[j][0] = obs_sig_bars_t[j]

    # Iterate over obstacles, remaining time horizon to fill out the \bar{mu}[t] and \bar{sig}[t] over the time
    # horizon considered
    for j in range(num_lin_obs):
        cur_obs = lin_obs_list[j]
        for t in range(1, planning_horizon):
            cur_mu = obs_mu_bars_plan_hor[j][t - 1]
            cur_sig = obs_sig_bars_plan_hor[j][t - 1]
            obs_mu_bars_plan_hor[j][t] = np.matmul(cur_obs.A_matrix,cur_mu) + np.matmul(cur_obs.F_matrix,cur_obs.w_mean_vec)
            obs_sig_bars_plan_hor[j][t] = np.matmul(cur_obs.A_matrix,np.matmul(cur_sig,np.transpose(cur_obs.A_matrix)))\
                                          + np.matmul(cur_obs.F_matrix,np.matmul(cur_obs.w_cov_mat,np.transpose(cur_obs.F_matrix)))

    return obs_mu_bars_plan_hor, obs_sig_bars_plan_hor


def get_mu_sig_theta_over_time_hor(planning_horizon, num_nonlin_obs, obs_mu_bars_t,
                                   obs_sig_bars_t, nonlin_obs_list, obs_theta_t, sampling_time):
    """
    Propagate the nonlinear obstacles over the planning horizon
    ---
    param planning_hor: Number of time steps into the future the robotic agent plans
    param num_linlin_obs: Number of nonlinear obstacles in the environment of the
        robotic agent
    param obs_mu_bars_t: Current mean position of each nonlinear obstacle
    param obs_sig_bars_t: Current position covariance of each nonlinear obstacle
    param nonlin_obs_list: List of nonlinear obstacle objects
    param obs_theta_t: Current heading angle of each nonlinear obstacle
    param sampling_time: Time between subsequent time steps "k" and "k+1"
    ---
    return obs_mu_bars_plan_hor: Array of nonlinear obstacle mean positions over the
        course of the planning horizon
    return obs_sig_bars_plan_hor: Array of nonlinear obstacle position covariance
        matrices over the course of the time horizon
    return obs_theta_plan_hor: Array of nonlinear obstacle heading angles over the
        course of the time horizon
    """

    # Initialize lists to store this information
    obs_mu_bars_plan_hor = [[[] for i in range(planning_horizon)] for j in range(num_nonlin_obs)]
    obs_sig_bars_plan_hor = [[[] for i in range(planning_horizon)] for j in range(num_nonlin_obs)]
    obs_theta_plan_hor = [[[] for i in range(planning_horizon)] for j in range(num_nonlin_obs)]

    # In the first time step, simply use the current values
    for j in range(num_nonlin_obs):
        obs_mu_bars_plan_hor[j][0] = obs_mu_bars_t[j]
        obs_sig_bars_plan_hor[j][0] = obs_sig_bars_t[j]
        obs_theta_plan_hor[j][0] = obs_theta_t[j]

    # Iterate over obstacles, remaining time horizon to fill out the \bar{mu}[t] and \bar{sig}[t] over the time
    # horizon considered
    for j in range(num_nonlin_obs):
        cur_obs = nonlin_obs_list[j]
        for t in range(1, planning_horizon):
            B_0 = sampling_time * np.array([np.cos(obs_theta_plan_hor[j][t - 1]),
                                            np.sin(obs_theta_plan_hor[j][t - 1])])
            cur_mu = obs_mu_bars_plan_hor[j][t - 1]
            cur_sig = obs_sig_bars_plan_hor[j][t - 1]
            obs_mu_bars_plan_hor[j][t] = np.matmul(cur_obs.A_matrix,cur_mu) + np.matmul(B_0,cur_obs.w_mean_vec)
            obs_sig_bars_plan_hor[j][t] = np.matmul(cur_obs.A_matrix,np.matmul(cur_sig,np.transpose(cur_obs.A_matrix)))\
                                          + np.matmul(B_0,np.matmul(cur_obs.w_cov_mat,np.transpose(B_0)))
            obs_theta_plan_hor[j][t] = obs_theta_plan_hor[j][t - 1] + sampling_time*cur_obs.gamma

    return obs_mu_bars_plan_hor, obs_sig_bars_plan_hor, obs_theta_plan_hor


def get_Q_mats_over_time_hor(planning_horizon, num_obs, beta, obs_list, obs_sig_bars_plan_hor):
    """
    Return a list of the Q matrices (superlevel sets of Gaussian) of each obstacle
        over the time horizon
    ---
    param planning_horizon: Number of time steps into the future the robotic agent considers
        when constructing its trajectory to follow
    param num_obs: Number of obstacles (either number of linear or number of nonlinear) in
        the environment of the robotic agent
    param beta: Minimum probability for robotic agent to not intersect with the obstacle
    param obs_list: List of (non)linear obstacle objects
    param obs_sig_bars_time_hor: list of (non)linear obstacle position covariances over
        the planning horizon
    ---
    return obs_Q_mat_time_hor: Array of Q matrices over the course of the planning horizon
        for each (non)linear obstacle
    """

    # Initialize lists to store Q matrices
    obs_Q_mat_plan_hor = [[[] for i in range(planning_horizon)] for j in range(num_obs)]
    for j in range(num_obs):
        cur_obs = obs_list[j]
        for t in range(planning_horizon):
            sig_cur = obs_sig_bars_plan_hor[j][t]
            # TODO: See if this sqrt_term is still necessary after changing code
            sqrt_term = max(1e-12, LA.det(2 * np.pi * sig_cur))  # If current Sig is zero, may have numerical errors
            obs_Q_mat_plan_hor[j][t] = -2 * np.log((beta * np.sqrt(sqrt_term)) /
                (num_obs * planning_horizon * np.pi * cur_obs.radius ** 2)) * sig_cur

    return obs_Q_mat_plan_hor


def get_Qplus_mats_over_time_hor(planning_horizon, num_obs, obs_mu_bars_plan_hor, obs_Q_mat_plan_hor,
                                 robot_initial_trajectory_state_x_time, obs_rad_vector, rob_state_dim):
    """
    Obtain the Q+ matrices, which are the outer ellipsoidal outer approximation that are
        tight in our specified direction of interest
    ---
    param planning_horizon: Number of time steps into the future that the robotic agent
        considers when planning its trajectory
    param num_obs: Total number of obstacles (both linear and nonlinear) in the environment
    param obs_mu_bars_plan_hor: Array of obstacle mean positions over the course of the
        planning horizon
    param obs_Q_mat_plan_hor: Array of obstacle Q matrices over the course of the
        planning horizon
    param robot_initial_trajectory_state_x_time: Robotic agent's nominal trajectory, does
        not consider obstacle positions when constructing
    param obs_rad_vector: Concatenated array of linear and nonlinear obstacle radii
    param rob_state_dim: State dimension of the environment of the robotic agent
    ---
    return obs_Qplus_mat_time_hor: Array of Qplus ellipsoids for each obstacle over
        the course of the planning horizon
    """
    # Initialize the Q+ matrices
    obs_Qplus_mat_time_hor = [[[] for i in range(planning_horizon)] for j in range(num_obs)]
    obs_Qplus_mat_time_hor_inv = [[[] for i in range(planning_horizon)] for j in range(num_obs)]

    # Iterate over each obstacle over the time horizon to determine the corresponding Qplus matrix
    for j in range(num_obs):
        r_j = obs_rad_vector[j]
        for t in range(planning_horizon):
            Q_cur = obs_Q_mat_plan_hor[j][t]
            rob_sta_cur = np.reshape(robot_initial_trajectory_state_x_time[:, t],(rob_state_dim,1))
            mu_cur = obs_mu_bars_plan_hor[j][t]
            l0_bar = mu_cur - rob_sta_cur
            sqrt_term = np.sqrt(np.matmul(l0_bar.T,np.matmul(Q_cur,l0_bar)))
            l0_norm = LA.norm(l0_bar)
'''
            obs_Qplus_mat_time_hor[j][t] = np.linalg.inv((sqrt_term + r_j*l0_norm)*\
                (Q_cur/sqrt_term + r_j/l0_norm*np.eye(rob_state_dim)))
'''
            obs_Qplus_mat_time_hor[j][t] = (sqrt_term + r_j*l0_norm)*\
                (Q_cur/sqrt_term + r_j/l0_norm*np.eye(rob_state_dim))
            obs_Qplus_mat_time_hor_inv[j][t] = np.linalg.inv(obs_Qplus_mat_time_hor[j][t])

    return obs_Qplus_mat_time_hor, obs_Qplus_mat_time_hor_inv


def propagate_linear_obstacles(num_lin_obs, lin_obs_list):
    """
    Propagate the linear obstacles' positions and position covariances to
    one time step in the future.
    ---
    param num_lin_obs: number of linear obstacles in the environment
    param lin_obs_list: list of linear obstacle objects in the environment
    ---
    return linear_obs_mu_bars_t_new: new array of linear obstacle mean positions
    return linear_obs_sig_bars_t_new: new array of linear obstacle position covariances
    """
    linear_obs_mu_bars_t_new = [[] for j in range(num_lin_obs)]
    linear_obs_sig_bars_t_new = [[] for j in range(num_lin_obs)]
    for j in range(num_lin_obs):
        cur_obs = lin_obs_list[j]
        linear_obs_mu_bars_t_new[j] = np.matmul(cur_obs.A_matrix,cur_obs.sim_position)\
                                    + np.matmul(cur_obs.F_matrix,cur_obs.w_mean_vec)
        linear_obs_sig_bars_t_new[j] = np.matmul(cur_obs.A_matrix,np.matmul(cur_obs.sig_mat,np.transpose(cur_obs.A_matrix)))\
                                    + np.matmul(cur_obs.F_matrix,np.matmul(cur_obs.w_cov_mat,np.transpose(cur_obs.F_matrix)))
    linear_obs_mu_bars_t_new = linear_obs_mu_bars_t_new[:]
    linear_obs_sig_bars_t_new = linear_obs_sig_bars_t_new[:]

    return linear_obs_mu_bars_t_new, linear_obs_sig_bars_t_new


def propagate_nonlinear_obstacles(num_nonlin_obs, sampling_time, nonlin_obs_list):
    """
    Propagate the nonlinear obstacles to one time step in the future (for determining
        trajectory of the robotic agent)
    ---
    param num_nonlin_obs: Number of nonlinear obstacles in the environment of the
        robotic agent
    param sampling_time: Time between successive steps "k" and "k+1"
    param nonlin_obs_list: List of nonlinear obstacle objects
    ---
    return nonlinear_obs_mu_bars_t: Array of nonlinear obstacle mean positions at
        the current time step
    return nonlinear_obs_sig_bars_t: Array of nonlinear obstacle position covariances
        at the current time step
    return nonlinear_obs_theta_t: Array of nonlinear obstacle heading angles at
        the current time step
    """
    nonlinear_obs_mu_bars_t_new = [[] for j in range(num_nonlin_obs)]
    nonlinear_obs_theta_t_new = [[] for j in range(num_nonlin_obs)]
    nonlinear_obs_sig_bars_t_new = [[] for j in range(num_nonlin_obs)]
    for j in range(num_nonlin_obs):
        cur_obs = nonlin_obs_list[j]
        B_0 = sampling_time * np.array([np.cos(cur_obs.theta), np.sin(cur_obs.theta)])
        nonlinear_obs_mu_bars_t_new[j] = np.matmul(cur_obs.A_matrix,cur_obs.sim_position) + np.matmul(B_0,cur_obs.w_mean_vec)
        nonlinear_obs_sig_bars_t_new[j] = np.matmul(cur_obs.A_matrix,np.matmul(cur_obs.sig_mat,np.transpose(
            cur_obs.A_matrix))) + np.matmul(B_0,np.matmul(cur_obs.w_cov_mat,np.transpose(B_0)))
        nonlinear_obs_theta_t_new[j] = cur_obs.theta + sampling_time * cur_obs.gamma
    nonlinear_obs_mu_bars_t = nonlinear_obs_mu_bars_t_new[:]
    nonlinear_obs_sig_bars_t = nonlinear_obs_sig_bars_t_new[:]
    nonlinear_obs_theta_t = nonlinear_obs_theta_t_new[:]

    return nonlinear_obs_mu_bars_t, nonlinear_obs_theta_t, nonlinear_obs_sig_bars_t