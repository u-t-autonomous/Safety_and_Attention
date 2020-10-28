import numpy as np
import time as time
import cvxpy as cp
import ecos

def create_dc_motion_planner(Z, H, time_horizon, target_state, robot_state_max,
                             robot_input_max,n_obstacles,mult_matrix_stack):
    """
    Use difference of convex programming to enforce obstacle avoidance criterion

    This implementation is based on Lipp and Boyd, "Variations and extension of
    the convex-concave procedure", 2016.

    :param Z:
    :param H:
    :param time_horizon:
    :param target_state:
    :param robot_state_max:
    :param robot_input_max:
    :param obs_matrix_vector:
    :param obs_center_vector:
    :return: dc_motion_planner          - CVXPY Problem object
             robot_trajectory           - CVXPY Variable that specifies the
                                          robot trajectory
                                          (robot_state_dimension*time_horizon,)
             robot_input_sequence       - CVXPY Variable that specifies the
                                          input sequence
                                          (robot_input_dimension*time_horizon,)
             slack_variables            - CVXPY Variable to enforce the
                                          non-convex constraint
                                          (n_obstacles*time_horizon,)
             robot_current_state        - CVXPY Parameter that specifies the
                                          robot's current state in the receding
                                          horizon control
                                          (robot_state_dimension,)
             previous_robot_trajectory  - CVXPY Parameter to specify the
                                          previous trajectory for difference
                                          of convex programming
                                          (robot_state_dimension*time_horizon,)
             tau                        - CVXPY Parameter to penalize the sum
                                          of slack_variables for difference
                                          of convex programming (scalar)
    -
    """
    robot_state_dimension = Z.shape[1]
    robot_input_dimension = int(H.shape[1]/time_horizon)

    # # Parameters
    obs_Qplus_stack = cp.Parameter((robot_state_dimension*time_horizon*n_obstacles,
                                     robot_state_dimension*time_horizon*n_obstacles))
    obs_mu_bar_stack = cp.Parameter((robot_state_dimension*time_horizon*n_obstacles,))

    # Robot current state
    robot_current_state = cp.Parameter((robot_state_dimension,))
    # Previous robot trajectory
    previous_robot_trajectory = \
        cp.Parameter((time_horizon*robot_state_dimension,))
    # Scaling of the objective
    tau = cp.Parameter((1,), nonneg=True)

    # # Variables
    # Robot trajectory
    robot_trajectory = cp.Variable((time_horizon*robot_state_dimension,))
    # Robot input vector
    robot_input_sequence = cp.Variable((time_horizon*robot_input_dimension,))
    # Slack variables (for non-convex constraint enforcement)
    slack_variables = cp.Variable((time_horizon*n_obstacles,), nonneg=True)

    prev_rob_traj_stack = cp.hstack([previous_robot_trajectory for i in range(n_obstacles)])
    cur_rob_traj_stack = cp.hstack([robot_trajectory for i in range(n_obstacles)])
    # Constraints
    # 1. Dynamics
    # 2. Trajectory constraints (stay within safe set)
    # 3. Input constraints (actuation limits)
    # 4. Linearized Difference-of-convex constraints
    # const = [robot_trajectory == Z@robot_current_state + H@robot_input_sequence,
    #          robot_trajectory >= -robot_state_max,
    #          robot_trajectory <= robot_state_max,
    #          robot_input_sequence >= -robot_input_max,
    #          robot_input_sequence <= robot_input_max]
    const = [robot_trajectory == Z*robot_current_state + H*robot_input_sequence,
        robot_trajectory >= -robot_state_max,
        robot_trajectory <= robot_state_max,
        robot_input_sequence >= -robot_input_max,
        robot_input_sequence <= robot_input_max,
             1 - mult_matrix_stack*cp.diag((prev_rob_traj_stack - obs_mu_bar_stack))*obs_Qplus_stack * \
             (prev_rob_traj_stack - obs_mu_bar_stack) + \
             -2*mult_matrix_stack*cp.diag((prev_rob_traj_stack - obs_mu_bar_stack))*obs_Qplus_stack * \
             (cur_rob_traj_stack - prev_rob_traj_stack) <= slack_variables]

    # Objective: Regulation error to the target + tau*(slack variables L1-norm)
    target_trajectory = np.tile(target_state, (time_horizon,))
    displacement = robot_trajectory - target_trajectory
    #displacement_state_x_time = cp.reshape(displacement,
    #    (robot_current_state.shape[0],
    #     int(target_trajectory.shape[0] / (robot_current_state.shape[0]))))
    #objective = cp.Minimize(cp.quad_over_lin(displacement_state_x_time, 1) +
    #                        tau * cp.sum(slack_variables))
    objective = cp.Minimize(cp.norm(displacement) + tau*cp.sum(slack_variables))

    dc_motion_planner = cp.Problem(objective, const)

    return dc_motion_planner, robot_trajectory, slack_variables, \
        robot_input_sequence, robot_current_state, previous_robot_trajectory, tau, \
        obs_Qplus_stack, obs_mu_bar_stack


def solve_dc_motion_planning(dc_motion_planner, robot_current_state,
        initial_robot_trajectory, dc_tau, dc_previous_robot_trajectory,
        dc_robot_current_state, dc_robot_trajectory, dc_slack_variables,n_obstacles,time_horizon,
        dc_obs_Qplus_stack, dc_obs_mu_bar_stack, obs_Qplus_stack_all, obs_mu_bar_stack_all,
        mu=10,tau_max=np.array([10000]), delta=1e-4, verbose=False):
    """
    Solve the dc_motion_planner initialized with the trajectory
    from the obstacle-free motion planner

    This implementation is based on Lipp and Boyd, "Variations and extension of
    the convex-concave procedure", 2016.

    :param dc_motion_planner: CVXPY Problem object obtained from
        create_dc_motion_planner
    :param robot_current_state: Current state of the robot
    :param initial_robot_trajectory: Initial guess for the robot trajectory
    :param dc_tau: CVXPY parameter associated with dc_motion_planner
    :param dc_previous_robot_trajectory: CVXPY parameter associated with
        dc_motion_planner
    :param dc_robot_current_state: CVXPY parameter associated with
        dc_motion_planner
    :param dc_robot_trajectory: CVXPY variable associated with
        dc_motion_planner
    :param dc_slack_variables: CVXPY variable associated with
        dc_motion_planner
    :param mu: Larger values mean faster convergence | smaller values mean
        more exploration
    :param tau_max: Upper bound on the penalty term
    :param delta: Convergence threshold for difference of convex programming
    :param verbose: Should we update the status
    :return: None
    """

    # Prepare for the first run
    dc_tau.value = np.array([1])
    dc_previous_robot_trajectory.value = initial_robot_trajectory
    dc_robot_current_state.value = robot_current_state
    dc_obs_Qplus_stack.value = obs_Qplus_stack_all
    dc_obs_mu_bar_stack.value = obs_mu_bar_stack_all

    # prob_data, ph1, ph2 = dc_motion_planner.get_problem_data(cp.ECOS)
    # prob_data_c = prob_data["c"]
    # prob_data_h = prob_data["h"]
    # prob_data_b = prob_data["b"]
    # prob_data_A = prob_data["A"]
    # prob_data_G = prob_data["G"]
    # prob_data_dims = prob_data["dims"]
    dc_motion_planner.solve(solver='ECOS', verbose=False)
    if dc_motion_planner.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError('CVXPY failed with status: {:s}!'.format(
            dc_motion_planner.status))

    # Repeat until convergence
    iteration_count = 0
    solver_times = []
    num_active_duals_per_iter = []
    while True:
        tic = time.clock()
        # Save the existing optimal value
        previous_optimal_value = dc_motion_planner.value
        # Update the parameters and solve the planner
        dc_robot_current_state.value = robot_current_state
        dc_tau.value = min([tau_max, mu * dc_tau.value])
        dc_previous_robot_trajectory.value = dc_robot_trajectory.value
        dc_motion_planner.solve(solver='ECOS',warm_start=True,verbose=False)
        obs_dual_ph = dc_motion_planner.constraints[-1].dual_value
        num_active_duals_per_iter.append(np.sum((np.asarray(obs_dual_ph) >= 1e-3).astype(int)))
        solver_times.append(dc_motion_planner.solver_stats.solve_time)
        # Convergence criterion
        if dc_motion_planner.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError('CVXPY failed with status: {:s}!'.format(
                dc_motion_planner.status))
        # Print the optimization status, if verbose
        if verbose:
            print_string = '{:d}. Current optimal value: {:1.2f} | Previous ' \
                           'optimal value: {:1.2f} | Slack sum: {:1.2f} | ' \
                           'CVXPY status: {:s}'
            print(print_string.format(iteration_count, dc_motion_planner.value,
                previous_optimal_value, cp.sum(dc_slack_variables).value,
                dc_motion_planner.status))

        if abs(previous_optimal_value - dc_motion_planner.value) <= delta:
            break
        else:
            iteration_count += 1

        toc = time.clock()

    # Access the dual variables corresponding to obstacle constraints
    obs_dual_vals = dc_motion_planner.constraints[-1].dual_value

    return obs_dual_vals, iteration_count, solver_times, num_active_duals_per_iter

def create_cvxpy_motion_planner_obstacle_free(Z, H, time_horizon, target_state,
                                robot_state_max, robot_input_max):
    """
    Plan a trajectory ignoring the obstacles => A simple quadratic program
    that minimizes the quadratic tracking (regulation) error while subject to
    linear dynamics of the robot and the constraints on the robot state and
    input

    :param Z:
    :param H:
    :param time_horizon:
    :param target_state:
    :param robot_state_max:
    :param robot_input_max:
    :param n_obstacles:
    :return:
    """
    robot_state_dimension = Z.shape[1]
    robot_input_dimension = int(H.shape[1]/time_horizon)

    # # Parameter
    # Robot current state
    robot_current_state = cp.Parameter((robot_state_dimension,))
    # Robot trajectory
    robot_trajectory = cp.Variable((time_horizon * robot_state_dimension,))
    # Robot input vector
    robot_input_sequence = cp.Variable((time_horizon * robot_input_dimension,))

    # Constraints
    # 1. Dynamics
    # 2. Trajectory constraints (stay within safe set)
    # 3. Input constraints (actuation limits)
    const = [robot_trajectory == Z*robot_current_state + H*robot_input_sequence,
             robot_trajectory >= -robot_state_max,
             robot_trajectory <= robot_state_max,
             robot_input_sequence >= -robot_input_max,
             robot_input_sequence <= robot_input_max]

    # Objective: Regulation error to the target
    target_trajectory = np.tile(target_state, (time_horizon,))
    displacement = robot_trajectory - target_trajectory
    displacement_state_x_time = cp.reshape(displacement,
        (robot_current_state.shape[0],
         int(target_trajectory.shape[0] / robot_current_state.shape[0])))
    objective = cp.Minimize(cp.quad_over_lin(displacement_state_x_time, 1))

    cvxpy_motion_planner = cp.Problem(objective, const)

    return cvxpy_motion_planner, robot_current_state, robot_trajectory, \
           robot_input_sequence