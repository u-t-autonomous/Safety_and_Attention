import ecos
import numpy as np
import time as time
from scipy.linalg import block_diag as block_diag_nonsparse
from scipy.sparse import bmat
from scipy.sparse import block_diag
from scipy.sparse import hstack
from scipy.sparse import identity
from scipy.sparse import spdiags
from scipy.sparse import vstack
from scipy.sparse import csr_matrix as csr
from scipy.sparse import csc_matrix as csc

#
# HOW TO WRITE IN ECOS STANDARD FORM:
# FOR A VERY SIMPLE QUADRATIC CONE, ||x||_2 <= t, x = [x1,x2]'
#
#     [  0    0  -1/sqrt(2)  ]          [ 1/sqrt(2) ]
# G = [ -1    0       0      ],     h = [     0     ]
#     [  0   -1       0      ]          [     0     ]
#     [  0    0   1/sqrt(2)  ]          [ 1/sqrt(2) ]
#
#
# ECOS requires the form:
# minimize c'x
#     s.t. Ax = b
#          Gx <=_K h     (i.e., Gx-h belongs to the cone K)
#
# Bypass CVXPY and call ECOS directly.
def dc_motion_planning_call_ecos(n_obstacles,time_horizon, robot_state_dimension,
                                robot_input_dimension, Z, current_state,obs_free_traj,
                                obs_mu_bar_stack, obs_Qplus_stack, mult_matrix_stack,
                                A, G_linear_partial_stack, G_quadratic,
                                h_linear_partial_stack, h_quadratic,
                                target_tile,tau_max,mu,delta,tau):

    # For ease of notation
    r = robot_state_dimension
    i = robot_input_dimension
    h = time_horizon
    n = n_obstacles
    total_num_vars = r * h + i * h + n * h + 1

    # Reshape some of the variables
    obs_mu_bar_stack = np.reshape(np.asarray(obs_mu_bar_stack),(r*n*h,1))
    obs_Qplus_stack = np.asarray(obs_Qplus_stack)

    # Right-hand side of equality constraints (requires information
    # about the current state, so cannot precompute)
    b = np.reshape((np.matmul(Z,current_state)),(r*h,))

    # Store information about each loop iteration
    iteration_count = 0
    solver_times = []
    num_active_duals_per_iter = []

    # Just to get through the initial run...
    prev_obs_func_val = -1

    # Repeat until convergence
    while True:

        tic = time.clock()

        if iteration_count == 0:
            prev_traj = np.reshape(np.asarray(obs_free_traj),(r*h,1))
        else:
            prev_traj = np.reshape(traj,(r*h,1))
        prev_traj_stack = np.tile(prev_traj,(n_obstacles,1))

        # Vectors for A matrix
        cur_stack = csc(np.matmul(mult_matrix_stack),spdiags((prev_traj_stack - obs_mu_bar_stack).T,0,r*n*h,
                                                r*n*h),csc(obs_Qplus_stack))
        prev_stack = csc(np.tile(([1]),(n*h,1))) + np.matmul(cur_stack,csc(prev_traj_stack + obs_mu_bar_stack))

        # Need to get cur_stack (dimension currently n*h x n*h*r. Need to make n*h x h*r.
        cur_stack_proper_dims = sum(-2*cur_stack[:,obs*r*h:(obs+1)*r*h] for obs in range(n_obstacles))

        # Linear objective function constraint values
        c = np.reshape(np.concatenate((-2*target_tile,np.zeros((i*h,1)),
                                        tau*np.ones((n*h,1)),np.ones((1,1))),axis=0),(total_num_vars,))

        # Right-hand side of equality constraints
        h_linear_6 = (-prev_stack.reshape((n*h,1))).toarray()
        h_linear = np.concatenate((h_linear_partial_stack,h_linear_6),axis=0)

        # Remaining portions of the A matrix
        G_linear_6 = hstack((csc(cur_stack_proper_dims),csc((n*h,i*h)),-identity(n*h),csc((n*h,1))))
        G_linear = vstack([G_linear_partial_stack, G_linear_6])

        # Concatenate linear and quadratic conic constraints
        G_total = csc(vstack([G_linear,G_quadratic]))
        h_total = np.concatenate((h_linear,h_quadratic),axis=0)
        h_total = np.reshape(h_total, (h_total.shape[0],))

        # Call ECOS
        dims = {"l": G_linear.shape[0], "q": [G_quadratic.shape[0]]}
        solution = ecos.solve(c, G_total, h_total, dims, A, b,verbose=False)
        sol_x = solution['x']
        sol_z = solution['z']

        # Get the desired objective function (norm, not norm squared)

        # Solution parameters that we care about: the updated trajectory, the input, the dual variables,
        # and the cost function. We iterate until the cost function has converged.
        traj = sol_x[0 : r*h]
        input = sol_x[r*h : r*h + i*h]
        slack_vars = sol_x[r*h + i*h : r*h + i*h + n*h]
        quad_cone_var = sol_x[-1]
        obs_func_val = np.sqrt(quad_cone_var - np.matmul(2*target_tile.T,traj) + np.matmul(target_tile.T,target_tile)) \
                       + tau*np.sum(slack_vars)
        dual_vars_dc = sol_z[G_linear_partial_stack.shape[0]:G_linear.shape[0]]

        # Append the number of active dual variables
        num_active_duals_per_iter.append(np.sum((dual_vars_dc >= 1e-3).astype(int)))

        # Convergence criterion
        if abs(prev_obs_func_val - obs_func_val) <= delta:
            break
        else:
            iteration_count += 1

        # Update the parameter for tau
        tau = min([tau_max, mu * tau])

        # Save the existing optimal value
        prev_obs_func_val = obs_func_val

        # Time for this iteration
        toc = time.clock()
        solver_times.append(toc - tic)

    obs_dual_vals = dual_vars_dc

    return obs_dual_vals, iteration_count, solver_times, num_active_duals_per_iter, traj, input

def stack_params_for_ecos(n_obstacles, time_horizon, obs_mu_bars_time_hor,obs_Qplus_mat_time_hor):
    """
    :param n_obstacles: total number of obstacles (linear and nonlinear)
    :param time_horizon: number of time steps into the future we plan
    :param obs_mu_bars_time_hor: mean positions of each obstacle over the time horizon
    :param obs_Qplus_mat_time_hor: ellipsoids of obstacles over the time horizon
    :return obs_Qplus_stack_all: stacked obstacle ellipsoids
    :return obs_mu_bar_stack_all: stacked obstacle mean positions
    """

    obs_Qplus_stack = [[] for obs in range(n_obstacles)]
    obs_mu_bar_stack = [[] for obs in range(n_obstacles)]
    for obs_index in range(n_obstacles):
        for t_step in range(time_horizon):
            if t_step == 0:
                obs_mu_bar_stack[obs_index] = np.asarray(obs_mu_bars_time_hor[obs_index][t_step])
                obs_Qplus_stack[obs_index] = np.linalg.inv(obs_Qplus_mat_time_hor[obs_index][t_step])
            else:
                obs_mu_bar_stack[obs_index] = np.hstack((obs_mu_bar_stack[obs_index],
                                                         np.asarray(obs_mu_bars_time_hor[obs_index][t_step])))
                obs_Qplus_stack[obs_index] = block_diag_nonsparse(obs_Qplus_stack[obs_index],
                                                        np.linalg.inv(obs_Qplus_mat_time_hor[obs_index][t_step]))
    obs_Qplus_stack_all = block_diag_nonsparse(*obs_Qplus_stack)
    obs_mu_bar_stack_all = np.hstack(tuple(obs_mu_bar_stack))

    return obs_Qplus_stack_all, obs_mu_bar_stack_all

def set_up_independent_ecos_params(n_obstacles, time_horizon, robot_state_dimension,
                                  robot_input_dimension, H, target_state,
                                  state_max, input_max):
    """
    :param n_obstacles:
    :param time_horizon:
    :param robot_state_dimension:
    :param robot_input_dimension:
    :param Z:
    :param H:
    :param current_state:
    :param target_state:
    :param state_max:
    :param input_max:
    :return:
    """

    # For ease of notation
    r = robot_state_dimension
    i = robot_input_dimension
    h = time_horizon
    n = n_obstacles

    total_num_vars = r * h + i * h + n * h + 1

    # Left-hand of quadratic cone constraints
    G_quadratic = vstack([hstack([csr((1, total_num_vars - 1)), csr(-1.)]),
                          hstack([-2 * identity(r * h), csr((r * h, i * h + n * h + 1))]),
                          hstack([csr((1, total_num_vars - 1)), csr(1.)])])

    # Right-hand side of quadratic cone constraints
    h_quadratic = np.reshape(np.concatenate(([1], np.zeros(r * h), [1])),
                             (G_quadratic.shape[0], 1))

    # Left-hand side of linear cone constraints
    G_linear_1 = hstack((csr((n * h, r * h + i * h)), -identity(n * h), csr((n * h, 1))))
    G_linear_2 = hstack((-identity(r * h), csr((r * h, i * h + n * h + 1))))
    G_linear_3 = hstack((identity(r * h), csr((r * h, i * h + n * h + 1))))
    G_linear_4 = hstack((csr((i * h, r * h)), -identity(i * h), csr((i * h, n * h + 1))))
    G_linear_5 = hstack((csr((i * h, r * h)), identity(i * h), csr((i * h, n * h + 1))))
    G_linear_partial_stack = vstack([G_linear_1, G_linear_2, G_linear_3, G_linear_4, G_linear_5])

    # Right-hand side of linear cone constraints
    h_linear_1 = np.zeros((n * h, 1))
    h_linear_2 = np.tile(state_max, (r * h, 1))
    h_linear_3 = np.tile(state_max, (r * h, 1))
    h_linear_4 = np.tile(input_max, (i * h, 1))
    h_linear_5 = np.tile(input_max, (i * h, 1))
    h_linear_partial_stack = np.concatenate((h_linear_1,
        h_linear_2, h_linear_3, h_linear_4, h_linear_5), axis=0)

    # Left-hand side of equality constraints
    # (those blocks that can be expressed without loop-dependent data)
    A = csc(hstack((identity(r * h), csr(-H), csr((r * h, n * h + 1)))))

    # Tile the target state
    target_tile = np.tile(np.reshape(target_state, (robot_state_dimension, 1)), (h, 1))

    return A, G_linear_partial_stack, G_quadratic,\
           h_linear_partial_stack, h_quadratic, target_tile