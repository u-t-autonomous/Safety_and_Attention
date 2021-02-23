import ecos
import numpy as np
import time as time
import cvxpy as cp
from scipy.linalg import block_diag as block_diag_nonsparse
from scipy.sparse import bmat
from scipy.sparse import block_diag
from scipy.sparse import hstack
from scipy.sparse import identity
from scipy.sparse import spdiags
from scipy.sparse import vstack
from scipy.sparse import csr_matrix as csr
from scipy.sparse import csc_matrix as csc

# See if this expedites the process...
from ecosqp_file import ecosqp

# UPDATE CODE TO ALLOW FOR THE ROBOTIC AGENT TO BE SUBJECT TO
# TURNING RATE CONSTRAINTS!
#
# ECOS requires the form:
# minimize c'x
#     s.t. Ax = b
#          Gx <=_K h     (i.e., Gx-h belongs to the cone K)
#
def dc_motion_planning_call_ecos_unicycle(n_obstacles,time_horizon, robot_state_dimension,
                                robot_input_dimension,obs_free_traj,
                                obs_mu_bar_stack, obs_Qplus_stack, mult_matrix_stack,
                                target_tile, obs_mu_bars_time_hor, obs_Qplus_mat_time_hor,
                                most_rel_obs_ind, heading_angle_array,
                                A_eq_obs_free, b_eq_obs_free, A_obs_free,
                                b_obs_free, H_obs_free, f_obs_free,tau_max,mu,delta,tau):

    # For ease of notation
    r = robot_state_dimension
    i = robot_input_dimension
    h = time_horizon
    n = n_obstacles
    total_num_vars = r * h + i * h + n * h + 1

    # Reshape some of the variables
    obs_mu_bar_stack = np.reshape(np.asarray(obs_mu_bar_stack),(r*n*h,1))
    obs_Qplus_stack = np.asarray(obs_Qplus_stack)

    # Store information about each loop iteration
    iteration_count = 0
    solver_times = []
    num_active_duals_per_iter = []

    # Just to get through the initial run...
    prev_obs_func_val = -1.

    tau_max_counter = 0

    # Repeat until convergence
    while True and tau_max_counter <= 2:

        tic = time.clock()

        # First, adjust the existing equality and inequality constraints to account for the
        # addition of slack variables in the CCP formulation
        A = np.hstack([A_obs_free, np.zeros((A_obs_free.shape[0], n*h))])
        A_eq = np.hstack([A_eq_obs_free, np.zeros((A_eq_obs_free.shape[0], n*h))])
        b_eq = b_eq_obs_free[:]

        if iteration_count == 0:
            prev_traj = np.reshape(np.asarray(obs_free_traj),(r*h,1))
        else:
            prev_traj = np.reshape(traj,(r*h,1))
        prev_traj_stack = np.tile(prev_traj,(n_obstacles,1))

        # Vectors for constructing additional inequality constraints
        tic = time.clock()
        cur_stack = np.matmul(mult_matrix_stack,np.matmul(spdiags((prev_traj_stack - obs_mu_bar_stack).T,0,r*n*h,
                                                r*n*h).toarray(),obs_Qplus_stack))
        prev_stack = np.tile(([1]),(n*h,1)) + np.matmul(cur_stack, (prev_traj_stack + obs_mu_bar_stack))

        # Need to get cur_stack (dimension currently n*h x n*h*r. Need to make n*h x h*r.
        cur_stack_proper_dims = sum(-2*cur_stack[:,obs*r*h:(obs+1)*r*h] for obs in range(n_obstacles))

        # Convexified inequality constraints
        dual_var_ind_1 = np.shape(A)[0]
        A_ccp = np.hstack((cur_stack_proper_dims, np.zeros((n*h,i*h)), -np.eye(n*h)))
        b_ccp = -prev_stack.reshape((n*h,1))
        toc = time.clock()
        #print(toc - tic)
        A = np.vstack([A, A_ccp])
        b = np.vstack((b_obs_free,b_ccp))
        dual_var_ind_2 = np.shape(A)[0]

        # Slack variables must be non-negative
        A_no_slack = A[:]
        A = np.vstack([A, np.hstack((np.zeros((n*h,r*h+i*h)),-1*np.eye(n*h)))])
        b_no_slack = b[:]
        b = np.vstack([b, np.zeros((n*h,1))])

        # Construct "H" matrix and "f" vector for the objective function. Note that the parameters differ
        # in the case that we have a relevant obstacle identified.
        H = np.zeros((r*h + i*h + n*h, r*h + i*h + n*h))
        f = np.zeros((r*h + i*h + n*h, 1))
        # if most_rel_obs_ind is not None:
        #     x_g_1 = target_tile[0][0]
        #     x_g_2 = target_tile[1][0]
        #     c_th = np.cos(heading_angle_array[0])
        #     s_th = np.sin(heading_angle_array[0])
        #     most_rel_obs_mu_t = obs_mu_bars_time_hor[most_rel_obs_ind][0][:]
        #     obs_1 = most_rel_obs_mu_t[0]
        #     obs_2 = most_rel_obs_mu_t[1]
        #     # H[0:r,0:r] =\
        #     #     np.array([[a_w*c_th**2 + 1,a_w*c_th*s_th],[a_w*c_th*s_th,a_w*s_th**2 + 1]])
        #     # f[0:r] = np.array([[-2*a_w*obs_1*c_th**2 - 2*a_w*c_th*obs_2*s_th - 2*x_g_1],
        #     #                                      [-2*a_w*obs_2*s_th**2 - 2*a_w*c_th*obs_1*s_th - 2*x_g_2]])
        #     H[0:r,0:r] = np.eye(r)
        #     f[0:r,0:r] = np.array([[c_th - 2*x_g_1],[s_th - 2*x_g_2]])
        #     H[r:h*r,r:h*r] = np.eye((h-1)*r)
        #     f[r:h*r] = np.tile(np.array([[- 2 * x_g_1],[- 2 * x_g_2]]),(h-1,1))
        # else:
        #     H[0:r*h+i*h,0:r*h+i*h] = H_obs_free[:,:]
        #     f[0:r*h+i*h] = f_obs_free[:]
        H[0:r * h + i * h, 0:r * h + i * h] = H_obs_free[:,:]
        f[0:r * h + i * h] = f_obs_free[:]

        # The variables relating to the control input, v[t], do not appear in the objective function.
        # However, in the convex-concave procedure, still need to include \tau*\sum(slack variables)
        f[r*h+i*h:] = tau*np.ones((n*h,1))

        # Send the problem to ecosqp
        solution = ecosqp(H, f, A, b, A_eq, b_eq)
        sol_x = solution["x"]
        sol_z = solution["z"]

        # Get the desired objective function (norm, not norm squared)

        # Solution parameters that we care about: the updated trajectory, the input, the dual variables,
        # and the cost function. We iterate until the cost function has converged.
        traj = sol_x[0 : r*h]
        obs_func_val = solution["fval"]
        dual_vars_dc = sol_z[dual_var_ind_1:dual_var_ind_2]

        # Append the number of active dual variables
        num_active_duals_per_iter.append(np.sum((dual_vars_dc >= 1e-3).astype(int)))

        # Convergence criterion
        if abs(prev_obs_func_val - obs_func_val) <= delta:
            break
        else:
            iteration_count += 1

        # Update the parameter for tau
        tau = min([tau_max, mu * tau])

        if tau == tau_max:
            tau_max_counter += 1

        # Save the existing optimal value
        prev_obs_func_val = obs_func_val

        # Time for this iteration
        toc = time.clock()
        solver_times.append(toc - tic)

    # Now that we have found a trajectory, we can use the current trajectory (that we know is
    # feasible) as the input for a set of projection-based constraints. Will then use these constraints
    # for our final motion-planning problem.
    traj_state_x_time = np.reshape(traj, (time_horizon, robot_state_dimension)).T
    counter_1 = 0
    counter_2 = 0

    # Determine separating hyperplanes
    for obs in range(n_obstacles):
        if counter_2 == 0:
            for t_step in range(time_horizon):
                if counter_1 == 0:
                    p_vec,q_vec = solve_proj_problem_and_get_hyperplane(obs_Qplus_mat_time_hor[obs][t_step],
                        obs_mu_bars_time_hor[obs][t_step],traj_state_x_time[:,t_step])
                    p_mat = np.transpose(p_vec)
                    q_vec = q_vec
                    counter_1 += 1
                else:
                    p_vec, q_vec_cur = solve_proj_problem_and_get_hyperplane(obs_Qplus_mat_time_hor[obs][t_step],
                        obs_mu_bars_time_hor[obs][t_step], traj_state_x_time[:, t_step])
                    p_mat = block_diag_nonsparse(p_mat,np.transpose(p_vec))
                    q_vec = np.vstack((q_vec,q_vec_cur))
            counter_1 = 0
            counter_2 = 1
        else:
            for t_step in range(time_horizon):
                if counter_1 == 0:
                    p_vec, q_vec_new = solve_proj_problem_and_get_hyperplane(obs_Qplus_mat_time_hor[obs][t_step],
                        obs_mu_bars_time_hor[obs][t_step], traj_state_x_time[:, t_step])
                    p_mat_cur = np.transpose(p_vec)
                    q_vec_cur = q_vec_new
                    counter_1 += 1
                else:
                    p_vec, q_vec_new = solve_proj_problem_and_get_hyperplane(obs_Qplus_mat_time_hor[obs][t_step],
                        obs_mu_bars_time_hor[obs][t_step], traj_state_x_time[:, t_step])
                    p_mat_cur = block_diag_nonsparse(p_mat_cur,np.transpose(p_vec))
                    q_vec_cur = np.vstack((q_vec_cur,q_vec_new))
            counter_1 = 0
            p_mat = np.vstack((p_mat,p_mat_cur))
            q_vec = np.vstack((q_vec,q_vec_cur))


    # Now that we have our updated projection constraints, solve the new optimization problem
    traj_new, input, dual_vars, obs_func_val = solve_proj_opt_prob_unicycle(A_no_slack,b_no_slack,A_eq,b_eq,H,f,
                                                            robot_state_dimension,robot_input_dimension,
                                                            time_horizon,n_obstacles,p_mat,q_vec)

    obs_dual_vals = dual_vars

    return obs_dual_vals, traj_new, input, obs_func_val

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

def get_polytope_face_for_ellipsoid_for_an_exterior_point(
        obs_center, obs_matrix, query_state):

    """
    If the query_state is not in the obstacle ellipsoid, then the face is
    the supporting hyperplane at the projection of a point query_state onto an
    ellipse. This hyperplane is given by solving the following optimization
    problem:

    maximize_z (query_state - obs_center)'z - ||sqrt_obs_matrix z||_2
    s.t.       ||z||_2 <= 1

    where obs_center is the obstacle ellipsoid, sqrt_obs_matrix is the square
    root of the obs_matrix (computed via Cholesky decomposition). This
    optimization problem utilizes the fact that the support function of an
    ellipsoid is obs_center'z + ||sqrt_obs_matrix z||_2, and the discussion
    in Boyd's textbook Section 8.1.3. The face is given by
        z*'query_state <= z*'x
    or equivalently,
        -z*'x <= -z*'query_state
    provided the optimization problem above has a positive objective.

    :param obs_center: center of the current ellipse obstacle of interest
    :param obs_matrix: quadratic form of the current ellipse obstacle of
        interest. The ellipse is given by
            {x| (x - obs_center)' @ obs_matrix @ (x - obs_center) <= 1}
    :param query_state: the exterior point which is to be projected for the
        separating hyperplane construction
    :return:
        p_vector, q_scalar that characterizes the hyperplane p_vector'x <=
        q_scalar
    """

    # Get sqrt_obs_matrix for the optimization problem
    # Computes the square root of a matrix while preserving sparsity
    try:
        sqrt_obs_matrix = np.linalg.cholesky(obs_matrix)
    except np.linalg.LinAlgError:
        raise RuntimeError('obs_matrix must be positive definite!')

    # Set up and solve optimization problem
    tic1 = time.clock()
    z = cp.Variable(query_state.shape[0])
    objective = cp.Maximize(z.T @ (query_state - obs_center)/np.linalg.norm(query_state-obs_center)
                            - cp.norm(sqrt_obs_matrix @ z, 2))
    constraints = [cp.norm(z, 2) <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='ECOS')

    if prob.status in ['optimal', 'optimal_inaccurate']:
        if prob.value >= 0:
            # From optimal solution, obtain the separating hyperplane
            # FIXME: Will it ever be that z.value is not unit norm?
            p_vector = - z.value/np.linalg.norm(z.value)
        else:
            raise RuntimeError('Did not expect this quantity to be negative!')
    else:
        raise RuntimeError('CVXPY could not solve the problem. CVXPY status: '
            '{:s}'.format(prob.status))

    return p_vector, q_vector

def get_polytope_face_for_ellipsoid_for_an_exterior_point_ecos(
        obs_center, obs_matrix, query_state):

    # Get sqrt_obs_matrix for the optimization problem
    # Computes the square root of a matrix while preserving sparsity
    try:
        sqrt_obs_matrix = np.linalg.cholesky(obs_matrix)
    except np.linalg.LinAlgError:
        raise RuntimeError('obs_matrix must be positive definite!')

    o1 = sqrt_obs_matrix[0,0]
    o2 = sqrt_obs_matrix[0,1]
    o3 = sqrt_obs_matrix[1,0]
    o4 = sqrt_obs_matrix[1,1]

    # Set up and solve optimization problem

    dif = (obs_center - query_state)
    c = np.array([dif[0],dif[1],1.0,0])
    A = csc(np.array([[0,0,0,1.0]]))
    b = np.array([1.0])
    G = csc(np.array([[0,0,-1/np.sqrt(2),0],[-o1,-o2,0,0],[-o3,-o4,0,0],[0,0,1/np.sqrt(2),0],
                   [0,0,0,-1/np.sqrt(2)],[-1,0,0,0],[0,-1,0,0],[0,0,0,1/np.sqrt(2)]]))
    h = np.array([1/np.sqrt(2),0,0,1/np.sqrt(2),1/np.sqrt(2),0,0,1/np.sqrt(2)])

    dims = {"l": 0, "q": [G.shape[0]]}

    solution = ecos.solve(c, G, h, dims, A, b, verbose=False)
    sol_x = solution['x']
    z_vals = sol_x[0:2]

    p_vector = - z_vals / np.linalg.norm(z_vals)
    q_vector = np.matmul(p_vector, query_state)
    p_vector = p_vector.reshape(2,1)

    # Now, find the actual point that is on the face of the ellipse
    unit_dir = (query_state - obs_center) / np.linalg.norm(query_state - obs_center)
    cur_sup_func = (query_state - obs_center)*z_vals - np.linalg.norm(np.matmul(sqrt_obs_matrix,z))
    cur_proj_point = obs_center + cur_sup_func*unit_dir

    return p_vector, q_vector

def solve_proj_opt_prob_unicycle(A,b,A_eq,b_eq,H,f,rob_sta_dim,rob_in_dim,time_hor,
                        num_obstacles,p_mat,q_vec):

    # Minimize ||x-x_targ||
    # subject to
    #  x_1[t] <= x_max \forall t
    #  x_2[t] <= x_max \forall t
    # -x_1[t] <= x_min \forall t
    # -x_1[t] <= x_min \forall t
    #  u_1[t] <= u_max \forall t
    #  u_2[t] <= u_max \forall t
    # -u_1[t] <= u_min \forall t
    # -u_2[t] <= u_min \forall t
    # x[t+1] = Zx[0] + Hu[t] \forall t\in [0,T-1]

    r = rob_sta_dim
    i = rob_in_dim
    h = time_hor
    n = num_obstacles

    # Trim the slack variables from the A and A_eq matrices
    A = A[:,0:r*h+i*h]
    A_eq = A_eq[:,0:r*h+i*h]
    dual_index_1 = np.shape(A)[0]

    # Input must be in safe polytope. Add this inequality constraint to the set of
    # precomputed inequality constraints
    A = np.vstack((A,np.hstack((p_mat,np.zeros((n*h,i*h))))))
    b = np.vstack((b,q_vec[:]))
    dual_index_2 = np.shape(A)[0]

    # Objective function remains the same as in the CCP case (whether or not to
    # consider the most relevant obstacle). We just need to trim off the extra rows/
    # columns that were attributed to slack variables.
    H = H[0:r*h+i*h,0:r*h+i*h]
    f = f[0:r*h+i*h]

    # Call ecosqp
    solution = ecosqp(H,f,A,b,A_eq,b_eq)
    sol_x = solution['x']
    sol_z = solution['z']

    # Get the desired objective function (norm, not norm squared)

    # Solution parameters that we care about: the updated trajectory, the input, the dual variables,
    # and the cost function. We iterate until the cost function has converged.
    traj = sol_x[0: r*h]
    input = sol_x[r*h:r*h+i*h]
    obs_func_val = solution["fval"]
    obs_dual_vars = sol_z[dual_index_1:dual_index_2]

    return traj, input, obs_dual_vars, obs_func_val

def solve_proj_problem_and_get_hyperplane(Q,mu_bar,query_point):

    # Relevant parameters
    query_point = np.reshape(query_point,(2,1))
    mu_bar = np.reshape(mu_bar,(2,1))
    sqrt_Q = np.linalg.cholesky(Q)

    query_point_circle = np.matmul(np.linalg.inv(sqrt_Q), (query_point - mu_bar))
    proj_point_circle = query_point_circle / np.linalg.norm(query_point_circle)
    proj_point = np.matmul(sqrt_Q, proj_point_circle + mu_bar)

    # Q_cp.value = np.linalg.inv(Q)
    # mu_bar_cp.value = mu_bar
    # query_point_cp.value = np.reshape(query_point,(2,1))
    # proj.solve(verbose=False,solver="ECOS")
    #
    # proj_point = proj_point_cp.value

    # For numerical stability...
    if np.linalg.norm(proj_point - query_point) <= 5e-5:
        # TODO: Change this to linearization rather than this
        unit_dir = np.reshape((mu_bar - proj_point)/np.linalg.norm(proj_point - mu_bar),(2,1))
    else:
        unit_dir = np.reshape((proj_point - query_point) / np.linalg.norm(proj_point - query_point), (2, 1))

    hyperplane_p = unit_dir
    hyperplane_q = np.matmul(mu_bar.T, unit_dir) - np.sqrt(np.matmul(unit_dir.T, np.matmul(Q, unit_dir)))

    return hyperplane_p, hyperplane_q

def solve_obs_free_ecos_unicycle(Z_mat, H_mat, current_state, rob_sta_dim,
                                 rob_in_dim,time_hor, A, b, H, f):

    min_velocity = 0.0

    # Minimize ||x-x_goal||
    # subject to
    #  x_1[t] <= x_max \forall t
    #  x_2[t] <= x_max \forall t
    # -x_1[t] <= x_min \forall t
    # -x_2[t] <= x_min \forall t
    #  u_1[t] <= u_max \forall t
    #  u_2[t] <= u_max \forall t
    # -u_1[t] <= u_min \forall t
    # -u_2[t] <= u_min \forall t
    # x[t+1] = Zx[0] + Hu[t] \forall t\in [0,T-1]

    r = rob_sta_dim
    i = rob_in_dim
    h = time_hor

    # Adjust code to use ecosqp python implementation to solve problem.
    # Automatically constructs the quadratic cone (should be much less of
    # a headache to implement. Specifically, ecosqp assumes the form:
    #
    # min 1/2*x'*H*x + f'*x
    # s.t. A*x \leq b
    #      A_eq*x \leq b_eq
    #

    # Equality constraints relating trajectory to input for specific fixed turning rate
    A_eq = np.hstack((np.eye(r * h), -H_mat))
    b_eq = np.matmul(Z_mat, current_state)

    # Solve
    solution = ecosqp(H, f, A, b, A_eq, b_eq)
    #dims = {"l": G_linear.shape[0], "q": [G_quadratic.shape[0]]}
    sol_x = solution['x']
    traj = sol_x[0: r*h]
    input = sol_x[r*h:r*h+i*h]
    obs_func_val = solution["fval"]

    return traj, input, A_eq, b_eq, obs_func_val

def ecos_unicycle_shared_cons(r,i,h,state_max,max_velocity,target_tile):

    min_velocity = 0.01

    A_linear_1 = np.hstack((np.eye(r * h), np.zeros((r * h, i * h))))
    b_linear_1 = np.tile(state_max, (r * h, 1))

    # State must be greater than minimum -> assumes that state_min = -state_max (symmetry)
    A_linear_2 = np.hstack((-np.eye(r * h), np.zeros((r * h, i * h))))
    b_linear_2 = np.tile(state_max, (r * h, 1))

    # Input must be less than maximum -> recall, only have velocity constraint now
    A_linear_3 = np.hstack((np.zeros((i * h, r * h)), np.eye(i * h)))
    b_linear_3 = np.tile(max_velocity, (i * h, 1))

    # Input must be greater than minimum
    A_linear_4 = np.hstack((np.zeros((i * h, r * h)), -np.eye(i * h)))
    b_linear_4 = np.tile(-min_velocity, (i * h, 1))

    # Stack A and b
    A = np.concatenate((A_linear_1, A_linear_2, A_linear_3, A_linear_4), axis=0)
    b = np.vstack([b_linear_1, b_linear_2, b_linear_3, b_linear_4])

    # Define the "H" matrix and the "f" vector for the obtacle-free case.
    H = 2 * block_diag_nonsparse(np.eye(r * h), np.zeros((i * h, i * h)))
    f = np.vstack((-2 * target_tile, np.zeros((i * h, 1))))

    return A, b, H, f