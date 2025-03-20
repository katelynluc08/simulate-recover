import random
import numpy as np
import scipy.stats as stats

#CONSTANTS

THRESHOLD = 0.4


def random_parameters(): # Generates random predicted parameters for you to simulate data

    a = random.uniform(0.5,2) # Boundary separation (how much evidence is needed)
    v = random.uniform(0.5,2) # Drift rate (how fast the evidence is gathered)
    t = random.uniform(0.1,0.5) # Non decision time (the time spent NOT making decision)

    return a, v, t

def forward_eq(a,v,t): # Forward equations to generate the expected results(mean RT, var, & accuracy) from parameters w/o no noise
   # (baseline results to compare to)
    y = np.exp(-a * v)

    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
   #  V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v - y**2) / ((y + 1) ** 2))
   #  V_pred = max((a / (2 * v**3)) * ((1 - 2 * a * v - y**2) / ((y + 1) ** 2)), 0) # Makes sure V_pred is positive not neg
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1) ** 2) 

    return {"R_pred": R_pred, "M_pred": M_pred, "V_pred": V_pred}


def simulate_R_obs(R_pred, N): # this passes the previous function variables in order to simulate data.
    # first you have to calculate the observed parameters (parameters with noise so real data)

    R_obs = stats.binom.rvs(n=N, p=R_pred) / N
    return R_obs


def simulate_M_obs(M_pred, V_pred, N):
    print(f"Debug: V_pred = {V_pred}, N = {N}, V_pred/N = {V_pred / N}")

    # Ensure V_pred is non-negative
    if V_pred < 0:
        print("Warning: V_pred is negative. Setting to 0.")
        V_pred = 0

    # Check if V_pred is NaN
    if np.isnan(V_pred):
        print("Error: V_pred is NaN!")
    if V_pred < 1e-6:
        print(f"Warning: V_pred is too small ({V_pred}) at iteration {i}, sample size {n}")

    std_dev = np.sqrt(V_pred / N) if V_pred / N > 0 else 0  # Ensure non-negative standard deviation
    M_obs = stats.norm.rvs(loc=M_pred, scale=std_dev)

    return M_obs


def simulate_V_obs(V_pred, N):
    shape = (N - 1) / 2
    scale = max(2 * V_pred / (N - 1), 1e-10)  # Prevent scale from being zero
    V_obs = stats.gamma.rvs(a=shape, scale=scale)

    if np.isnan(V_obs) or V_obs <= 0:
        print("Warning: V_obs is NaN or zero, setting to small positive value.")
        V_obs = 1e-10  # Avoid division by zero in inverse equations

    return V_obs


def inverse_eq(R_obs, V_obs, M_obs):
    R_obs = np.clip(R_obs, 0.001, 0.999)  # Avoid extreme values

    L = np.log(R_obs / (1 - R_obs))

    # Ensure V_obs is not zero before division
    if V_obs <= 0 or np.isnan(V_obs):
        print("Warning: V_obs is zero or NaN, setting to small value.")
        V_obs = 1e-10

    v_est_numerator = L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)
    v_est_denominator = V_obs
    v_est_value = v_est_numerator / v_est_denominator

    # Ensure v_est_value is non-negative before taking sqrt
    if v_est_value < 0:
        print("Warning: v_est_value is negative, setting to zero.")
        v_est_value = 0

    v_est_magnitude = np.sqrt(v_est_value)
    v_est = np.sign(R_obs - 0.5) * v_est_magnitude

    a_est = L / v_est if v_est != 0 else 1e-10  # Prevent division by zero
    bracket_term = (1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est))

    t_est = M_obs - (a_est / (2 * v_est)) * bracket_term if v_est != 0 else M_obs
    return v_est, a_est, t_est


def consistency_func(sample_size_arr : list, num_iter : int) -> tuple:

    bias_v_iter = []
    bias_a_iter = []
    bias_t_iter = []

    total_iterations = len(sample_size_arr) * num_iter # 3000 iterations

    total_avg_error = 0 # Accumulates the avg errors across all N and iterations
    total_squared_error = 0 # Accumulates squared errors across all N and iterations
    squared_errors_per_sample_size = [] # Stores avg squared error for each N

    with open('Results.txt', 'w') as f:

        for n in sample_size_arr: # For each sample size

            sample_size_errors = 0 # Accumulate errors for this current sample size
            sample_size_squared_errors = 0 # Accumulate squared errors for current sample size

            for i in range(num_iter): # For each iteration starting from 0-999

                a, v, t = random_parameters() # Generate random parameters

                res_pred = forward_eq(a,v,t) # Expected results(baseline results to compare to)
                R_pred, M_pred, V_pred = res_pred["R_pred"], res_pred["M_pred"], res_pred["V_pred"] # Extract results from res_pred for next steps

                res_R_obs = simulate_R_obs(R_pred, n) # Real data (simulate expected results with noise)
                res_M_obs = simulate_M_obs(M_pred, V_pred, n)
                res_V_obs = simulate_V_obs(V_pred, n)


                v_est, a_est, t_est = inverse_eq(res_R_obs, res_V_obs, res_M_obs) # Given the results you estimate the og parameters based on it (using obs and estimating the parameters that were given to pred function)

                avg_error = (abs(a - a_est) + abs(v - v_est) + abs(t - t_est))/3 # Average error of the parameters for this iteration (bias)

                avg_squared_error = avg_error**2 # Average error of the parameters for this iteration (bias^2)

                bias_v_iter.append(v - v_est)
                bias_a_iter.append(a - a_est)
                bias_t_iter.append(t - t_est)

                sample_size_errors += avg_error # Stores 1000 iterations for N
                sample_size_squared_errors += avg_squared_error # Stores 1000 iterations for N

                total_avg_error += avg_error # Stores 3000 avg errors
                total_squared_error += avg_squared_error # Stores 3000 avg squared errors
                print(f"Iteration {i}, Sample Size {n}: v_est={v_est}, a_est={a_est}, t_est={t_est}")

            avg_squared_error_for_size = sample_size_squared_errors / num_iter # Calculates average of the 1000 avg squared errors in each N
            squared_errors_per_sample_size.append(avg_squared_error_for_size) # Adding this average per N to list

            avg_bias_v = sum(bias_v_iter) / len(bias_v_iter)
            avg_bias_a = sum(bias_a_iter) / len(bias_a_iter)
            avg_bias_t = sum(bias_t_iter) / len(bias_t_iter)

            f.write(f"Sample size {n}: avg squared error = {avg_squared_error_for_size}") # Prints avg squared error for current sample size
            f.write(f"Sample size {n}: Average Bias for v: {avg_bias_v:.5f}, a: {avg_bias_a:.5f}, t: {avg_bias_t:.5f}") # Prints avg error for each parameter for current sample size 

            bias_v_iter = []
            bias_a_iter = []
            bias_t_iter = []

        overall_avg_error = total_avg_error / total_iterations # Overall avg error to check if it is equal to or less than 0.4

    return overall_avg_error <= THRESHOLD, overall_avg_error, squared_errors_per_sample_size # you need to calculate b^2 here in order to output the mean squared error in the print statement 


if __name__ == "__main__":
    sample_size = [10,40,4000]
    iterations = 1000

    is_consistent, overall_avg_error, squared_errors_per_sample_size = consistency_func(sample_size, iterations)

    print(f"Result is {'consistent' if is_consistent else 'not consistent'} with an average error of {overall_avg_error}")

    print("\nAnalysis of squared error by sample size:")
    for i, n in enumerate(sample_size):
        print(f"Sample size {n}: {squared_errors_per_sample_size[i]}")
