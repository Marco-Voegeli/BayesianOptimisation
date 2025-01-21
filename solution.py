import random
import os
import typing
import logging
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.stats import norm
from statistics import NormalDist
EXTENDED_EVALUATION = True
# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.


""" Solution """


#
class BO_algo(object):
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        self.previous_points = []
        # IMPORTANT: DO NOT REMOVE THOSE ATTRIBUTES AND USE sklearn.gaussian_process.GaussianProcessRegressor instances!
        # Otherwise, the extended evaluation will break.
        k_c = kernels.RBF(2) * kernels.ConstantKernel(3.5)
        k_o = kernels.RBF(1.5) * kernels.ConstantKernel(1.5)
        self.constraint_model = GaussianProcessRegressor(k_c,
                                                         alpha=0.005)  # GP model for the constraint function
        self.objective_model = GaussianProcessRegressor(k_o,
                                                        alpha=0.01)  # GP model for the objective function

    def next_recommendation(self) -> np.ndarray:
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        if not self.previous_points:
            sample = np.array([[np.random.uniform(0, 6), np.random.uniform(0, 6)]])
            return sample
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self) -> np.ndarray:  
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that approximately maximizes the acquisition function.
        """

        def objective(x: np.array):
            return - self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain_x[0, 0] + (domain_x[0, 1] - domain_x[0, 0]) * \
                 np.random.rand(1)
            x1 = domain_x[1, 0] + (domain_x[1, 1] - domain_x[1, 0]) * \
                 np.random.rand(1)
            result = fmin_l_bfgs_b(objective, x0=np.array([x0, x1]), bounds=domain_x,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain_x[0]))
            f_values.append(result[1])

        ind = np.argmin(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            point in the domain of f

        Returns
        ------
        af_value: float
            value of the acquisition function at x
        """

        mean_y = np.mean([x[2] for x in self.previous_points])
        min_y = np.min([x[2] for x in self.previous_points])
        target = np.max(mean_y)
        x_reshaped = x.reshape(1, -1)
        mu_x, sigma_x = self.objective_model.predict(x_reshaped, return_std=True)
        mu_c, sigma_c = self.constraint_model.predict(x_reshaped, return_std=True)
        c_dist = NormalDist(mu_c,sigma_c)

        z_x = (min_y - mu_x) / sigma_x
        cdf_z = norm.cdf(z_x)
        pdf_z = norm.pdf(z_x)
        p_c = c_dist.cdf(0) #p(c(x) <= 0) that's why cdf

        ei = sigma_x * (z_x * cdf_z + pdf_z)

        return ei * p_c # acquisition function

    def add_data_point(self, x: np.ndarray, z: float, c: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            point in the domain of f
        z: np.ndarray
            value of the objective function at x
        c: np.ndarray
            value of the constraint function at x
        """

        assert x.shape == (1, 2)
        self.previous_points
        self.previous_points.append([float(x[:, 0]), float(x[:, 1]), float(z), float(c)])

        # We can now train the model with the new point (our new posterior)
        # x_train = [(x[:][0], x[:][1]) for x in self.previous_points]
        x_train = [(float(x[0]), float(x[1])) for x in self.previous_points]

        z_train = [x[:][2] for x in self.previous_points]
        c_train = [x[:][3] for x in self.previous_points]
        self.constraint_model.fit(x_train, c_train)
        self.objective_model.fit(x_train, z_train)

    def get_solution(self) -> np.ndarray:
        """
        Return x_opt that is believed to be the minimizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # Here something to do with the previous points getting the one with the smallest z that holds for the constraint
        # good_values = [x for x in self.previous_points if x[:][3]<=0]
        good_values = [x for x in self.previous_points if float(x[3]) <= 0]

        if not good_values:
            sample = np.array([[np.random.random_sample() * 6, np.random.random_sample() * 6]])
            return sample
        f_values = [x[:][2] for x in good_values]
        x_values = [(x[:][0], x[:][1]) for x in good_values]
        best_ind = np.argmin(f_values)
        return np.atleast_2d(x_values[best_ind])


""" 
    Toy problem to check  you code works as expected
    IMPORTANT: This example is never used and has nothing in common with the task you
    are evaluated on, it's here only for development and illustration purposes.
"""
domain_x = np.array([[0, 6], [0, 6]])
EVALUATION_GRID_POINTS = 250
CONSTRAINT_OFFSET = - .2  # This is an offset you can change to make the constraint more or less difficult to fulfill
LAMBDA = 0.0  # You shouldn't change this value


def check_in_domain(x) -> bool:
    """Validate input"""
    x = np.atleast_2d(x)
    v_dim_0 = np.all(x[:, 0] >= domain_x[0, 0]) and np.all(x[:, 0] <= domain_x[0, 1])
    v_dim_1 = np.all(x[:, 1] >= domain_x[1, 0]) and np.all(x[:, 0] <= domain_x[1, 1])

    return v_dim_0 and v_dim_1


def f(x) -> np.ndarray:
    """Dummy objective"""
    l1 = lambda x0, x1: np.sin(x0) + x1 - 1

    return l1(x[:, 0], x[:, 1])


def c(x) -> np.ndarray:
    """Dummy constraint"""
    c1 = lambda x, y: np.cos(x) * np.cos(y) - 0.1

    return c1(x[:, 0], x[:, 1]) - CONSTRAINT_OFFSET


def get_valid_opt(f, c, domain) -> typing.Tuple[float, float, np.ndarray, np.ndarray]:
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(domain[0, 0], domain[0, 1], nx)
    y = np.linspace(domain[1, 0], domain[1, 1], ny)
    xv, yv = np.meshgrid(x, y)
    samples = np.array([xv.reshape(-1), yv.reshape(-1)]).T

    true_values = f(samples)
    true_cond = c(samples)
    valid_data_idx = np.where(true_cond < LAMBDA)[0]
    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = samples[valid_data_idx][np.argmin(true_values[np.where(true_cond < LAMBDA)])]
    f_max = np.max(np.abs(true_values))
    x_max = np.argmax(np.abs(true_values))
    return f_opt, f_max, x_opt, x_max


def perform_extended_evaluation(agent, output_dir='./'):
     # Set up the evaluation grid
    fig = plt.figure(figsize=(25, 5), dpi=100)
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(0.0, 6.0, nx)
    y = np.linspace(0.0, 6.0, ny)
    xv, yv = np.meshgrid(x, y)

    # Get solution from the agent
    solution = agent.get_solution()
    x_b, y_b = solution[0][0], solution[0][1]

    # Prepare samples for predictions
    samples = np.array([xv.ravel(), yv.ravel()]).T

    # Make predictions for objective and constraint models
    predictions, stds = agent.objective_model.predict(samples, return_std=True)
    predictions = predictions.reshape(nx, ny)

    conds = agent.constraint_model.predict(samples)
    conds = conds.reshape(nx, ny)

    # Evaluate true objective and constraint values
    true_values = f(samples)
    true_cond = c(samples)
    conditions_valid = (true_cond < LAMBDA).astype(float)

    # Prepare constraint visualization with NaNs
    conditions_with_nans = 1 - conditions_valid
    conditions_with_nans[conditions_with_nans == 0] = np.nan
    conditions_with_nans = conditions_with_nans.reshape(nx, ny)

    # Find true optimum under constraints
    valid_data_idx = np.where(true_cond < LAMBDA)[0]
    f_opt = np.min(true_values[valid_data_idx])
    x_opt = samples[valid_data_idx][np.argmin(true_values[valid_data_idx])]

    # Retrieve sampled points
    sampled_points = np.array(agent.previous_points)

    # Plot constraint GP posterior
    ax_condition = fig.add_subplot(1, 4, 4)
    im_cond = ax_condition.pcolormesh(xv, yv, conds, shading='auto')
    fig.colorbar(im_cond, ax=ax_condition)
    
    ax_condition.scatter(sampled_points[:, 0], sampled_points[:, 1], s=40, marker='x', label='Sampled Points', antialiased=True)
    ax_condition.pcolormesh(xv, yv, conditions_with_nans, shading='auto', cmap='Reds', alpha=0.7, vmin=0, vmax=1.0)
    ax_condition.set_title('Constraint GP Posterior + True Constraint (Red = Infeasible)')
    ax_condition.legend(fontsize='x-small')

    # Plot 3D objective GP posterior
    ax_gp_f = fig.add_subplot(1, 4, 2, projection='3d')
    ax_gp_f.plot_surface(xv, yv, predictions, rcount=100, ccount=100, linewidth=0, antialiased=False)
    ax_gp_f.set_title('Objective GP Posterior (3D)')

    # Plot 3D constraint GP posterior
    ax_gp_c = fig.add_subplot(1, 4, 3, projection='3d')
    ax_gp_c.plot_surface(xv, yv, conds, rcount=100, ccount=100, linewidth=0, antialiased=False)
    ax_gp_c.set_title('Constraint GP Posterior (3D)')

    # Plot objective GP posterior with constraints
    ax_predictions = fig.add_subplot(1, 4, 1)
    im_predictions = ax_predictions.pcolormesh(xv, yv, predictions, shading='auto')
    fig.colorbar(im_predictions, ax=ax_predictions)

    ax_predictions.pcolormesh(xv, yv, conditions_with_nans, shading='auto', cmap='Reds', alpha=0.7, vmin=0, vmax=1.0)
    ax_predictions.scatter(x_b, y_b, s=40, marker='x', label='Predicted Value by BO')
    ax_predictions.scatter(x_opt[0], x_opt[1], s=20, marker='o', label='True Optimum Under Constraint')
    ax_predictions.set_title('Objective GP Posterior + True Constraint (Red = Infeasible)')
    ax_predictions.legend(fontsize='x-small')

    # Save and display the figure
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    plt.show()


def train_on_toy(agent, iteration):
    logging.info('Running model on toy example.')
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    for j in range(iteration):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain_x.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain_x.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(size=(x.shape[0],), scale=0.01)
        cost_val = c(x) + np.random.normal(size=(x.shape[0],), scale=0.005)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain_x.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain_x.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    f_opt, f_max, x_opt, x_max = get_valid_opt(f, c, domain_x)
    if c(solution) > 0.0:
        regret = 1
    else:
        regret = (f(solution) - f_opt) / f_max

    print(f'Optimal value: {f_opt}\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')
    return agent


def main():
    
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = BO_algo()

    agent = train_on_toy(agent, 20)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(agent)


if __name__ == "__main__":
    main()
