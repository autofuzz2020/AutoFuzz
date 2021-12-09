import numpy as np

from pymoo.factory import get_decomposition, get_algorithm, ZDT
from pymoo.operators.repair.out_of_bounds_repair import repair_out_of_bounds
from pymoo.optimize import minimize
from pymoo.problems.util import decompose
from pymoo.visualization.scatter import Scatter


class ModifiedZDT1(ZDT):

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        out_of_bounds = np.any(repair_out_of_bounds(self, x.copy()) != x)

        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum((x[:, 1:]) ** 2, axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        if out_of_bounds:
            f1 = np.full(x.shape[0], np.inf)
            f2 = np.full(x.shape[0], np.inf)

        out["F"] = np.column_stack([f1, f2])


n_var = 2
original_problem = ModifiedZDT1(n_var=n_var)
weights = np.array([0.5, 0.5])

decomp = get_decomposition("asf",
                           ideal_point=np.array([0.0, 0.0]),
                           nadir_point=np.array([1.0, 1.0]))

pf = original_problem.pareto_front()

problem = decompose(original_problem,
                    decomp,
                    weights
                    )


for i in range(100):

    if i != 23:
        continue

    res = minimize(problem,
                   get_algorithm("nelder-mead", n_max_restarts=10, adaptive=True),
                   #scipy_minimize("Nelder-Mead"),
                   #termination=("n_eval", 30000),
                   seed=i,
                   verbose=False)

    #print(res.X)

    F = ModifiedZDT1(n_var=n_var).evaluate(res.X, return_values_of="F")
    print(i, F)

opt = decomp.do(pf, weights).argmin()



print(pf[opt])
print(decomp.do(pf, weights).min())

plot = Scatter()
plot.add(pf)
plot.add(F)
plot.add(np.row_stack([np.zeros(2), weights]), plot_type="line")
plot.add(pf[opt])
plot.show()
