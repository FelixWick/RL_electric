import numpy as np
from scipy.optimize import minimize


class MPCController:
    def __init__(self, env, horizon=10):
        self.env = env
        self.horizon = horizon

    def model(self, I, U):
        for _ in range(10):
            dI_dt = (U - self.env.R_base * I) / self.env.L_base
            I = I + dI_dt * self.env.dt
        return I

    def objective(self, U_seq, I0):
        I = I0
        cost = 0
        for U in U_seq:
            I = self.model(I, U)
            cost += (I - self.env.I_ref) ** 2
        return cost

    def optimize_voltage(self, current_state):
        I0 = current_state[0]
        U0 = current_state[1]

        U_init = np.full(self.horizon, U0)

        bounds = [(self.env.U_min, self.env.U_max) for _ in range(self.horizon)]

        result = minimize(self.objective, U_init, args=(I0,), bounds=bounds, method='SLSQP')

        return result.x[0] if result.success else U0
