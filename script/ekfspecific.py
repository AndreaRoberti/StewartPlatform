from EKFbase import *
import numpy as np


class EKFSpecific(EKFBase):
    def __init__(self):
        super().__init__(dim_x = 3, dim_z = 1)
        self.A_ = 500  # Ampiezza
        self.phi_ = 0  # Fase
        self.rk.F = np.zeros((3, 3))
        self.sigma1 = 1
        self.sigma2 = 1
        self.sigma3 = 1
        self.rk.Q = np.array([[self.sigma1, 0, 0], [0, self.sigma2, 0], [0, 0, self.sigma3]])
        self.noise_variance = 0.0001
        self.rk.R = np.array([[self.noise_variance]])

    def Hx(self, m):
        return m[0]

    def Jacobian(self, m):
        return np.array([[1, 0, 0]])

    def update(self):
        t = rospy.get_time() - self.start_t_
        self.stewart_platform.respiration_ik(t)
        self.sim.step()

        omega = self.stewart_platform.alpha_ * t + self.stewart_platform.beta_
        x1 = self.A_ * np.sin(omega * t + self.phi_)
        x2 = self.A_ * omega * np.cos(omega * t + self.phi_)
        x3 = omega
        self.rk.x = np.array([x1, x2, x3])

        omegadot = self.stewart_platform.alpha_
        omegadotdot = 0
        self.rk.F = np.array([[0, 1, 0],
                              [-(t * omegadot + x3) ** 2, (t * omegadotdot + omegadot) / (t * omegadot + x3),
                               -x1 * (2 * x3 + 2 * t * omegadot) - (x2 * (t * omegadotdot + omegadot) / ((t * omegadot + omega) ** 2))],
                              [0, 0, 0]])

        noise = np.random.normal(loc=0, scale=np.sqrt(self.noise_variance))

        for _ in range(5):
            z = self.centroid_pose_.pose.position.z + noise
            self.rk.predict()
            self.rk.update(z, self.Jacobian, self.Hx)

        self.pos_.append(self.centroid_pose_.pose.position.z)
        self.xs_.append(self.rk.x[0])
