import numpy as np
from kfsims.tracker2d import run_sim
from kfsims.kfmodels import KalmanFilterBase

# Simulation Options
sim_options = {'time_step': 0.01,
               'end_time': 120,
               'measurement_rate': 1,
               'measurement_noise_std': 10,
               'motion_type': 'straight',
               'start_at_origin': True,
               'start_at_random_speed': False,
               'start_at_random_heading': False,
               'draw_plots': True,
               'draw_animation': True} 

# Kalman Filter Model
class KalmanFilterModel(KalmanFilterBase):
    
    def initialise(self, time_step):

        deltaT = time_step

        # Define a np.array 4x1 with the initial state (px,py,vx,vy)
        #Initially object starts at origin and moves at 10m/s with 45 deg. inclination to x-axis
        #px = 0
        #py = 0
        #vx = 10cos(45) = 7.07
        #vy= 10sin(45) = 7.07
        px = 0
        py = 0
        vx = 0
        vy= 0

        self.state = np.array([px,py,vx,vy]).T

        # Define a np.array 4x4 for the initial covariance
        #[0 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        # 0 0 0 0 ]
        pos_x_covariance = 0
        pos_y_covariance = 0
        vel_x_covariance = (7/3)**2
        vel_y_covariance = (7/3)**2

        self.covariance = np.diag([pos_x_covariance,
                                    pos_y_covariance,
                                    vel_x_covariance,
                                    vel_y_covariance])

        # Setup the Model F Matrix
        #[ 1  0 deltaT 0
        #  0  1   0  deltaT
        #  0  0   1    0
        #  0  0   0    1 ]

        self.F = np.array([[1,0,deltaT,0],
                           [0,1,0,deltaT],
                           [0,0,1,0],
                           [0,0,0,1]])

        # Set the Q Matrix
        #  sigma_a^2[ 1/2deltaT    0      0        0
        #             0   1/2deltaT   0        0
        #             0        0   1/2deltaT   0
        #             0        0      0    1/2deltaT ]
        acc_covariance = 0
        self.Q = acc_covariance*0.5 * deltaT * np.identity(4)
        
        return
    
    def prediction_step(self):
        # Make Sure Filter is Initialised
        if self.state is not None:
            x = self.state
            P = self.covariance
            Q = self.Q
            F = self.F

            # Calculate Kalman Filter Prediction
            
            # State Prediction: x_predict = F * x
            x_predict = np.matmul(F,x)

            # Covariance Prediction: P_predict = F * P * F' + Q
            P_predict = np.matmul(F,np.matmul(P,F.T)) + Q

            # Save Predicted State
            self.state = x_predict
            self.covariance = P_predict

        return

    def update_step(self, measurement):
        return 



# Run the Simulation
run_sim(KalmanFilterModel, sim_options, {})