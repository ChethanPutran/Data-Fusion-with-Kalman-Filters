import numpy as np
from numpy.core.fromnumeric import transpose
from tracker2d import run_sim
from kfmodels import KalmanFilterBase

# Simulation Options
sim_options = {'time_step': 0.1,
               'end_time': 120,
               'measurement_rate': 1,
               'measurement_noise_std': 10,
               'motion_type': 'straight',
               'start_at_origin': True,
               'start_at_random_speed': True,
               'start_at_random_heading': True,
               'draw_plots': True,
               'draw_animation': True} 

# Kalman Filter Model
class KalmanFilterModel(KalmanFilterBase):
    
    def initialise(self, time_step):
       
        initial_pos_std = 100
        initial_vel_std = 10

        # Define a np.array 4x1 with the initial state (px,py,vx,vy)
        #self.state = np.transpose(np.array([0,0,0,0]))

        # Define a np.array 4x4 for the initial covariance (P0)
        # self.covariance = np.diag(np.array([initial_pos_std**2,
        #                                     initial_pos_std**2,
        #                                     initial_vel_std**2,
        #                                     initial_vel_std**2]))

        # Setup the Model F Matrix
        self.F = np.array([ [1,0,time_step,0],
                            [0,1,0,time_step],
                            [0,0,1,0],
                            [0,0,0,1]])

        accel_std = .0

        # Set the Q Matrix
        self.Q = accel_std**2 * np.array([  [0.5*time_step**2,0,0,0],
                                            [0,0.5*time_step**2,0,0],
                                            [0,0,time_step,0],
                                            [0,0,0,time_step]])

         # Setup the Model H Matrix
        self.H = np.array([[1,0,0,0],[0,1,0,0]])

        # Set the R Matrix
        measurement_std = 10.0
        self.R = np.diag([measurement_std**2, measurement_std**2])                    
        
        return
    
    def prediction_step(self):
        # Make Sure Filter is Initialised
        if self.state is not None:
            x = self.state
            P = self.covariance

            # Calculate Kalman Filter Prediction
            
            # State Prediction: x_predict = F * x
            x_predict = np.matmul(self.F,x)

            # Covariance Prediction: P_predict = F * P * F' + Q 
            P_predict = np.matmul(np.matmul(self.F,P),self.F.T) + self.Q

            # Save Predicted State
            self.state = x_predict
            self.covariance = P_predict

        return

    def update_step(self, measurement):
        # Make Sure Filter is Initialised
        if self.state is not None and self.covariance is not None:
            x = self.state
            P = self.covariance
            H = self.H
            R = self.R

            # Calculate Kalman Filter Update
            z = np.array([measurement[0],measurement[1]])

            # Predicted Measurement: z_hat = H * x
            z_hat = np.matmul(H,x)

            # Innovation: y = z - z_hat
            y = z - z_hat

            # Innovation Covariance: S = H * P * H' + R
            S = np.matmul(np.matmul(H,P),np.transpose(H)) + R

            # Kalman Gain: K = P * H' * S^-1
            K =  np.matmul(np.matmul(P,np.transpose(H)),np.linalg.inv(S))

            # Kalman State Update: x_update = x + K*y
            x_update = x + np.matmul(K,y)

            # Kalman Covariance Update: P_update = (I - K*H)*P
            P_update = np.matmul((np.eye(4) - np.matmul(K,H)),P)

            # Save Updated State
            self.innovation = y
            self.innovation_covariance = S
            self.state = x_update
            self.covariance = P_update

        else:
            # Set Initial State and Covariance 
            init_vel_std = 10

            self.state = np.array([measurement[0],measurement[1],0,0])
            self.covariance = np.diag(np.array([self.R[0,0],self.R[1,1],init_vel_std**2,init_vel_std**2]))

        return 




# Run the Simulation
run_sim(KalmanFilterModel, sim_options, {})