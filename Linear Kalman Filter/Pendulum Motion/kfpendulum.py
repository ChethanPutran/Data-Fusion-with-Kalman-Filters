import numpy as np
from .kfmodels import KalmanFilterBase
from scipy.linalg import expm


# Kalman Filter Model
class KalmanFilterModel(KalmanFilterBase):
 
    def initialise(self, time_step, torque_std, meas_std, mass = 1, length = 0.5, init_on_measurement=False, init_pos_std = 0.1, init_vel_std = 0.001):
        dt = time_step
        
        A = np.array([[0,1],[-9.81/length,0]])

        # Set Model F and H Matrices
        self.F = expm(A*dt)
        #self.F = np.array([[1,dt],[-9.81*dt/length,1]])
        self.H = np.array([[1,0]])

        # Set R and Q Matrices
        self.Q = np.diag(np.array([0,1]) * (torque_std*torque_std))
        self.R = meas_std*meas_std

        # Set Initial State and Covariance 
        if init_on_measurement is False:
            self.state = np.transpose(np.array([[0,0]])) # Assume we are at zero position and velocity
            self.covariance = np.diag(np.array([init_pos_std*init_pos_std,init_vel_std*init_vel_std]))
        
        return
    
    def prediction_step(self):
        # Make Sure Filter is Initialised
        if self.state is not None:
            x = self.state
            P = self.covariance

            # Calculate Kalman Filter Prediction
            x_predict = np.matmul(self.F, x) 
            P_predict = np.matmul(self.F, np.matmul(P, np.transpose(self.F))) + self.Q

           # x_predict[0] = ((x_predict[0] + np.pi) % (2 * np.pi)) - np.pi
            #P_predict = (P_predict+np.transpose(P_predict))*0.5

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
            z = measurement
            z_hat = x[0]
            y = z - z_hat
            
            #y = ((y + np.pi) % (2 * np.pi)) - np.pi
            
            S = np.matmul(H,np.matmul(P,np.transpose(H))) + R
            K = np.matmul(P,np.matmul(np.transpose(H),np.linalg.inv(S)))
            x_update = x + K*y # np.matmul(K, y)

            P_update = np.matmul( (np.eye(2) - np.matmul(K,H)), P)

            #A = (np.eye(2) - np.matmul(K,H))

            #P_update = np.matmul(np.matmul(A,P),np.transpose(A)) + np.matmul(K,np.transpose(K))*R

            # Save Updated State
            self.innovation = y
            self.innovation_covariance = S
            self.state = x_update
            self.covariance = P_update

        else:

            # Set Initial State and Covariance 
            self.state = np.transpose(np.array([[measurement,0]]))
            self.covariance = np.diag(np.array([self.R,0.1])) # Assume we don't know our velocity

        return 