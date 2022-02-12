import numpy as np
from kfsims.tracker2d import run_sim
from kfsims.kftracker2d import KalmanFilterModel

# Simulation Options
sim_options = {'time_step': 0.01,
               'end_time': 120,
               'measurement_rate': 1,
               'measurement_noise_std': 10,
               'motion_type': 'circle',
               'start_at_origin': False,
               'start_at_random_speed': True,
               'start_at_random_heading': True,
               'draw_plots': True,
               'draw_animation': True} 

kf_options = {'accel_std':0.5, # Q Matrix Param
              'meas_std':10, # R Matrix  
              'init_on_measurement':True}

# Run the Simulation
run_sim(KalmanFilterModel, sim_options, kf_options)