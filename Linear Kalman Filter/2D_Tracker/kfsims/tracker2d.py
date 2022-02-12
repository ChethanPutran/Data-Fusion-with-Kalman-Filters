import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from .vehiclemodel2d import VehicleModel2D



def run_sim(KalmanFilterModel, sim_option, kf_options):

    dt = sim_option['time_step']         
    end_time = sim_option['end_time']
    measurement_rate = sim_option['measurement_rate']
    measurement_noise_std = sim_option['measurement_noise_std']
    motion_type = sim_option['motion_type']
    draw_plots = sim_option['draw_plots']
    draw_animation = sim_option['draw_animation']
    start_at_origin = sim_option['start_at_origin']
    start_at_random_speed = sim_option['start_at_random_speed']
    start_at_random_heading = sim_option['start_at_random_heading']

    initial_x_position = 0
    initial_y_position = 0
    if start_at_origin is False:
        initial_x_position = 1000*(np.random.rand()-0.5)
        initial_y_position = 1000*(np.random.rand()-0.5)

    initial_speed = 10
    if start_at_random_speed is True:
        initial_speed = np.random.rand() * 20

    initial_heading = 45
    if start_at_random_heading is True:
        initial_heading = 180*(np.random.rand()-0.5)


    vehicle_params = {'initial_x_position': initial_x_position,
                      'initial_y_position': initial_x_position,
                      'initial_heading': np.deg2rad(initial_heading),
                      'initial_speed': initial_speed}




    meas_steps = np.ceil((1/measurement_rate)/dt).astype(int)
    num_steps = np.ceil(end_time/dt).astype(int)

    # Create the Simulation Objects
    vehicle_model = VehicleModel2D()
    kalman_filter = KalmanFilterModel()

    # Initialise the Components
    vehicle_model.initialise(vehicle_params)
    kalman_filter.initialise(dt, **kf_options)
    

    # Save the Initial States
    time_history = np.linspace(0.0, dt*num_steps, num_steps+1)
    vehicle_position_history = [vehicle_model.get_position()]
    vehicle_velocity_history = [vehicle_model.get_velocity()]
    measurement_history = [None]
    estimated_state_history = [kalman_filter.get_current_state()]
    estimated_covariance_history = [kalman_filter.get_current_covariance()]
    estimated_error_history = [None]

    measurement_innovation_history = [None]
    measurement_innovation_covariance_history = [None]

    rand_param = np.random.randn()*0.1

    # Run the Simulation
    for k in range(1, num_steps+1):

        # Update the Vehicle Model
        if motion_type == 'random':
            vehicle_model.update_vehicle(dt, np.random.randn()*2 , np.random.randn()*2)
        elif motion_type == 'circle':
            vehicle_model.update_vehicle(dt, 0 , np.deg2rad(360/120))
        elif motion_type == 'linear':
            vehicle_model.update_vehicle(dt,rand_param,0)
        else:
            vehicle_model.update_vehicle(dt, 0, 0)

        vehicle_position = vehicle_model.get_position()
        vehicle_velocity = vehicle_model.get_velocity()

        # KF Prediction
        kalman_filter.prediction_step()

        # KF Measurement
        measurement = None
        if (k % meas_steps) == 0:
            x_meas = vehicle_position[0] + np.random.randn()*measurement_noise_std
            y_meas = vehicle_position[1] + np.random.randn()*measurement_noise_std
            measurement = [x_meas,y_meas]
            kalman_filter.update_step(measurement)
            measurement_innovation_history.append(kalman_filter.get_last_innovation())
            measurement_innovation_covariance_history.append(kalman_filter.get_last_innovation_covariance())
            
        # Estimation Error
        estimation_error = None
        estimated_state = kalman_filter.get_current_state()
        if estimated_state is not None:
            estimation_error = [estimated_state[0] - vehicle_position[0], 
                                estimated_state[1] - vehicle_position[1],
                                estimated_state[2] - vehicle_velocity[0], 
                                estimated_state[3] - vehicle_velocity[1]]
        
        # Save Data
        vehicle_position_history.append(vehicle_model.get_position())
        vehicle_velocity_history.append(vehicle_model.get_velocity())
        measurement_history.append(measurement)
        estimated_state_history.append(kalman_filter.get_current_state())
        estimated_covariance_history.append(kalman_filter.get_current_covariance())
        estimated_error_history.append(estimation_error)

    # Calculate Stats
    x_innov_std = np.std([v[0] for v in measurement_innovation_history if v is not None])
    y_innov_std = np.std([v[1] for v in measurement_innovation_history if v is not None])
    pos_mse = np.mean([(v[0]**2+v[1]**2) for v in estimated_error_history if v is not None])
    vel_mse = np.mean([(v[2]**2+v[3]**2) for v in estimated_error_history if v is not None])
    print('X Position Measurement Innovation Std: {} (m)'.format(x_innov_std))
    print('Y Position Measurement Innovation Std: {} (m)'.format(y_innov_std))
    print('Position Mean Squared Error: {} (m)^2'.format(pos_mse))
    print('Velocity Mean Squared Error: {} (m/s)^2'.format(vel_mse))



    if draw_plots is True:
        # Plot Analysis
        fig1 = plt.figure(constrained_layout=True)
        fig1_gs = gridspec.GridSpec(ncols=2, nrows=4, figure=fig1)

        fig1_ax1 = fig1.add_subplot(fig1_gs[0,0],title='X Position')
        fig1_ax2 = fig1.add_subplot(fig1_gs[1,0],title='Y Position')
        fig1_ax3 = fig1.add_subplot(fig1_gs[2,0],title='X Velocity')
        fig1_ax4 = fig1.add_subplot(fig1_gs[3,0],title='Y Velocity')
        fig1_ax5 = fig1.add_subplot(fig1_gs[0,1],title='X Position Error')
        fig1_ax6 = fig1.add_subplot(fig1_gs[1,1],title='Y Position Error')
        fig1_ax7 = fig1.add_subplot(fig1_gs[2,1],title='X Velocity Error')
        fig1_ax8 = fig1.add_subplot(fig1_gs[3,1],title='Y Velocity Error')
        fig1_ax1.grid(True)
        fig1_ax2.grid(True)
        fig1_ax3.grid(True)
        fig1_ax4.grid(True)
        fig1_ax5.grid(True)
        fig1_ax6.grid(True)
        fig1_ax7.grid(True)
        fig1_ax8.grid(True)
        fig1_ax4.set_xlabel('Time (sec)')
        fig1_ax8.set_xlabel('Time (sec)')
        fig1_ax1.set_ylabel('X Position (m)')
        fig1_ax2.set_ylabel('Y Position (m)')
        fig1_ax3.set_ylabel('X Velocity (m/s)')
        fig1_ax4.set_ylabel('Y Velocity (m/s)')

        # Plot Vehicle State
        fig1_ax1.plot(time_history, [v[0] for v in vehicle_position_history], 'b')
        fig1_ax2.plot(time_history, [v[1] for v in vehicle_position_history], 'b')
        fig1_ax3.plot(time_history, [v[0] for v in vehicle_velocity_history], 'b')
        fig1_ax4.plot(time_history, [v[1] for v in vehicle_velocity_history], 'b')

        # Plot Estimated States
        time_plot = [t for t,v in zip(time_history, estimated_state_history) if v is not None ]
        fig1_ax1.plot(time_plot, [v[0] for v in estimated_state_history if v is not None], 'r')
        fig1_ax2.plot(time_plot, [v[1] for v in estimated_state_history if v is not None], 'r')
        fig1_ax3.plot(time_plot, [v[2] for v in estimated_state_history if v is not None], 'r')
        fig1_ax4.plot(time_plot, [v[3] for v in estimated_state_history if v is not None], 'r')

        # Plot Measurements
        time_plot = [t for t,v in zip(time_history, measurement_history) if v is not None ]
        fig1_ax1.plot(time_plot, [v[0] for v in measurement_history if v is not None], 'k+')
        fig1_ax2.plot(time_plot, [v[1] for v in measurement_history if v is not None], 'k+')

        # Plot Errors
        time_plot = [t for t,v in zip(time_history, estimated_error_history) if v is not None ]
        fig1_ax5.plot(time_plot, [v[0] for v in estimated_error_history if v is not None], 'r')
        fig1_ax6.plot(time_plot, [v[1] for v in estimated_error_history if v is not None], 'r')
        fig1_ax7.plot(time_plot, [v[2] for v in estimated_error_history if v is not None], 'r')
        fig1_ax8.plot(time_plot, [v[3] for v in estimated_error_history if v is not None], 'r')
        time_plot = [t for t,v in zip(time_history, estimated_covariance_history) if v is not None ]
        fig1_ax5.plot(time_plot, [3.0*np.sqrt(v[0][0]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax6.plot(time_plot, [3.0*np.sqrt(v[1][1]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax7.plot(time_plot, [3.0*np.sqrt(v[2][2]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax8.plot(time_plot, [3.0*np.sqrt(v[3][3]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax5.plot(time_plot, [-3.0*np.sqrt(v[0][0]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax6.plot(time_plot, [-3.0*np.sqrt(v[1][1]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax7.plot(time_plot, [-3.0*np.sqrt(v[2][2]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax8.plot(time_plot, [-3.0*np.sqrt(v[3][3]) for v in estimated_covariance_history if v is not None], 'g')

        fig2 = plt.figure(constrained_layout=True)
        fig2_gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig2)

        fig2_ax1 = fig2.add_subplot(fig2_gs[0,0],title='X Measurement Innovation')
        fig2_ax2 = fig2.add_subplot(fig2_gs[1,0],title='Y Measurement Innovation')
        fig2_ax1.grid(True)
        fig2_ax2.grid(True)
        fig2_ax2.set_xlabel('Time (sec)')

        time_plot = [t for t,v in zip(time_history, measurement_innovation_history) if v is not None ]
        fig2_ax1.plot(time_plot, [v[0] for v in measurement_innovation_history if v is not None], 'b')
        fig2_ax2.plot(time_plot, [v[1] for v in measurement_innovation_history if v is not None], 'b')
        time_plot = [t for t,v in zip(time_history, measurement_innovation_covariance_history) if v is not None ]
        fig2_ax1.plot(time_plot, [1.0*np.sqrt(v[0][0]) for v in measurement_innovation_covariance_history if v is not None], 'g--')
        fig2_ax1.plot(time_plot, [-1.0*np.sqrt(v[0][0]) for v in measurement_innovation_covariance_history if v is not None], 'g--')
        fig2_ax2.plot(time_plot, [1.0*np.sqrt(v[1][1]) for v in measurement_innovation_covariance_history if v is not None], 'g--')
        fig2_ax2.plot(time_plot, [-1.0*np.sqrt(v[1][1]) for v in measurement_innovation_covariance_history if v is not None], 'g--')

    if draw_animation is True:
        
        # Plot Animation
        fig2 = plt.figure(constrained_layout=True)
        fig_ax2 = fig2.add_subplot(111,title='2D Position', aspect='equal')#, autoscale_on=True, xlim=(0, 1000), ylim=(0, 1000))
        fig_ax2.set_xlabel('X Position (m)')
        fig_ax2.set_ylabel('Y Position (m)')
        fig_ax2.grid(True)
        vehicle_plot, = fig_ax2.plot([], [], 'bo-', lw=1)
        meas_plot, = fig_ax2.plot([],[],'+k')
        estimate_plot, = fig_ax2.plot([],[], 'ro-')
        estimate_var_plot, = fig_ax2.plot([],[], 'g-')

        def update_plot(i):
            # Plot Vehicle
            vehicle_plot.set_data(vehicle_position_history[i])
            
            # Plot Measurements
            x_data = [meas[0] for meas in measurement_history[1:i] if meas is not None]
            y_data = [meas[1] for meas in measurement_history[1:i] if meas is not None]
            meas_plot.set_data(x_data,y_data)

            # Plot Estimates
            if estimated_state_history[i] is not None:
                est_xpos = estimated_state_history[i][0]
                est_ypos = estimated_state_history[i][1]
                estimate_plot.set_data(est_xpos, est_ypos)
                if estimated_covariance_history[i] is not None:
                    cov = estimated_covariance_history[i]
                    cov_mat = np.array([[cov[0][0],cov[0][1]],[cov[1][0],cov[1][1]]])
                    U, S, V = np.linalg.svd(cov_mat)
                    theta = np.linspace(0,2*np.pi, 100)
                    theta_mat = np.array([np.cos(theta),np.sin(theta)])
                    D = np.matmul(np.matmul(U,np.diag(3.0*np.sqrt(S))),theta_mat)
                    estimate_var_plot.set_data([x+est_xpos for x in D[0]], [y+est_ypos for y in D[1]])

            fig_ax2.relim()
            fig_ax2.autoscale_view()

            return vehicle_plot, meas_plot, estimate_plot, estimate_var_plot

        # # Create the Animation
        plot_animation = animation.FuncAnimation(fig2, update_plot, frames=range(0,num_steps,50),interval=1, repeat=False, blit=False)


    # Show Animation
    plt.show()

    return (pos_mse, vel_mse)