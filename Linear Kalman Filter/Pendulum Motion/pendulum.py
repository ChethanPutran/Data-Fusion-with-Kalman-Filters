import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from .pendulummodel import PendulumModel


def run_sim(KalmanFilterModel, sim_option, kf_options):

    dt = sim_option['time_step']
    end_time = sim_option['end_time']
    measurement_rate = sim_option['measurement_rate']
    measurement_noise_std = sim_option['measurement_noise_std']
    draw_plots = sim_option['draw_plots']
    draw_animation = sim_option['draw_animation']
    start_at_random_angle = sim_option['start_at_random_angle']

    initial_theta = 10

    if start_at_random_angle is True:
        initial_theta = 45*2*(np.random.rand()-0.5)

    pendulum_params = {
        'initial_position': np.deg2rad(initial_theta),
        'initial_velocity': np.deg2rad(0),
        'mass': 1,
        'length': 0.5
        }

    meas_steps = np.ceil((1/measurement_rate)/dt).astype(int)
    num_steps = np.ceil(end_time/dt).astype(int)

    # Create the Simulation Objects
    pendulum_model = PendulumModel()
    kalman_filter = KalmanFilterModel()

    # Initialise the Components
    pendulum_model.initialise(pendulum_params)
    kalman_filter.initialise(dt, **kf_options)
    
    # Save the Initial States
    time_history = np.linspace(0.0, dt*num_steps, num_steps+1)
    pendulum_position_history = [pendulum_model.get_position()]
    pendulum_velocity_history = [pendulum_model.get_velocity()]
    measurement_history = [None]
    estimated_state_history = [kalman_filter.get_current_state()]
    estimated_covariance_history = [kalman_filter.get_current_covariance()]
    estimated_error_history = [None]
    measurement_innovation_history = [None]
    measurement_innovation_covariance_history = [None]

    # Run the Simulation
    for k in range(1, num_steps+1):

        # Update the  Model
        pendulum_model.update(dt, 0)
        pendulum_position = pendulum_model.get_position()
        pendulum_velocity = pendulum_model.get_velocity()

        # KF Prediction
        kalman_filter.prediction_step()

        # KF Measurement
        measurement = None
        if (k % meas_steps) == 0:
            theta_meas = pendulum_position + np.random.randn()*measurement_noise_std
            measurement = theta_meas
            kalman_filter.update_step(measurement)
            measurement_innovation_history.append(kalman_filter.get_last_innovation())
            measurement_innovation_covariance_history.append(kalman_filter.get_last_innovation_covariance())
            
        # Estimation Error
        estimation_error = None
        estimated_state = kalman_filter.get_current_state()
        if estimated_state is not None:
            estimation_error = [estimated_state[0] - pendulum_position, 
                                estimated_state[1] - pendulum_velocity]
            
        
        # Save Data
        pendulum_position_history.append(pendulum_position)
        pendulum_velocity_history.append(pendulum_velocity)
        measurement_history.append(measurement)
        estimated_state_history.append(kalman_filter.get_current_state())
        estimated_covariance_history.append(kalman_filter.get_current_covariance())
        estimated_error_history.append(estimation_error)
 
    # Calculate Stats
    pos_innov_std = np.std([np.rad2deg(v[0]) for v in measurement_innovation_history if v is not None])
    pos_mse = np.mean([(np.rad2deg(v[0])**2) for v in estimated_error_history if v is not None])
    vel_mse = np.mean([(np.rad2deg(v[1])**2) for v in estimated_error_history if v is not None])
    print('Position Measurement Innovation Std: {} (m)'.format(pos_innov_std))
    print('Position Mean Squared Error: {} (deg)^2'.format(pos_mse))
    print('Velocity Mean Squared Error: {} (deg/s)^2'.format(vel_mse))

    if draw_plots is True:
        # Plot Analysis
        fig1 = plt.figure(constrained_layout=True)
        fig1_gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig1)

        fig1_ax1 = fig1.add_subplot(fig1_gs[0,0],title='Theta Position')
        fig1_ax2 = fig1.add_subplot(fig1_gs[1,0],title='Theta Velocity')
        fig1_ax3 = fig1.add_subplot(fig1_gs[0,1],title='Theta Position Error')
        fig1_ax4 = fig1.add_subplot(fig1_gs[1,1],title='Theta Velocity Error')

        fig1_ax1.grid(True)
        fig1_ax2.grid(True)
        fig1_ax3.grid(True)
        fig1_ax4.grid(True)
        fig1_ax2.set_xlabel('Time (sec)')
        fig1_ax4.set_xlabel('Time (sec)')
        fig1_ax1.set_ylabel('Theta Position (deg)')
        fig1_ax2.set_ylabel('Theta Velocity (deg/s)')

        # Plot State
        fig1_ax1.plot(time_history, np.rad2deg(pendulum_position_history), 'b')
        fig1_ax2.plot(time_history, np.rad2deg(pendulum_velocity_history), 'b')

        # Plot Estimated States
        time_plot = [t for t,v in zip(time_history, estimated_state_history) if v is not None ]
        fig1_ax1.plot(time_plot, [np.rad2deg(v[0]) for v in estimated_state_history if v is not None], 'r')
        fig1_ax2.plot(time_plot, [np.rad2deg(v[1]) for v in estimated_state_history if v is not None], 'r')

        # Plot Measurements
        time_plot = [t for t,v in zip(time_history, measurement_history) if v is not None ]
        fig1_ax1.plot(time_plot, [np.rad2deg(v) for v in measurement_history if v is not None], 'k+')


        # Plot Errors
        time_plot = [t for t,v in zip(time_history, estimated_error_history) if v is not None ]
        fig1_ax3.plot(time_plot, [np.rad2deg(v[0]) for v in estimated_error_history if v is not None], 'r')
        fig1_ax4.plot(time_plot, [np.rad2deg(v[1]) for v in estimated_error_history if v is not None], 'r')

        time_plot = [t for t,v in zip(time_history, estimated_covariance_history) if v is not None ]
        fig1_ax3.plot(time_plot, [3.0*np.rad2deg(np.sqrt(v[0][0])) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax4.plot(time_plot, [3.0*np.rad2deg(np.sqrt(v[1][1])) for v in estimated_covariance_history if v is not None], 'g')

        fig1_ax3.plot(time_plot, [-3.0*np.rad2deg(np.sqrt(v[0][0])) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax4.plot(time_plot, [-3.0*np.rad2deg(np.sqrt(v[1][1])) for v in estimated_covariance_history if v is not None], 'g')


        fig2 = plt.figure(constrained_layout=True)

        fig2_ax1 = fig2.add_subplot(111,title='Measurement Innovation')
        fig2_ax1.grid(True)
        fig2_ax1.set_xlabel('Time (sec)')

        time_plot = [t for t,v in zip(time_history, measurement_innovation_history) if v is not None ]
        fig2_ax1.plot(time_plot, [np.rad2deg(v[0]) for v in measurement_innovation_history if v is not None], 'b')

        time_plot = [t for t,v in zip(time_history, measurement_innovation_covariance_history) if v is not None ]
        fig2_ax1.plot(time_plot, [1.0*np.rad2deg(np.sqrt(v[0])) for v in measurement_innovation_covariance_history if v is not None], 'g--')
        fig2_ax1.plot(time_plot, [-1.0*np.rad2deg(np.sqrt(v[0])) for v in measurement_innovation_covariance_history if v is not None], 'g--')


    if draw_animation is True:
        
        # Plot Animation
        fig2 = plt.figure(constrained_layout=True)
        fig_ax = fig2.add_subplot(111,title='Pendulum Position', aspect='equal',xlim=(-1, 1), ylim=(-1.1, 0))
        fig_ax.grid(True)
        mass_plot, = fig_ax.plot([], [], 'bo')
        arm_plot, = fig_ax.plot([],[],'b-')
        est_mass_plot, = fig_ax.plot([], [], 'ro')
        est_arm_plot, = fig_ax.plot([],[],'r-')

        def update_plot(i):

            # Plot Pendulum
            theta = pendulum_position_history[i]
            x = 0 + 1 * np.sin(theta)
            y = 0 - 1 * np.cos(theta)
            mass_plot.set_data([x,y])
            arm_plot.set_data([0,x],[0,y])

            # Plot Estimates
            if estimated_state_history[i] is not None:
                est_theta = estimated_state_history[i][0]
                x = 0 + 1 * np.sin(est_theta)
                y = 0 - 1 * np.cos(est_theta)
                est_mass_plot.set_data([x,y])
                est_arm_plot.set_data([0,x],[0,y])

            return mass_plot, arm_plot, est_mass_plot, arm_plot

        # # Create the Animation
        plot_animation = animation.FuncAnimation(fig2, update_plot, frames=range(0,num_steps,10),interval=1, repeat=False, blit=False)


    # Show Animation
    plt.show()

    return (pos_mse, vel_mse)