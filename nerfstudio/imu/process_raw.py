import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def process_imu_data(data):
    # Extracting sensor data
    accelerometer_data = []
    gyroscope_data = []
    timestamps = []

    for entry in data:
        if 'sensor' in entry:
            sensor = entry['sensor']
            if sensor['type'] == 'accelerometer':
                accelerometer_data.append(sensor['values'])
                timestamps.append(entry['time'])
            elif sensor['type'] == 'gyroscope':
                gyroscope_data.append(sensor['values'])
                timestamps.append(entry['time'])

    accelerometer_data = np.array(accelerometer_data)
    gyroscope_data = np.array(gyroscope_data)
    timestamps = np.array(timestamps)

    # Calculate time intervals
    time_intervals = np.diff(timestamps, prepend=timestamps[0])

    # Preintegration of accelerometer data to get velocity and position
    velocity = cumtrapz(accelerometer_data, dx=time_intervals[:, None], initial=0)
    position = cumtrapz(velocity, dx=time_intervals[:, None], initial=0)

    # Plotting the path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(position[:, 0], position[:, 1], position[:, 2], label='Estimated Path')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Path Estimated from Accelerometer Data')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    file_path = 'imu_data.json'
    data = load_data(file_path)
    print(data)