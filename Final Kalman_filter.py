# from array import array
import numpy as np
import matplotlib.pyplot as plt
import csv

class KalmanFilter(object):
    def __init__(self):
        
        # Define sampling time (time for 1 cycle)
        self.dt = 0.1

        # Define the  control input variables
        # self.u = np.matrix([[u_x],[u_y]])
        self.u = np.matrix([[0],[0]]) #no acceleration in x and y 

        # standard deviation of the measurement in x-direction
        self.x_std_meas = 10 #value given

        # standard deviation of the measurement in y-direction
        self.y_std_meas = 10 #value given

        # Intial State
        self.x = np.matrix([[380], [-300], [12.5], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        # not in used, refer to Eqn. 3
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        #Initial Process Noise Covariance
        self.Q0 = 0.1
        self.Q = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]) * self.Q0      

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[self.x_std_meas**2, 0],
                           [0, self.y_std_meas**2]])

        #Initial Covariance Matrix
        self.P = np.matrix([[80, 0, 0, 0],
                            [0, 80, 0, 0],
                            [0, 0, 15, 0],
                            [0, 0, 0, 15]])

    def predict(self):
        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eqn.(2)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q               Eqn.(4)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        # S = H*P*H'+R                 Part of Eqn. (5)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)     Eqn.(5)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eqn.(7)

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eqn.(6)
        return self.x[0:2].tolist(), self.x[0:2] , self.P

if __name__ == "__main__":
    KalmanFilter_test = KalmanFilter()
    kalman_x = []
    kalman_y = []
    true_x = []
    true_y = []
    measured_x = []
    measured_y = []
    #measured data from excel
    data = [[[387.72], [-300.06]],
            [[382.86], [-284.94]],
            [[399.66], [-294.4]],
            [[380.77], [-287.84]],
            [[349.82], [-291.5]],
            [[340.51], [-282.48]],
            [[335.39], [-283.45]],
            [[319.15], [-296.09]],
            [[310.47], [-324.37]],
            [[288.96], [-300.2]],
            [[285.93], [-319.68]],
            [[267.9], [-308.17]],
            [[251.92], [-303.09]],
            [[243.14], [-296.92]],
            [[226.44], [-315.18]],
            [[209.71], [-326.58]],
            [[206.68], [-317.5]],
            [[209.73], [-320.68]],
            [[184.26], [-317.7]],
            [[183.67], [-326.83]],
            [[169.09], [-326.07]],
            [[141.19], [-324.78]],
            [[132.18], [-340.58]],
            [[128.21], [-339.79]],
            [[112.82], [-332.93]],
            [[112.29], [-343.03]],
            [[92.79], [-345.61]],
            [[79.5], [-340.99]],
            [[58.72], [-344.82]],
            [[58.59], [-348.81]],
            [[57.57], [-357.99]]]

    #Obtaining predicted values of Kalman Filter and storing into array
    for i in range(len(data)):
        arr_2d = np.reshape(data[i], (2, 1))
        KalmanFilter_test.predict()
        kalman_list, kalman_data, P_update = KalmanFilter_test.update(arr_2d)
        for x_val in kalman_list[0]:
            kalman_x.append(x_val)
        for y_val in kalman_list[1]:
            kalman_y.append(y_val)

    #To read value from excel and append into arrays for plotting
    with open("/Users/zannlim/Desktop/MeasGT.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            measured_x.append(float(row[1]))
            measured_y.append(float(row[2]))
            true_x.append(float(row[3]))
            true_y.append(float(row[4]))

    #Plotting graph using matplotlib
    plt.scatter(kalman_x, kalman_y, label = "predicted values from kalman filter")
    plt.scatter(true_x, true_y, label = "true values")
    plt.scatter(measured_x,measured_y, label = "measured values")
    plt.legend()
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.show()  

    plt.plot(kalman_x, kalman_y, label = "predicted values from kalman filter")
    plt.plot(true_x, true_y, label = "true values")
    plt.plot(measured_x,measured_y, label = "measured values")
    plt.legend()
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.show()             
