"""
Air Mobility Project- 16665
Author: Guanya Shi
guanyas@andrew.cmu.edu
"""

import numpy as np
import meshcat
import meshcat.geometry as geometry
import meshcat.transformations as tf
import matplotlib.pyplot as plt
from time import sleep
from control import lqr
import argparse

class Quadrotor_2D():
	def __init__(self):
		# parameters
		self.m = 0.027 # kg
		self.J = 8.571710e-5 # inertia
		self.arm = 0.0325 # arm length
		self.g = 9.81 # gravity

		# control actuation matrix
		self.B = np.array([[1, 1],
						   [-self.arm, self.arm]])
		self.B_inv = np.linalg.inv(self.B)

		# noise level
		self.sigma_t = 0.25 # for translational dynamics
		self.sigma_r = 0.25 # for rotational dynamics

		# initial state
		self.p = np.array([0., 0])
		self.v = np.array([0., 0])
		self.theta = 0.
		self.omega = 0.


		# initial control (hovering)
		self.u = np.array([self.m*self.g/2, self.m*self.g/2])

		# control limit for each rotor (N)
		self.umin = 0.
		self.umax = 0.024 * self.g

		# total time and discretizaiton step
		self.dt = 0.01
		self.step = 0
		self.t = 0.

	def reset(self):
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		self.p = np.array([0., 0])
		self.v = np.array([0., 0])
		self.theta = 0.
		self.omega = 0.
		self.u = np.array([self.m*self.g/2, self.m*self.g/2])
		self.step = 0
		self.t = 0.

	def dynamics(self, u):
		'''
		Problem A-1: Based on lecture 2, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		self.u is the control input (two rotor forces).
		Hint: first convert self.u to total thrust and torque using the control actuation matrix.
		'''
		u = np.clip(u, self.umin, self.umax)
		self.u = np.clip(u, self.umin, self.umax)

		actuator = self.B@self.u		

		pdot = self.v 
		vdot = np.array([0, -self.g]) + actuator[0] * (np.array([-np.sin(self.theta), np.cos(self.theta)])) / self.m
		thetadot = self.omega 
		omegadot = actuator[1]/self.J

		self.p += self.dt * pdot
		self.v += self.dt * vdot + self.dt * self.sigma_t * np.random.normal(size=2)
		self.theta += self.dt * thetadot
		self.omega += self.dt * omegadot + self.dt * self.sigma_r * np.random.normal()

		self.t += self.dt
		self.step += 1

	def cascaded_control(self, p_d, v_d, a_d, omega_d=0., tau_d=0.):
        
		K_P = 5.0
		K_D = 1.0

		
		K_Ptau = 170.0
		K_Dtau = 10.0

		
		e_p = self.p - p_d 
		e_v = self.v - v_d
		f_d = self.m * (- K_P * e_p - K_D * e_v - np.array([0, -self.g]) + a_d)
		

		
		T_des = f_d@np.array([[-np.sin(self.theta)], [np.cos(self.theta)]])
		
		theta_des = -np.arctan2(f_d[0], f_d[1])

		
		e_phi = self.theta - theta_des 
		e_omega = self.omega - omega_d 
		tau = self.J*np.array([-K_Ptau * e_phi - K_Dtau * e_omega + tau_d])

		
		
		u = (self.B_inv @ np.array([T_des, tau])).flatten()
		# print(u)
		
		return u	    

	def linear_control(self, p_d):
		'''
		Problem A-3: Based on lecture 3, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		Your goal is to develop a LQR control based on the linearized model around the hovering condition.
		Hint: use the lqr function in the control library.
		'''
		A = np.array([
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1],
			[0, 0, -self.g, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0]
			])
		

		B = np.array([
			[0, 0],
			[0, 0],
			[0, 0],
			[0, 0],
			[1, 0],
			[0, 1]
			])
		Q = np.diag([15, 15, 1, 1, 1, 0.1])

		R = np.diag([0.1, 0.1])
		K, _, _ = lqr(A, B, Q, R)
		x = np.array([self.p[0], self.p[1], self.theta, self.v[0], self.v[1], self.omega])
		x_d = np.array([p_d[0], p_d[1], 0, 0, 0, 0])
		u = -np.dot(K,x - x_d)
		
		u[0] *= self.m
		u[1] *= self.J
		u[0] += self.m*self.g
		control_mat = np.array([u[0], u[1]])
		u = self.B_inv @ control_mat
		return u

def plot(time, pos, vel, control, theta, omega, pos_des):
	plt.figure(figsize=(16, 4))
	plt.subplot(1, 4, 1)
	colors = ['tab:blue', 'tab:orange']
	names = ['x', 'y']
	for i in range(2):
		plt.plot(time, pos[:,i], color=colors[i], label=names[i]+" actual")
		plt.plot(time, pos_des[:,i], '--', color=colors[i], label=names[i]+" desired")
	plt.xlabel("time (s)")
	plt.ylabel("pos (m)")
	plt.legend()
	plt.subplot(1, 4, 2)
	plt.plot(time, vel)
	plt.xlabel("time (s)")
	plt.ylabel("vel (m/s)")
	plt.legend(["x", "y"])
	plt.subplot(1, 4, 3)
	plt.plot(time, control)
	plt.xlabel("time (s)")
	plt.ylabel("control (N)")
	plt.legend(["1", "2"])
	plt.subplot(1, 4, 4)
	plt.plot(time, theta)
	plt.plot(time, omega)
	plt.xlabel("time (s)")
	plt.legend(["theta (rad)", "omega (rad/s)"])
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	robot = Quadrotor_2D()
	total_time = 2 * np.pi
	# total_time = 2
	total_step = int(total_time/robot.dt+1)
	time = np.linspace(0, total_time, total_step)
	pos = np.zeros((total_step, 2))
	pos_des = np.zeros((total_step, 2))
	vel = np.zeros((total_step, 2))
	control = np.zeros((total_step, 2))
	control[0,:] = robot.u
	theta = np.zeros(total_step)
	omega = np.zeros(total_step)

	parser = argparse.ArgumentParser()
	parser.add_argument('question', type=int)
	question = parser.parse_args().question
	
	'''
	Problem A-1: system modeling
	'''
	if question == 1:
		robot.sigma_r = 0.
		robot.sigma_t = 0.
		for i in range(21):
			u = np.array([0.019, 0.023]) * 9.81
			robot.dynamics(u)
			if i % 10 == 0:
				print('************************')
				print('pos:', robot.p)
				print('vel:', robot.v)
				print('theta:', robot.theta)
				print('omega:', robot.omega)

	'''
	Problem A-2: cascaded setpoint control
	Complete p_d, v_d, and a_d
	'''
	rise_time_calculated_2 = False
	t_10_2 = None
	t_90_2 = None
	max_overshoot2 = 0
	robot.reset()	
	while True:
		if question != 2 or robot.step >= total_step-1:
			break
		p_d = np.array([1, 1])
		v_d = np.array([0, 0])
		a_d = np.array([0, 0])
		u = robot.cascaded_control(p_d, v_d, a_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega

		overshoot = np.max(np.abs(robot.p - p_d))
		max_overshoot2 = max(overshoot, max_overshoot2)


		if not rise_time_calculated_2:
			threshold_10_2 = 0.1 * p_d
			threshold_90_2 = 0.9 * p_d
        
			if t_10_2 is None and np.all(robot.p < threshold_10_2):
				t_10_2 = robot.t
			elif t_10_2 is not None and t_90_2 is None and np.all(robot.p > threshold_90_2):
				t_90_2 = robot.t
				rise_time = t_90_2 - t_10_2
				print(f"Rise time (10% to 90%): {rise_time:.4f} seconds")
				print(f'Max Overshoot is : {max_overshoot2}')
				rise_time_calculated_2 = True
		
		# print(f"Step {robot.step}, Desired Pos: {p_d}, Actual Pos: {robot.p}, Control Input: {u}")
		
	if question == 2:
		pos_des[0,:] = p_d
		plot(time, pos, vel, control, theta, omega, pos_des)
		# ces = np.vstack(control_energy_stack)
		ce2 = np.mean(np.linalg.norm(control)**2)
		mse2 = np.mean(np.square(np.subtract(pos_des, pos)))
		rmse2= np.sqrt(mse2)
		print(f"Average Control Energy = {ce2}")
		print(f"RMSE is equal to: {rmse2}")

	'''
	Problem A-3: linear setpoint control
	Complete p_d
	'''
	rise_time_calculated_3 = False
	t_10_3 = None
	t_90_3 = None
	max_overshoot3 = 0
	robot.reset()
	while True:
		if question != 3 or robot.step >= total_step-1:
			break
		p_d = np.array([1,0])
		u = robot.linear_control(p_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
		overshoot = np.max(np.abs(robot.p - p_d))
		max_overshoot3 = max(overshoot, max_overshoot3)
		if not rise_time_calculated_3:
			threshold_10_3 = 0.1 * p_d
			threshold_90_3 = 0.9 * p_d
        
			if t_10_3 is None and np.all(robot.p < threshold_10_3):
				t_10_3 = robot.t
			elif t_10_3 is not None and t_90_3 is None and np.all(robot.p > threshold_90_3):
				t_90_3 = robot.t
				rise_time = t_90_3 - t_10_3
				print(f"Rise time (10% to 90%): {rise_time:.4f} seconds")
				print(f'Max Overshoot is : {max_overshoot3}')
				rise_time_calculated_3 = True
	if question == 3:
		pos_des[0,:] = p_d
		plot(time, pos, vel, control, theta, omega, pos_des)
		ce2 = np.mean(np.linalg.norm(control)**2)
		mse2 = np.mean(np.square(np.subtract(pos_des, pos)))
		rmse2 = np.sqrt(mse2)
		print(f"Average Control Energy = {ce2}")
		print(f"RMSE is equal to: {rmse2}")

	'''
	Problem A-4: cascaded tracking control
	Complete p_d, v_d, and a_d
	'''
	
	robot.reset()
	while True:
		if question != 4 or robot.step >= total_step-1:
			break
		t = robot.t
		p_d = np.array([np.sin(t), 0.5 * np.cos(2*t + np.pi/2)])
		# v_d = np.array([np.cos(t), -np.sin(2*t + np.pi/2)])
		v_d = np.array([0,0])
		a_d = np.array([-np.sin(t), -2*np.cos(2*t + np.pi/2)])
		# a_d = np.array([0, 0])
		u = robot.cascaded_control(p_d, v_d, a_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 4:
		plot(time, pos, vel, control, theta, omega, pos_des)
		ce2 = np.mean(np.linalg.norm(control)**2)
		mse2 = np.mean(np.square(np.subtract(pos_des, pos)))
		rmse2= np.sqrt(mse2)
		print(f"Average Control Energy = {ce2}")
		print(f"RMSE is equal to: {rmse2}")


	'''
	Problem A-5: trajectory generation and differential flatness
	Design trajectory and tracking controllers here.
	'''
	robot.reset()
	while True:
			if question != 5 or robot.step >= total_step-1:
				break
			t = robot.t 
			T = total_time  

			A = np.array([
			[1, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 2, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 6, 0, 0, 0, 0, 0],
			[1, T, T**2, T**3, T**4, T**5, T**6, T**7, T**8],
			[0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4, 6*T**5, 7*T**6, 8*T**7],
			[0, 0, 2, 6*T, 12*T**2, 20*T**3, 30*T**4, 42*T**5, 56*T**6],
			[0, 0, 0, 6, 24*T, 60*T**2, 120*T**3, 210*T**4, 336*T**5],
			[0, 0, 0, 0, 24, 120*T, 360*T**2, 840*T**3, 1680*T**4]
			])

			B = np.array([[0], [0], [0], [0], [1], [0], [0], [0], [0]])

			
			coeffs = np.linalg.solve(A, B).flatten()

			
			p_r = np.array([coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5 + coeffs[6]*t**6 + coeffs[7]*t**7 + coeffs[8]*t**8,0])
			v_r = np.array([coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 4*coeffs[4]*t**3 + 5*coeffs[5]*t**4 + 6*coeffs[6]*t**5 + 7*coeffs[7]*t**6 + 8*coeffs[8]*t**7,0])
			a_r = np.array([2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 20*coeffs[5]*t**3 + 30*coeffs[6]*t**4 + 42*coeffs[7]*t**5 + 56*coeffs[8]*t**6,0])
			j_r = np.array([6*coeffs[3] + 24*coeffs[4]*t + 60*coeffs[5]*t**2 + 120*coeffs[6]*t**3 + 210*coeffs[7]*t**4 + 336*coeffs[8]*t**5,0])
			s_r = np.array([24*coeffs[4] + 120*coeffs[5]*t + 360*coeffs[6]*t**2 + 840*coeffs[7]*t**3 + 1680*coeffs[8]*t**5,0])
				
			
			'''
			differential flatness
			'''
			f_r = (a_r - np.array([0, robot.g]))
			y_vec = f_r/np.linalg.norm(f_r) 
			theta_r = np.arctan2(y_vec[0], y_vec[1])
			T_r = np.dot(f_r.T,y_vec)
			x_vec = np.array([np.cos(theta_r), np.sin(theta_r)])
			omega_d = (-np.dot(j_r.T, x_vec) / T_r)
			tau_d = (-(np.transpose(s_r).dot(x_vec) + (2*j_r.T.dot(y_vec))*(omega_d))/T_r)
			# omega_d = 0
			# tau_d = 0
			

			u = robot.cascaded_control(p_r, v_r, a_r, omega_d, tau_d)

			robot.dynamics(u)
			pos[robot.step,:] = robot.p
			pos_des[robot.step,:] = p_r
			vel[robot.step,:] = robot.v
			control[robot.step,:] = robot.u
			theta[robot.step] = robot.theta
			omega[robot.step] = robot.omega
	if question == 5:
		plot(time, pos, vel, control, theta, omega, pos_des)
		ce2 = np.mean(np.linalg.norm(control)**2)
		mse2 = np.mean(np.square(np.subtract(pos_des, pos)))
		rmse2= np.sqrt(mse2)
		print(f"Average Control Energy = {ce2}")
		print(f"RMSE is equal to: {rmse2}")


	'''
	Animation using meshcat
	'''
	vis = meshcat.Visualizer()
	vis.open()

	vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0,0,0]).dot(
		tf.euler_matrix(0,np.radians(-30),-np.pi/2)))

	vis["/Cameras/default/rotated/<object>"].set_transform(
		tf.translation_matrix([1,0,0]))

	vis["Quadrotor"].set_object(geometry.StlMeshGeometry.from_file('./crazyflie2.stl'))
	
	vertices = np.array([[0,0.5],[0,0],[0,0]]).astype(np.float32)
	vis["lines_segments"].set_object(geometry.Line(geometry.PointsGeometry(vertices), \
									 geometry.MeshBasicMaterial(color=0xff0000,linewidth=100.)))
	
	while True:
		for i in range(total_step):
			vis["Quadrotor"].set_transform(
				tf.translation_matrix([pos[i,0], 0, pos[i,1]]).dot(tf.euler_matrix(0, theta[i], 0)))
			vis["lines_segments"].set_transform(
				tf.translation_matrix([pos[i,0], 0, pos[i,1]]).dot(tf.euler_matrix(0, theta[i], 0)))				
			sleep(robot.dt)