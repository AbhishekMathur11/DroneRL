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
from math_utils import *
from scipy.spatial.transform import Rotation
import argparse

class Quadrotor():
	def __init__(self):
		# parameters
		self.m = 0.027 # kg
		self.J = np.diag([8.571710e-5, 8.655602e-5, 15.261652e-5]) # inertia matrix
		self.J_inv = np.linalg.inv(self.J)
		self.arm = 0.0325 # arm length
		self.t2t = 0.006 # thrust to torque ratio
		self.g = 9.81 # gravity

		# control actuation matrix
		self.B = np.array([[1., 1., 1., 1.],
			               [-self.arm, -self.arm, self.arm, self.arm],
			               [-self.arm, self.arm, self.arm, -self.arm],
			               [-self.t2t, self.t2t, -self.t2t, self.t2t]])
		self.B_inv = np.linalg.inv(self.B)
		
		# noise level
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		
		# disturbance and its estimation
		self.d = np.array([0., 0, 0])
		self.d_hat = np.array([0., 0, 0])

		# initial state
		self.p = np.array([0., 0, 0])
		self.v = np.array([0., 0, 0])
		self.R = np.eye(3)
		self.q = np.array([1., 0, 0, 0])
		self.omega = np.array([0., 0, 0])
		self.euler_rpy = np.array([0., 0, 0])

		# self.phi = np.arctan2(2*(self.q[0]*self.q[1] + self.q[2]*self.q[3]), 1 - 2*(self.q[1]**2 + self.q[2]**2))
		# self.theta = np.arctan2(2*(self.q[0]*self.q[2] - self.q[1]*self.q[3]))
		# self.psi = np.arctan2(2*(self.q[0]*self.q[3] + self.q[2]*self.q[1]), 1 - 2*(self.q[2]**2 + self.q[3]**2))

		# initial control (hovering)
		self.u = np.array([1, 1, 1, 1]) * self.m * self.g / 4.

		# control limit for each rotor (N)
		self.umin = 0.
		self.umax = 0.012 * self.g

		# total time and discretizaiton step
		self.dt = 0.01
		self.step = 0
		self.t = 0.

	def reset(self):
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		self.d = np.array([0., 0, 0])
		self.p = np.array([0., 0, 0])
		self.v = np.array([0., 0, 0])
		self.R = np.eye(3)
		self.q = np.array([1., 0, 0, 0])
		self.omega = np.array([0., 0, 0])
		self.euler_rpy = np.array([0., 0, 0])
		self.u = np.array([1, 1, 1, 1]) * self.m * self.g / 4.
		self.step = 0
		self.t = 0.

	def dynamics(self, u):
		'''
		Problem B-1: Based on lecture 2, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		self.u is the control input (four rotor forces).
		Hint: first convert self.u to total thrust and torque using the control actuation matrix.
		Hint: use the qintegrate function to update self.q
		'''
		u = np.clip(u, self.umin, self.umax)
		self.u = np.clip(u, self.umin, self.umax)
		# Rot_mat = np.array([np.cos(self.phi)*np.cos(self.theta) - np.sin(self.phi)*np.sin(self.psi)*np.sin(self.theta), -np.cos(self.phi)*np.sin(self.psi), np.cos(self.psi)*np.sin(self.theta) + np.cos()])

	

		pdot = self.v
		thrust = np.dot(self.B, self.u)[0]
		vdot = np.array([0, 0, -self.g]) + (self.R @ np.array([0, 0, thrust / self.m]))	
		torques = np.dot(self.B, self.u)[1:]
		omegadot = self.J_inv @ (torques - np.cross(self.omega, self.J @ self.omega))
		self.p += self.dt * pdot
		self.v += self.dt * vdot + self.dt * (self.sigma_t * np.random.normal(size=3) + self.d) 
		self.q = qintegrate(self.q, self.omega, self.dt)
		self.R = qtoR(self.q)
		self.omega += self.dt * omegadot + self.dt * self.sigma_r * np.random.normal(size=3)
		self.euler_rpy = Rotation.from_matrix(self.R).as_euler('xyz')

		self.t += self.dt
		self.step += 1

	def cascaded_control(self, p_d, v_d, a_d, yaw_d):
		'''
		Problem B-2: Based on lecture 3, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		Your goal is to develop a cascaded controller to track a trajectory (p_d, v_d, a_d, yaw_d).
		Hint for gain tuning: position control gain is smaller (1-10);
		Attitude control gain is bigger (10-200).
		'''
		# position control
		K_P = 1
		K_D =  8

		KP_tau = 180
		KD_tau = 20

		g_vec = np.array([0, 0, -self.g])

		f_d = -g_vec - K_P*(self.p - p_d) - K_D*(self.v - v_d) + a_d
		e3 = np.array([0, 0, 1])

		z = self.R@e3.T
		T = (np.dot(z,f_d))*self.m
		# print(T)
		z_d = f_d/np.linalg.norm(f_d)
		n = np.cross(e3, z_d)
		rho = np.arcsin(np.linalg.norm(n))
		n_norm = np.linalg.norm(n)

		if n_norm != 0:
			n_normalized = n/n_norm
		else:
			n_normalized = 0
	    
		rot_vector = n_normalized*rho
		R_EB = Rotation.from_rotvec(rot_vector).as_matrix()
		
		R_AE = Rotation.from_euler('z', yaw_d, degrees=False).as_matrix()

		R_d = R_AE@R_EB
		R_e = R_d.T@self.R
		alpha_d = np.array([0, 0, 0])
		alpha = -KP_tau*vee(R_e - R_e.T) - KD_tau*self.omega + alpha_d
		tau = self.J @ alpha - np.cross(self.J @ self.omega, self.omega)
		# print(tau)

		control_mat = np.hstack([T, tau])
		
		u = self.B_inv@ control_mat



		return u

	def adaptive_control(self, p_d, v_d, a_d, yaw_d):
		'''
		Problem B-3: Based on lecture 4, implement adaptive control methods.
		For integral control, this function should be same as cascaded_control, 
		with an extra I gain in the position control loop.
		Hint for integral control: you can use self.d_hat to accumlate/integrate the position error.
		'''
		K_P = 5.5
		K_D = 4.5
		K_I = 3
		
		KP_tau = 80
		KD_tau = 10
		  
		omega_c = 20  

		g_vec = np.array([0, 0, -self.g])

	
		e_p = self.p - p_d
		e_v = self.v - v_d
		self.d_hat += K_I * e_p * self.dt  
		
		self.d_hat = self.d_hat + self.dt * omega_c * (np.clip(self.d_hat, -1, 1) - self.d_hat)

		
		f_d = -g_vec - K_P * e_p - K_D * e_v - self.d_hat + a_d

		e3 = np.array([0, 0, 1])
		z = self.R @ e3
		T = (np.dot(z,f_d))*self.m
		z_d = f_d / np.linalg.norm(f_d)
		n = np.cross(e3, z_d)
		rho = np.arcsin(np.linalg.norm(n))
		n_norm = np.linalg.norm(n)

		if n_norm != 0:
			n_normalized = n / n_norm
		else:
			n_normalized = np.zeros(3)
		
		rot_vector = n_normalized * rho
		R_EB = Rotation.from_rotvec(rot_vector).as_matrix()
		R_AE = Rotation.from_euler('z', yaw_d, degrees=False).as_matrix()

		R_d = R_AE @ R_EB
		R_e = R_d.T @ self.R
		alpha_d = np.array([0, 0, 0])
		alpha = -KP_tau*vee(R_e - R_e.T) - KD_tau*self.omega + alpha_d
		tau = self.J @ alpha - np.cross(self.J @ self.omega, self.omega)
		# print(tau)

		control_mat = np.hstack([T, tau])
		
		u = self.B_inv@ control_mat

		
		
		return u


def plot(time, pos, vel, control, euler_rpy, omega, pos_des):
	plt.figure(figsize=(20, 4))
	plt.subplot(1, 5, 1)
	colors = ['tab:blue', 'tab:orange', 'tab:green']
	names = ['x', 'y', 'z']
	for i in range(3):
		plt.plot(time, pos[:,i], color=colors[i], label=names[i]+" actual")
		plt.plot(time, pos_des[:,i], '--', color=colors[i], label=names[i]+" desired")
	plt.xlabel("time (s)")
	plt.ylabel("pos (m)")
	plt.legend()
	plt.subplot(1, 5, 2)
	plt.plot(time, vel)
	plt.xlabel("time (s)")
	plt.ylabel("vel (m/s)")
	plt.legend(["x", "y", "z"])
	plt.subplot(1, 5, 3)
	plt.plot(time, control)
	plt.xlabel("time (s)")
	plt.ylabel("control (N)")
	plt.legend(["1", "2", "3", "4"])
	plt.subplot(1, 5, 4)
	plt.plot(time, euler_rpy)
	plt.xlabel("time (s)")
	plt.legend(["roll (rad)", "pitch (rad)", "yaw (rad)"])
	plt.subplot(1, 5, 5)
	plt.plot(time, omega)
	plt.xlabel("time (s)")
	plt.ylabel("angular rate (rad/s)")
	plt.legend(["x", "y", "z"])
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	robot = Quadrotor()
	total_time = 3 * np.pi
	total_step = int(total_time/robot.dt+1)
	time = np.linspace(0, total_time, total_step)
	pos = np.zeros((total_step, 3))
	pos_des = np.zeros((total_step, 3))
	vel = np.zeros((total_step, 3))
	control = np.zeros((total_step, 4))
	control[0, :] = robot.u
	quat = np.zeros((total_step, 4))
	quat[0, :] = robot.q
	euler_rpy = np.zeros((total_step, 3))
	omega = np.zeros((total_step, 3))

	parser = argparse.ArgumentParser()
	parser.add_argument('question', type=int)
	question = parser.parse_args().question

	'''
	Problem B-1: system modeling
	'''
	if question == 1:
		robot.sigma_r = 0.
		robot.sigma_t = 0.
		for i in range(21):
			u = np.array([0.006, 0.008, 0.010, 0.012]) * 9.81
			robot.dynamics(u)
			if i % 10 == 0:
				print('************************')
				print('pos:', robot.p)
				print('vel:', robot.v)
				print('quaternion:', robot.q)
				print('omega:', robot.omega)

	'''
	Problem B-2: cascaded tracking control
	'''
	rise_time_calculated_2 = False
	t_10_2 = None
	t_90_2 = None
	max_overshoot2 = 0
	robot.reset()
	while True:
		if question != 2 or robot.step >= total_step-1:
			break
		t = robot.t
		p_d = np.array([1, 1, 1])
		v_d = np.zeros(3)
		a_d = np.zeros(3)
		# yaw_d = 0
		yaw_d = (t/total_time)*(np.pi/3)
		u = robot.cascaded_control(p_d, v_d, a_d, yaw_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		quat[robot.step] = robot.q
		euler_rpy[robot.step] = robot.euler_rpy
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
	if question == 2:
		plot(time, pos, vel, control, euler_rpy, omega, pos_des)
		ce2 = np.mean(np.linalg.norm(control)**2)
		mse2 = np.mean(np.square(np.subtract(pos_des, pos)))
		rmse2= np.sqrt(mse2)
		print(f"Average Control Energy = {ce2}")
		print(f"RMSE is equal to: {rmse2}")


	'''
	Problem B-3: integral and adaptive control
	'''
	rise_time_calculated_3 = False
	t_10_3 = None
	t_90_3 = None
	max_overshoot3 = 0
	robot.reset()
	while True:
		if question != 3 or robot.step >= total_step-1:
			break
		t = robot.t
		robot.d = np.array([0.5, np.sin(t), np.cos(np.sqrt(2)*t)])
		# p_d = np.array([np.sin(2*t), np.cos(2*t - 1), 0.5*t])
		# v_d = np.array([2*np.cos(2*t), -2*np.sin(2*t - 1), 0.5])
		# a_d = np.array([-4*np.sin(2*t), -4*np.cos(2*t - 1), 0])
		p_d = np.array([1, 1, 1])
		v_d = np.zeros(3)
		a_d = np.zeros(3)
		yaw_d = 0.
		u = robot.adaptive_control(p_d, v_d, a_d, yaw_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		quat[robot.step] = robot.q
		euler_rpy[robot.step] = robot.euler_rpy
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
		plot(time, pos, vel, control, euler_rpy, omega, pos_des)
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
				tf.translation_matrix(pos[i]).dot(tf.quaternion_matrix(quat[i])))
			vis["lines_segments"].set_transform(
				tf.translation_matrix(pos[i]).dot(tf.quaternion_matrix(quat[i])))				
			sleep(robot.dt)