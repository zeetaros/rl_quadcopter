import numpy as np
from physics_sim import PhysicsSim

class Task():
    """
        Task (environment) that defines the goal and provides feedback to the agent.
        The simulator is initialized as an instance of the PhysicsSim class.

        For each timestep of the agent, we step the simulation action_repeats timesteps.

        We set the number of elements in the state vector. For the sample task,
        we only work with the 6-dimensional pose information. 
        To set the size of the state (state_size), we must take action repeats into account.
    """
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, orig_reward_func=False):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 

        # example: init_pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        # It is 6-dimensional (aka. 6 degree of freedom), which is location (x,y,z), 
        # and orientation in each of the axis, commonly known as Roll, Pitch & Yaw.

        self.action_repeat = 3 # wherein the action decided by the agent is repeated for 
                               # a fixed number of time steps regardless of the contextual 
                               # state while executing the task.

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        # by default the target is going back to where it starts

        self.coords_log = []
        self.speed_log = []
        self.init_pose = init_pose or np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        self.orig_reward_func = orig_reward_func

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        if self.orig_reward_func:
            reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        else:
            ## Assign extra penalty if the quadcopter cannot keep stable in the air
            ## The more it diverses from the current height the more penalty it gets
            if len(self.coords_log) > 1:
                # Difference between current and last position
                stability = abs(self.sim.pose[2] - self.coords_log[-1]['position'][2])
                
                # Further it is away from the target height larger the tolerance for unstability
                off_target_height_discount = np.exp(-abs(self.sim.pose[2] - self.target_pos[2]))
                stable_penalty = stability ** off_target_height_discount
            else:
                # because assume it start from infinitely far away from the target, off_target_height_discount will be close to ZERO
                stable_penalty = 1

            # Normalise the distance by dividing the current distance from the goal by the initial distance from the goal
            distance_penalty = abs(self.sim.pose[:3] - self.target_pos[:3]).sum() / abs(self.init_pose[:3] - self.target_pos[:3]).sum()

            velocity_penalty = min(self.sim.linear_accel[:2].sum(), 10) 

            reward = 1. - distance_penalty - .6 * stable_penalty - .5 * velocity_penalty
        return reward

    def step(self, rotor_speeds):
        """
            Uses action to obtain next state, reward, done.
            The parameter rotor_speeds is the action.
            The environment will always have a 4-dimensional action space, 
            with one entry for each rotor (action_size=4). 
            You can set the minimum (action_low) and maximum (action_high) values of each entry.
        """
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """
            Reset the sim to start a new episode.
            The agent should call this method every time the episode ends.
        """
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

    @property
    def current_coord(self):
        return {'position': self.sim.pose[:3],
                'reward': self.get_reward()}

    @property
    def best_coord(self):
        if self.coords_log != []:
            return max(self.coords_log, key=lambda c:c['reward'])

    def save_coord(self, i_episode=None):
        self.coords_log.append({'episode': i_episode, 'position': self.sim.pose[:3], 'reward': self.get_reward()})

    def save_speed(self, i_episode=None):
        self.speed_log.append({'episode': i_episode, 'velocity':self.sim.linear_accel})