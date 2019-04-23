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
        init_angle_velocities=None, runtime=5., target_pos=None, reward_func=None, early_stop=False):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            reward_func: apply different reward function depends on the task;
                        * "original": use the most basic reward function
                        * "takeoff": for task to teach the quadcopter to take off. Give a generous reward once its position(axis:z) is higher than the target
                        * "hover": for task to keep the quadcopter in the air. Punish it hard for hitting the ground(z=0)
                        * "land": for task to teach the quadcopter land. Reward very low speed when it is close to the ground(z=0)
                        * "target": for task to send the quadcopter from one point to another in a 3D space
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
        self.init_pose = init_pose if init_pose is not None else np.array([0., 0., 10., 0., 0., 0.])
        self.reward_func = reward_func if reward_func is not None else "target"
        self.early_stop = early_stop

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        if self.reward_func == "original":
            reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        elif self.reward_func == "takeoff":
            pass #TODO
        elif self.reward_func == "hover":
            pass #TODO
        elif self.reward_func == "land":
            pass #TODO

        elif self.reward_func == "target":
            ## Assign extra penalty if the quadcopter cannot keep stable in the air
            ## The more it diverses from the current height the more penalty it gets
            if len(self.coords_log) > 1:
                # Difference between current and last position
                stability = abs(self.sim.pose[2] - self.coords_log[-1]['position'][2])
                
                # Further it is away from the target height larger the tolerance for unstability
                off_target_height_discount = np.exp(-abs(self.sim.pose[2] - self.target_pos[2]))
                stable_penalty = stability ** off_target_height_discount
            else:
                # Because assume it start from infinitely far away from the target, off_target_height_discount will be close to ZERO
                stable_penalty = 1

            # Normalise the distance by dividing the current distance from the goal by the initial distance from the goal
            # Adding 0.1 to denominator to avoid division by zero
            distance_penalty = abs(self.sim.pose[:3] - self.target_pos[:3]).sum() / (abs(self.init_pose[:3] - self.target_pos[:3]).sum()  + 0.1)

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

    def early_stopping(self, current_episode, threshold=None):
        """ 
            Check if the quadcopter has reached the target in the last few episode.
            If so, stop the trainning before it forgets how to succeed!
            Params
            ======
                threshold: the distance away from the target that you prepare to accept for terminating the training
        """
        if self.early_stop:
            threshold = threshold or 0
            if current_episode <= 11:
                return False

            for episode in range(current_episode-9, current_episode+1):
                successes = 0
                final_pose = [c['position'] for c in self.coords_log if c['episode']==episode][-1]
                if abs(final_pose[:3] - self.target_pos[:3]).sum() <=  threshold:
                    successes += 1

            return True if successes >= 7 else False

        if not self.early_stop:     # if early_stop == False
            return False

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