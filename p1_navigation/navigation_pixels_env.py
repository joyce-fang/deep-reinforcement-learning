import numpy as np

class BananaPixelEnv():
    def __init__(self, env, num_frames=4):
        self.frame_buffer = []
        self.brain_names = env.brain_names
        self.env = env
        self.num_frames = num_frames

    def _update_state(self):
        frame = np.transpose(self.env_info.visual_observations[0], (0, 3, 1, 2))[:, :, :, :]
        frame_size = frame.shape
        self.state = np.zeros((1, frame_size[1], self.num_frames, frame_size[2], frame_size[3]))
        self.frame_buffer.insert(0, frame)
        if len(self.frame_buffer) > 4:
            self.frame_buffer.pop()

        for i, f in enumerate(self.frame_buffer):
            self.state[0, :, i ,:, :] = f

    def reset(self):
        self.env_info = self.env.reset(train_mode=True)[self.brain_names[0]]
        self._update_state()
        return self.state

    def step(self, action):
        self.env_info = self.env.step(np.int32(action).astype(np.int32))[self.brain_names[0]]
        self._update_state()
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        return self.state, reward, done, None



