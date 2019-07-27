import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetworkPixels(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetworkPixels, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(1, 8, 8), stride=(1, 4, 4))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(4, 4, 4), stride=(1, 2, 2))
        conv2_output_size = self._get_conv_output_size(state_size)
        fc = [conv2_output_size, 256]
        self.fc1 = nn.Linear(fc[0], fc[1])
        self.fc2 = nn.Linear(fc[1], action_size)

    def forward(self, state):
        x = self._cnn(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_conv_output_size(self, shape):
        x = torch.rand(shape)
        x = self._cnn(x)
        n_size = x.data.view(1, -1).size(1)
        return n_size

    def _cnn(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return x
