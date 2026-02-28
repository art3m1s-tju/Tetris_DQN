import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        # 【关键修改】输入维度改为 29
        # 特征构成：
        # 5 (全局统计: lines, holes, bumpiness, total_height, max_height)
        # + 10 (每列高度)
        # + 7 (当前方块 One-Hot)
        # + 7 (下一个方块 One-Hot)
        # = 29
        self.conv1 = nn.Sequential(nn.Linear(29, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x