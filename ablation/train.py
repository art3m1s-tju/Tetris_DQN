import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=0.01) # 保留1%底噪防死循环
    parser.add_argument("--num_decay_epochs", type=float, default=48000)
    parser.add_argument("--num_epochs", type=int, default=60000)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--target_update_interval", type=int, default=1000) # 目标网络同步频率
    parser.add_argument("--replay_memory_size", type=int, default=30000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    os.makedirs(opt.saved_path, exist_ok=True)
        
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    
    # 1. 初始化主网络
    model = DeepQNetwork()
    
    # 2. 初始化目标网络，并复制主网络参数
    target_model = DeepQNetwork()
    target_model.load_state_dict(model.state_dict())
    target_model.eval() # 目标网络不参与反向传播

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    # 3. 学习率自动衰减：将总轮数分5次衰减，每次减半
    decay_step = max(1, opt.num_epochs // 5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.5)
    
    # 4. 损失函数换成 SmoothL1Loss
    criterion = nn.SmoothL1Loss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        target_model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0

    print("开始经验池预热，正在收集前 3000 步数据，请稍等几十秒...")

    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        # 关闭画面渲染加速
        reward, done = env.step(action, render=False)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue

        # ==========================================================
        # 这里的 1/10 就是 3000，必须凑够 3000 步才会往下走去训练和打印
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        # ==========================================================

        epoch += 1
        
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(s for s in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(s for s in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # 计算当前 Q 值
        q_values = model(state_batch)
        
        # 使用 目标网络(target_model) 评估下一个状态的 Q 值
        with torch.no_grad():
            next_prediction_batch = target_model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        
        # 梯度裁剪防爆
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch, opt.num_epochs, action, final_score, final_tetrominoes, final_cleared_lines))
        
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)
        writer.add_scalar('Train/Loss', loss.item(), epoch - 1)
        writer.add_scalar('Train/Epsilon', epsilon, epoch - 1)
        writer.add_scalar('Train/Replay memory size', len(replay_memory), epoch - 1)
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
            
        # 定期将主网络的权重同步到目标网络
        if epoch % opt.target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

    torch.save(model, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)