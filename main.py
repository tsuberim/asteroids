import torch as t
from torch import nn
from torch import optim
import torch.nn.functional as F
import gym
from model import UNet, FullyConnectedNet
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np

C, H, W = 3, 210, 160


if t.cuda.is_available():
    device = t.device("cuda")
    print(f"Using CUDA device: {t.cuda.get_device_name(0)}")
elif t.backends.mps.is_available():
    device = t.device("mps")
    print("Using MPS device")
else:
    device = t.device("cpu")
    print("Using CPU")

run = wandb.init(project="asteroids_world_model")

def show(frame):
    # frame = frame.detach().permute(1, 2, 0).cpu().numpy()
    plt.imshow(frame)
    plt.title("Single Frame (Converted to HWC)")
    plt.axis("off")
    plt.show()

class Bank:
    def __init__(self, max_size=1024):
        self.max_size = max_size
        self.size = 0
        self.i = 0
        self.obs = t.zeros((max_size, C, H, W)).to(device)
        self.acts = t.zeros((max_size,1), dtype=int).to(device)
        self.rews = t.zeros((max_size,), dtype=t.float).to(device)

    def record(self, ob, act, rew):
        self.obs[self.i, :, : :] = ob
        self.acts[self.i, :] = act
        self.rews[self.i] = rew

        self.i = (self.i + 1) % self.max_size
        self.size += 1
        if self.size > self.max_size:
            self.size = self.max_size

    def frames_batch(self, batch_size):
        obs = self.obs[:self.size]
        if len(obs) < 5:
            return None, None, None, False
        acts = self.acts[:self.size]
        
        x = t.cat([obs[0:-4],obs[1:-3],obs[2:-2],obs[3:-1]], dim=1)
        a = t.cat([acts[0:-4],acts[1:-3],acts[2:-2],acts[3:-1]], dim=1)
        y = obs[4:]
        indices = t.randint(0, x.shape[0], (batch_size,))
        return x[indices], a[indices], y[indices], True

class Agent(nn.Module):
    def __init__(self, bank, max_prev_obs=4, num_actions=14, temp=1, lr=1e-4):
        super(Agent, self).__init__()
        self.unet = UNet()
        self.actor = FullyConnectedNet(168960, [1024, 1024, 512], num_actions)
        self.max_prev_obs = max_prev_obs
        self.frames = t.zeros(1, C * 4, H, W).to(device)
        self.actions = t.zeros((1, 4), dtype=int).to(device) # NOOP
        self.temp = temp
        self.bank = bank

        self.next_frame_pred_loss = nn.MSELoss()

    def forward(self, frames, actions):
        emb = self.unet(frames, actions, only_bottleneck=True)
        emb = emb.view(emb.shape[0], -1)
        act = t.softmax(self.actor(emb) / self.temp, dim=1)
        return act

    def play(self, ob, rew) -> int:
        ob = self.transform_ob(ob)
        act_dist = self.forward(self.frames, self.actions)
        act = t.multinomial(act_dist, num_samples=1).item()
        self.frames = t.roll(self.frames, shifts=-3, dims=1)
        self.frames[:, 9:, :, :] = ob
        self.actions = t.roll(self.actions, shifts=-3, dims=1)
        self.actions[:, 3:] = act
        self.bank.record(ob, act, rew)
        return act

    def transform_ob(self, ob):
        ob = t.from_numpy(ob).float().to(device) / 255.0
        ob = ob.permute(2,0,1).contiguous().view((1, C, H, W))
        return ob.detach()
    
    def train_world_model_step(self, batch_size):
        self.optim.zero_grad()
        frames, actions, next_frame, has = self.bank.frames_batch(batch_size)
        if not has:
            return t.zeros(1), t.zeros(1)
    
        pred = self.unet(frames, actions)
        loss = self.next_frame_pred_loss(pred, next_frame)

        loss.backward()
        grad_norm = t.norm(t.stack([t.norm(p.grad, 2) for p in self.parameters() if p.grad is not None]), 2)
        self.optim.step()

        pred_img = wandb.Image(pred[0].detach().permute(0, 2, 1).cpu().numpy().transpose(2, 1, 0))
        gt_img = wandb.Image(next_frame[0].detach().permute(0, 2, 1).cpu().numpy().transpose(2, 1, 0))
        prev_frame_1 = wandb.Image(frames[0, :3].detach().permute(0, 2, 1).cpu().numpy().transpose(2, 1, 0))
        prev_frame_2 = wandb.Image(frames[0, 3:6].detach().permute(0, 2, 1).cpu().numpy().transpose(2, 1, 0))
        prev_frame_3 = wandb.Image(frames[0, 6:9].detach().permute(0, 2, 1).cpu().numpy().transpose(2, 1, 0))
        prev_frame_4 = wandb.Image(frames[0, 9:].detach().permute(0, 2, 1).cpu().numpy().transpose(2, 1, 0))
        wandb.log({"prediction": [pred_img], "next_frame": [gt_img], "prev_frame_1": [prev_frame_1], "prev_frame_2": [prev_frame_2], "prev_frame_3": [prev_frame_3], "prev_frame_4": [prev_frame_4]})

        return loss, grad_norm
    
    def reset(self, ob):
        pass

    def load(self, filename="checkpoint.pth", artifact_name="world_model_checkpoint"):
        """Load model and optimizer state from a wandb artifact if it exists."""
        try:
            artifact = wandb.use_artifact(f"{artifact_name}:latest", type="model")
            artifact_dir = artifact.download()
            checkpoint_path = os.path.join(artifact_dir, filename)
            checkpoint = t.load(checkpoint_path, map_location=device)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded checkpoint from", checkpoint_path)
            return checkpoint
        except Exception as e:
            print("No checkpoint loaded; starting from scratch. Exception:", e)
            return None

    def save(self, loss, filename="checkpoint.pth", artifact_name="world_model_checkpoint"):
        """Save model and optimizer state to a file and log as a wandb artifact."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "loss": loss,
        }
        t.save(checkpoint, filename)
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(filename)
        wandb.log_artifact(artifact)
        print("Checkpoint saved as artifact:", artifact_name)
        

def main():
    # Set global device to MPS if available
    print(f"Using device: {device}")  # Check if MPS is active

    lr = 1e-4
    batch_size = 128
    bank = Bank()
    agent = Agent(bank)
    agent.optim = optim.Adam(agent.parameters(), lr=lr)
    agent = agent.to(device)
    # agent.load()

    env = gym.make('AsteroidsNoFrameskip-v4', obs_type="rgb")
    ob, info = env.reset(seed=42)
    prev = np.zeros_like(ob)
    rew = 0.0
    idx = 0
    while True:
        real_ob = ob + prev # atari env frames flickers between asteroids and player
        act = agent.play(real_ob, rew)
        prev = ob
        if idx == 0:
            prev = np.zeros_like(ob)
        ob, rew, done, truncated, info = env.step(act)
    
        if idx % batch_size == 0:
            for epoch in range(3):
                loss, grad_norm = agent.train_world_model_step(batch_size=batch_size)
                wandb.log({"loss": loss.item(), "grad_norm": grad_norm.item()})
            # agent.save(loss)

        if done or truncated:
            ob, info = env.reset()
            agent.reset(ob)

        idx += 1

if __name__ == '__main__':
    main()