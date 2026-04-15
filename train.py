import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env import RoboticArmEnv

def train():
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Environment with DIRECT mode 
    env = RoboticArmEnv(render_mode="direct")
    
    #  wraps the env and automatically exports rewards/lengths to a CSV file in log_dir
    env = Monitor(env, log_dir)
    
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Starting RL training loop (PPO)...")
    try:
        # Limited timesteps for quick demo, but enough to see the graph move
        model.learn(total_timesteps=50000)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        
    print("Saving model...")
    model.save("models/ppo_robotic_arm")
    env.close()

if __name__ == "__main__":
    train()
