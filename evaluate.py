import time
import os
from stable_baselines3 import PPO
from env import RoboticArmEnv

def evaluate():
    model_path = "models/ppo_robotic_arm"
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}. Please train first.")
        return
        
    env = RoboticArmEnv(render_mode="human")
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    try:
        for i in range(2000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # small delay for visualization realism
            time.sleep(1./120.) 
            
            if terminated or truncated:
                obs, _ = env.reset()
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("Demo stopped.")
        
    env.close()

if __name__ == "__main__":
    evaluate()
