from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

train_process = None
demo_process = None
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RL_DIR = os.path.join(BASE_DIR, "rl_engine")
PYTHON_EXEC = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")

@app.post("/start-training")
def start_training():
    global train_process
    if train_process and train_process.poll() is None:
        return {"status": "Training already running"}
    
    train_process = subprocess.Popen([PYTHON_EXEC, "train.py"], cwd=RL_DIR)
    return {"status": "Training started"}

@app.post("/stop-training")
def stop_training():
    global train_process
    if train_process and train_process.poll() is None:
        train_process.kill() # Terminate process
        train_process = None
        return {"status": "Training stopped"}
    return {"status": "No active training process"}

@app.post("/run-demo")
def run_demo():
    global demo_process
    if demo_process and demo_process.poll() is None:
        return {"status": "Demo already running"}
        
    demo_process = subprocess.Popen([PYTHON_EXEC, "evaluate.py"], cwd=RL_DIR)
    return {"status": "Demo started"}

@app.get("/get-training-stats")
def get_training_stats():
    log_file = os.path.join(RL_DIR, "logs", "monitor.csv")
    is_training = train_process is not None and train_process.poll() is None
    
    if not os.path.exists(log_file):
        return {"data": [], "is_training": is_training, "status": "No log file found. Has training started?"}
        
    try:
        # SB3 Monitor CSV has a generic info header on line 1, column headers on line 2
        df = pd.read_csv(log_file, skiprows=1)
        if df.empty:
            return {"data": [], "is_training": is_training}
            
        # Calculate moving average to smooth the plot (window of 10 episodes)
        df['reward_ma'] = df['r'].rolling(window=10, min_periods=1).mean()
        df['length_ma'] = df['l'].rolling(window=10, min_periods=1).mean()
        
        # Downsample if too large (>500 points) to prevent browser lag
        if len(df) > 500:
            step = len(df) // 500
            sampled_df = df.iloc[::step].copy()
            # keep the very last row too
            if df.index[-1] not in sampled_df.index:
                 sampled_df = pd.concat([sampled_df, df.iloc[[-1]]])
            df = sampled_df
            
        data = []
        for index, row in df.iterrows():
            data.append({
                "episode": int(index),
                "reward": float(row["reward_ma"]),
                "length": float(row["length_ma"])
            })
            
        return {"data": data, "is_training": is_training}
    except Exception as e:
        return {"data": [], "is_training": is_training, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
