import os
import sys

project_root = r"c:\Users\ASUS\Downloads\Adaptive-Distributed-Multi-Task-Energy-Consumption-Prediction-for-Smart-Buildings-main\Adaptive-Distributed-Multi-Task-Energy-Consumption-Prediction-for-Smart-Buildings-main\BDA_proj"
sys.path.insert(0, project_root)

from src.config.config_loader import ConfigLoader
from src.visualization.visualization_manager import VisualizationManager
from src.logging.logger_factory import LoggerFactory

def run_viz():
    loader = ConfigLoader(project_root)
    config = loader.config
    config["project_root"] = project_root
    
    log_cfg = config.get("logging", {})
    LoggerFactory.configure(log_cfg, project_root)

    manager = VisualizationManager(config)
    print("Generating visualizations...")
    manifest = manager.generate_all()
    print("Done.")

if __name__ == "__main__":
    run_viz()
