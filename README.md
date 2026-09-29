# Adaptive Distributed Multi-Task Energy Consumption Prediction for Smart Buildings

This repository contains the codebase for an advanced, distributed machine learning pipeline designed to predict energy consumption across multiple smart buildings adaptively. The system leverages multi-task learning paradigms to forecast power demand and analyze associated carbon emissions, providing a scalable solution for smart grid and building energy management.

## Project Overview

The project is built to handle the complexities of energy consumption forecasting in modern smart building environments. By utilizing a multi-task learning approach, the models can identify and exploit shared patterns across different buildings or zones while adapting to shifting consumption behaviors over time.

### Key Capabilities

- **Distributed Data Processing:** Supports ingestion and processing of large-scale building data with configurations suited for distributed environments (e.g., Hadoop).
- **Multi-Task Learning:** Simultaneously predicts energy usage for multiple buildings to capture both shared and building-specific patterns.
- **Adaptive Forecasting:** Dynamically adjusts to changing energy consumption behaviors.
- **Carbon Emission Analysis:** Calculates and visualizes the carbon footprint associated with energy usage to support sustainability goals.
- **Automated Reporting:** Generates comprehensive visualization artefacts and structured HTML experiment reports.

## Project Structure

The primary codebase is located in the `BDA_proj` directory, structured as follows:

```text
BDA_proj/
├── config/              # Configuration files (e.g., app_config.yaml, hadoop_config.yaml)
├── data/                # Raw and processed datasets
├── hdfs_local/          # Local Hadoop distributed file system components
├── logs/                # System and execution logs
├── models/              # Saved predictive models and checkpoints
├── reports/             # Generated HTML and markdown reports
├── results/             # Prediction outputs and metrics
├── src/                 # Core source code
│   ├── ingestion/       # Data loading and preprocessing pipelines
│   ├── features/        # Feature engineering modules
│   ├── models/          # Model training, pipelines, and managers
│   └── tasks/           # Multi-task learning abstractions and definitions
├── visualizations/      # Generated plots and visual artefacts
├── main.py              # Main entry point for training and pipeline execution
├── generate_future_predictions.py # Script for generating future energy forecasts
├── run_viz.py           # Pipeline script for generating reports and visual artefacts
└── plot_carbon_emissions.py       # Specific script for analyzing and plotting carbon footprints
```

## System Components

- **Configuration Management:** The `config/` folder contains YAML files (`app_config.yaml` and `hadoop_config.yaml`) that define both application-level parameters and distributed system settings.
- **Source Code (`src/`):** Houses the core logic, including robust data ingestion, feature engineering, and the machine learning model pipelines.
- **Execution Scripts:** 
  - `main.py` drives the end-to-end training process.
  - `generate_future_predictions.py` utilizes trained models to forecast future consumption.
  - `run_viz.py` and `plot_carbon_emissions.py` handle the creation of detailed visualizations and environmental impact metrics.