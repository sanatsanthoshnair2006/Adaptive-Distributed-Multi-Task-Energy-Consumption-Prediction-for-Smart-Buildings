import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_emissions():
    project_root = os.path.dirname(os.path.abspath(__file__))
    monthly_path = os.path.join(project_root, "hdfs_local", "user", "energy_prediction", "energy_estimates", "monthly")
    
    if not os.path.exists(monthly_path):
        print(f"Path does not exist: {monthly_path}")
        print("Please run the pipeline first using 'python main.py'")
        return

    try:
        # Load the parquet file
        df = pd.read_parquet(monthly_path)
    except Exception as e:
        print(f"Failed to read parquet data: {e}")
        return

    # Check if our new column exists
    if "Carbon_Emission_kgCO2" not in df.columns:
        print("The column 'Carbon_Emission_kgCO2' was not found in the data.")
        print("This means the pipeline has not been run since the carbon emission calculation was added.")
        print("Please run 'python main.py' to generate the updated data.")
        return

    # Process data for plotting
    # Assuming Monthly data has 'Year' and 'Month'
    if "Year" in df.columns and "Month" in df.columns:
        # Create a datetime column for plotting
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
        df = df.sort_values('Date')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['Carbon_Emission_kgCO2'], marker='o', linestyle='-', color='r', linewidth=2)
        plt.title('Monthly Carbon Emissions Over Time')
        plt.xlabel('Date')
        plt.ylabel('Carbon Emissions (kg CO2)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join(project_root, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "monthly_carbon_emissions.png")
        plt.savefig(output_path)
        print(f"Successfully generated monthly graph: {output_path}")
        
        # Show the plot if in an interactive environment
        try:
            plt.show()
        except:
            pass
    else:
        print("Expected 'Year' and 'Month' columns for plotting monthly data.")

def plot_seasonal_emissions():
    project_root = os.path.dirname(os.path.abspath(__file__))
    seasonal_path = os.path.join(project_root, "hdfs_local", "user", "energy_prediction", "energy_estimates", "seasonal")
    
    if not os.path.exists(seasonal_path):
        print(f"Seasonal path does not exist: {seasonal_path}")
        print("Please run 'python main.py' to generate the seasonal data.")
        return

    try:
        df = pd.read_parquet(seasonal_path)
    except Exception as e:
        print(f"Failed to read seasonal parquet data: {e}")
        return

    if "Carbon_Emission_kgCO2" not in df.columns:
        print("The column 'Carbon_Emission_kgCO2' was not found in the seasonal data.")
        return

    if "Year" in df.columns and "Season" in df.columns:
        # Create a categorical order for seasons
        season_order = ["Spring", "Summer", "Fall", "Winter"]
        df['Season'] = pd.Categorical(df['Season'], categories=season_order, ordered=True)
        df = df.sort_values(['Year', 'Season'])
        
        # Create a combined label for plotting (e.g., "2008 Spring")
        df['Year_Season'] = df['Year'].astype(str) + " " + df['Season'].astype(str)
        
        plt.figure(figsize=(12, 6))
        # Use a bar chart to show seasonal emissions
        bars = plt.bar(df['Year_Season'], df['Carbon_Emission_kgCO2'], color='skyblue', edgecolor='black')
        
        # Color bars based on season for better visual distinction
        color_map = {"Spring": "#77DD77", "Summer": "#FFB347", "Fall": "#FF6961", "Winter": "#AEC6CF"}
        for bar, season in zip(bars, df['Season']):
            bar.set_color(color_map.get(season, 'grey'))
            bar.set_edgecolor('black')

        plt.title('Seasonal Carbon Emissions')
        plt.xlabel('Year & Season')
        plt.ylabel('Carbon Emissions (kg CO2)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_dir = os.path.join(project_root, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "seasonal_carbon_emissions.png")
        plt.savefig(output_path)
        print(f"Successfully generated seasonal graph: {output_path}")
        
        try:
            plt.show()
        except:
            pass
    else:
        print("Expected 'Year' and 'Season' columns for plotting seasonal data.")

def plot_time_of_day_profile():
    project_root = os.path.dirname(os.path.abspath(__file__))
    hourly_path = os.path.join(project_root, "hdfs_local", "user", "energy_prediction", "energy_estimates", "hourly")
    
    if not os.path.exists(hourly_path):
        print(f"Hourly path does not exist: {hourly_path}")
        print("Please run 'python main.py' to generate the hourly data.")
        return

    try:
        df = pd.read_parquet(hourly_path)
    except Exception as e:
        print(f"Failed to read hourly parquet data: {e}")
        return

    if "Energy_Global_active_power" not in df.columns or "Window_Start" not in df.columns:
        print("Required columns missing for time-of-day profile.")
        return

    # Extract the hour (0-23) from the Window_Start timestamp
    df['HourOfDay'] = pd.to_datetime(df['Window_Start']).dt.hour
    
    # Calculate average energy consumption for each hour across all days
    hourly_avg = df.groupby('HourOfDay')['Energy_Global_active_power'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_avg['HourOfDay'], hourly_avg['Energy_Global_active_power'], 
             marker='o', linestyle='-', color='indigo', linewidth=2.5)
    
    # Fill under the curve for aesthetics
    plt.fill_between(hourly_avg['HourOfDay'], hourly_avg['Energy_Global_active_power'], color='indigo', alpha=0.1)

    plt.title('Average Energy Consumption by Time of Day')
    plt.xlabel('Hour of Day (0 - 23)')
    plt.ylabel('Average Energy Consumption (kWh)')
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_dir = os.path.join(project_root, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "time_of_day_energy_profile.png")
    plt.savefig(output_path)
    print(f"Successfully generated time-of-day profile graph: {output_path}")
    
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    plot_emissions()
    plot_seasonal_emissions()
    plot_time_of_day_profile()
