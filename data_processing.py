import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Data extraction and cleaning
def extract_and_clean_data(file_path):
    """
    Extract data from Excel file and perform initial cleaning
    """
    # Read the data
    df = pd.read_excel(file_path)
    
    # Basic cleaning
    # Check for negative values in budget and revenue
    df['budget'] = df['budget'].clip(lower=0)
    df['revenue'] = df['revenue'].clip(lower=0)
    
    # Handle extreme outliers (clip to P1/P99)
    for col in ['impressions', 'clicks', 'purchases', 'budget', 'revenue']:
        if col in df.columns:
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Standardize platform names (noticed 'Tiktok' and 'TikTok' in the data)
    df['platform'] = df['platform'].replace('Tiktok', 'TikTok')
    
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

# Feature engineering
def engineer_features(df):
    """
    Create additional features for modeling
    """
    # Create copy to avoid modifying original
    df_features = df.copy()
    
    # Create day of week, month, quarter features
    df_features['day_of_week'] = df_features['date'].dt.dayofweek
    df_features['month'] = df_features['date'].dt.month
    df_features['quarter'] = df_features['date'].dt.quarter
    df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
    
    # Create non-linear transforms as mentioned in the instructions
    df_features['log_budget'] = np.log1p(df_features['budget'])
    df_features['sqrt_impressions'] = np.sqrt(df_features['impressions'])
    
    # Create platform-funnel interaction dummies
    df_features['platform_funnel'] = df_features['platform'] + '_' + df_features['communication_type']
    
    # Calculate efficiency metrics
    df_features['ctr'] = np.where(df_features['impressions'] > 0, 
                                 df_features['clicks'] / df_features['impressions'], 
                                 0)
    df_features['cvr'] = np.where(df_features['clicks'] > 0, 
                                 df_features['purchases'] / df_features['clicks'], 
                                 0)
    df_features['cpa'] = np.where(df_features['purchases'] > 0, 
                                 df_features['budget'] / df_features['purchases'], 
                                 0)
    df_features['roas'] = np.where(df_features['budget'] > 0, 
                                  df_features['revenue'] / df_features['budget'], 
                                  0)
    
    # Create lagged variables (7-day lag for each platform-funnel combination)
    for lag in [1, 3, 7]:
        for metric in ['budget', 'revenue', 'impressions', 'clicks', 'purchases']:
            df_features[f'{metric}_lag_{lag}'] = df_features.groupby('platform_funnel')[metric].shift(lag)
    
    # Create adstock variables for budget (especially important for upper funnel)
    # We'll create multiple lambda values and select the best during modeling
    for lambda_val in [0.3, 0.5, 0.7]:
        df_features[f'adstock_{lambda_val}'] = 0
        
        # Calculate adstock for each platform-funnel combination
        for platform_funnel in df_features['platform_funnel'].unique():
            mask = df_features['platform_funnel'] == platform_funnel
            temp_df = df_features[mask].copy()
            
            # Initialize adstock
            adstock = 0
            adstock_values = []
            
            # Calculate adstock recursively
            for budget in temp_df['budget']:
                adstock = budget + lambda_val * adstock
                adstock_values.append(adstock)
            
            # Assign values back to the dataframe
            df_features.loc[mask, f'adstock_{lambda_val}'] = adstock_values
    
    # Drop rows with NaN values created by lagging
    df_features = df_features.dropna()
    
    return df_features

# Split data for training and testing
def split_data(df, test_size=0.2):
    """
    Split data into training and testing sets based on time
    """
    # Sort by date
    df = df.sort_values('date')
    
    # Determine split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df, test_df

# Main function to process data
def process_data(input_file, output_dir):
    """
    Main function to process data and save results
    """
    # Extract and clean data
    df = extract_and_clean_data(input_file)
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Split data
    train_df, test_df = split_data(df_features)
    
    # Save processed data
    df.to_csv(os.path.join(output_dir, 'cleaned_data.csv'), index=False)
    df_features.to_csv(os.path.join(output_dir, 'featured_data.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    
    # Generate and save basic statistics
    stats = {
        'total_rows': len(df),
        'date_range': [df['date'].min(), df['date'].max()],
        'platforms': df['platform'].unique().tolist(),
        'funnel_types': df['communication_type'].unique().tolist(),
        'total_budget': df['budget'].sum(),
        'total_revenue': df['revenue'].sum(),
        'avg_roas': df['revenue'].sum() / df['budget'].sum() if df['budget'].sum() > 0 else 0
    }
    
    # Save stats as text file
    with open(os.path.join(output_dir, 'data_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # Generate some exploratory plots
    # 1. Budget vs Revenue by Platform
    plt.figure(figsize=(12, 8))
    for platform in df['platform'].unique():
        platform_data = df[df['platform'] == platform]
        plt.scatter(platform_data['budget'], platform_data['revenue'], alpha=0.5, label=platform)
    
    plt.xlabel('Budget')
    plt.ylabel('Revenue')
    plt.title('Budget vs Revenue by Platform')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'budget_vs_revenue_by_platform.png'))
    
    # 2. Budget vs Revenue by Funnel Type
    plt.figure(figsize=(12, 8))
    for funnel in df['communication_type'].unique():
        funnel_data = df[df['communication_type'] == funnel]
        plt.scatter(funnel_data['budget'], funnel_data['revenue'], alpha=0.5, label=funnel)
    
    plt.xlabel('Budget')
    plt.ylabel('Revenue')
    plt.title('Budget vs Revenue by Funnel Type')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'budget_vs_revenue_by_funnel.png'))
    
    return df_features

if __name__ == "__main__":
    input_file = '/home/ubuntu/upload/IKEA_historical_data 2 (1).xlsx'
    output_dir = '/home/ubuntu/dashboard_project/data'
    
    processed_data = process_data(input_file, output_dir)
    print(f"Data processing complete. Files saved to {output_dir}")
