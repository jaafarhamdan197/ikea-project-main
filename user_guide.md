# Revenue Prediction Dashboard - User Guide

## Overview
This dashboard predicts revenue based on marketing budget inputs across different platforms and funnel levels. It visualizes diminishing returns and provides optimal budget distribution recommendations.

## Accessing the Dashboard
The dashboard is available at: http://8050-ixo6oz8mttldt66qsgu60-53b43614.manus.computer

## Dashboard Features

### Input Parameters
- **Total Budget**: Adjust the slider to set your total marketing budget (from $5,000 to $200,000)
- **Funnel Weights**: Modify the importance of each funnel level (Lower, Middle, Upper) using the sliders
- **Forecast Days**: Set the number of days for revenue forecasting (7-90 days)
- **Generate Prediction**: Click this button to update all visualizations based on your inputs

### Visualization Tabs

#### 1. Revenue Forecast
- Shows predicted daily revenue based on your budget allocation
- Displays total predicted revenue and overall ROAS (Return on Ad Spend)

#### 2. Diminishing Returns
- Visualizes how revenue increases with budget (diminishing returns curve)
- Shows marginal ROAS (return on each additional dollar spent)
- Identifies the optimal budget point where marginal ROAS equals 1

#### 3. Optimal Budget Allocation
- Displays recommended budget distribution across platforms and funnel levels
- Shows a stacked bar chart visualization and detailed allocation table
- Sorted by ROAS to highlight the most efficient channels

#### 4. Platform Comparison
- Compares performance across different platforms
- Bubble size and color indicate ROAS (larger/darker = better performance)

## How to Use the Dashboard

1. **Set Your Budget**: Adjust the total budget slider to your desired spending level
2. **Adjust Funnel Weights**: If you want to prioritize certain funnel levels, increase their weights
3. **Click "Generate Prediction"**: Update all visualizations with your new inputs
4. **Explore Different Tabs**: Review the various visualizations to understand:
   - Expected revenue over time
   - Diminishing returns and optimal spending point
   - Recommended budget allocation across platforms and funnel levels
   - Relative performance of different platforms

## Key Insights

- **Diminishing Returns**: The dashboard shows where additional spending yields diminishing returns
- **Optimal Allocation**: The model recommends how to distribute your budget for maximum revenue
- **Platform Efficiency**: Identifies which platforms deliver the highest ROAS
- **Funnel Strategy**: Allows you to adjust strategy by weighting different funnel levels

## Model Information
The prediction model is built using gradient-boosted trees with Hill functions to model diminishing returns. It's trained on your historical marketing data across multiple platforms and funnel levels.

## Technical Details
- The model achieves high accuracy through ensemble learning
- Diminishing returns are modeled using Hill functions
- Budget optimization uses constrained optimization techniques
- The dashboard is built with Dash and Plotly for interactive visualizations
