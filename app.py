import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State
from scipy.optimize import minimize

# Initialize Dash app
app = Dash(__name__)

# Global variables
data_loaded = False
model_loaded = False
curves_loaded = False
df = None
curve_params = {}

# Load data
try:
    df = pd.read_excel(r'C:\Users\dell\Downloads\IkeaCodes_Google\IKEA_RAW_DATA.xlsx')
    data_loaded = True
except Exception as e:
    print(f"Error loading data: {e}")

# Define helper functions
def hill_function(x, V_max, k, n):
    """Hill function for modeling diminishing returns"""
    # Add small epsilon to avoid division by zero
    x = np.maximum(x, 1e-10)
    return (V_max * x**n) / (k**n + x**n)

def optimize_budget(total_budget, curve_params, funnel_weights):
    """Optimize budget allocation across platforms and funnel levels"""
    if not curve_params or total_budget <= 0:
        return pd.DataFrame()
    
    # Get all platform-funnel combinations
    platform_funnels = []
    for key in curve_params.keys():
        if '_' in key:
            platform, funnel = key.split('_')
            platform_funnels.append((platform, funnel))
    
    if not platform_funnels:
        return pd.DataFrame()
    
    # Initial equal allocation
    n_combinations = len(platform_funnels)
    initial_allocation = [total_budget / n_combinations] * n_combinations
    
    # Bounds for optimization (each allocation between 0 and total budget)
    bounds = [(0, total_budget) for _ in range(n_combinations)]
    
    # Constraint: sum of allocations equals total budget
    constraint = {'type': 'eq', 'fun': lambda x: sum(x) - total_budget}
    
    # Objective function: maximize total revenue with funnel weights
    def objective(allocations):
        total_revenue = 0
        for i, (platform, funnel) in enumerate(platform_funnels):
            pf = f"{platform}_{funnel}"
            if pf in curve_params:
                params = curve_params[pf]
                revenue = hill_function(allocations[i], params['V_max'], params['k'], params['n'])
                # Apply funnel weight
                weighted_revenue = revenue * funnel_weights.get(funnel, 1.0)
                total_revenue += weighted_revenue
        # Return negative revenue for minimization
        return -total_revenue
    
    # Perform optimization
    try:
        result = minimize(objective, initial_allocation, bounds=bounds, constraints=constraint, method='SLSQP')
        optimal_allocations = result.x
    except Exception as e:
        print(f"Optimization error: {e}")
        return pd.DataFrame()
    
    # Create results dataframe
    results = []
    for i, (platform, funnel) in enumerate(platform_funnels):
        pf = f"{platform}_{funnel}"
        if pf in curve_params:
            params = curve_params[pf]
            budget = optimal_allocations[i]
            revenue = hill_function(budget, params['V_max'], params['k'], params['n'])
            roas = revenue / budget if budget > 0 else 0
            
            results.append({
                'Platform': platform,
                'Funnel': funnel,
                'Optimal_Budget': budget,
                'Expected_Revenue': revenue,
                'ROAS': roas
            })
    
    # Convert to dataframe and sort by ROAS
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('ROAS', ascending=False)
    
    return results_df

def generate_daily_forecast(budget, days, curve_params, optimal_allocation):
    """Generate daily revenue forecast based on optimal allocation"""
    # Create date range
    dates = pd.date_range(start='today', periods=days)
    
    # Initialize forecast dataframe
    forecast = pd.DataFrame({
        'Day': range(1, days + 1),
        'Date': dates,
        'Budget': budget / days,  # Distribute budget equally across days
        'Revenue': 0
    })
    
    # Calculate daily revenue based on optimal allocation
    if len(optimal_allocation) > 0:
        # Calculate total expected revenue
        total_revenue = optimal_allocation['Expected_Revenue'].sum()
        
        # Distribute revenue across days (with some randomness for realism)
        np.random.seed(42)  # For reproducibility
        revenue_distribution = np.random.normal(1, 0.1, days)
        revenue_distribution = revenue_distribution / revenue_distribution.sum()  # Normalize
        
        forecast['Revenue'] = total_revenue * revenue_distribution
    
    return forecast

# Load curve parameters (simplified for this example)
# In a real application, these would be derived from model training
platforms = ['Meta', 'Google_Search', 'Google_Display', 'Google_Video', 'Snapchat', 'TikTok']
funnel_levels = ['Lower', 'Middle', 'Upper']

# Create synthetic curve parameters based on the data
if data_loaded:
    for platform in platforms:
        for funnel in funnel_levels:
            # Filter data for this platform and funnel
            mask = (df['platform'] == platform) & (df['communication_type'] == funnel)
            if mask.any():
                platform_data = df[mask]
                
                # Calculate average metrics
                avg_spend = platform_data['budget'].mean()
                avg_revenue = platform_data['revenue'].mean()
                
                if avg_spend > 0 and avg_revenue > 0:
                    # Create Hill function parameters
                    V_max = avg_revenue * 3  # Maximum possible revenue
                    k = avg_spend  # Half-saturation constant
                    n = 1.2  # Hill coefficient
                    
                    # Store parameters
                    curve_params[f"{platform}_{funnel}"] = {
                        'V_max': V_max,
                        'k': k,
                        'n': n
                    }
    
    curves_loaded = True

# App layout
app.layout = html.Div([
    html.H1("Revenue Prediction Dashboard", style={'textAlign': 'center', 'marginBottom': '30px', 'color': 'white'}),
    
    html.Div([
        html.Div([
            html.H3("Input Parameters", style={'marginBottom': '20px', 'color': 'white'}),
            
            html.Label("Total Budget ($)", style={'color': 'white'}),
            dcc.Input(
                id='budget-input',
                type='number',
                placeholder='Enter budget amount',
                value=50000,
                style={'width': '100%', 'padding': '8px', 'marginBottom': '5px', 'backgroundColor': '#333', 'color': 'white', 'border': '1px solid #555'}
            ),
            html.Div(id='budget-display', style={'marginTop': '5px', 'marginBottom': '20px'}),
            
            html.Label("Funnel Level", style={'color': 'white'}),
            dcc.Dropdown(
                id='funnel-level-dropdown',
                options=[
                    {'label': 'Lower Funnel', 'value': 'Lower'},
                    {'label': 'Middle Funnel', 'value': 'Middle'},
                    {'label': 'Upper Funnel', 'value': 'Upper'}
                ],
                value='Lower',
                style={'color': 'black'}
            ),
            html.Div(style={'marginBottom': '20px'}),
            
            html.Label("Forecast Days", style={'color': 'white'}),
            dcc.Slider(
                id='days-slider',
                min=7,
                max=90,
                step=1,
                value=30,
                marks={i: str(i) for i in range(7, 91, 14)},
            ),
            html.Div(id='days-display', style={'marginTop': '10px', 'marginBottom': '20px'}),
            
            html.Button('Generate Prediction', id='predict-button', 
                       style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
            
        ], style={'width': '30%', 'padding': '20px', 'backgroundColor': '#222', 'borderRadius': '10px', 'color': 'white'}),
        
        html.Div([
            html.H3("Revenue Prediction Results", style={'marginBottom': '20px', 'color': 'white'}),
            
            dcc.Tabs([
                dcc.Tab(label='Revenue Forecast', children=[
                    html.Div([
                        html.Div(id='total-revenue-display', style={'fontSize': '24px', 'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px'}),
                        html.Div(id='metrics-container', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'justifyContent': 'space-between'}),
                    ], style={'padding': '20px'})
                ], style={'backgroundColor': '#222', 'color': 'white'}),
                
                dcc.Tab(label='Diminishing Returns', children=[
                    dcc.Graph(id='diminishing-returns-graph'),
                    html.Div(id='diminishing-returns-insight', style={'marginTop': '20px', 'fontSize': '16px', 'color': 'white', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px'})
                ]),
                
                dcc.Tab(label='Optimal Budget Allocation', children=[
                    dcc.Graph(id='budget-allocation-graph'),
                    html.Div(id='allocation-table-container', style={'marginTop': '20px', 'color': 'white', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px'})
                ]),
                
                dcc.Tab(label='Platform Comparison', children=[
                    dcc.Graph(id='platform-comparison-graph'),
                    html.Div(id='platform-insights', style={'marginTop': '20px', 'fontSize': '16px', 'color': 'white', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px'})
                ]),
            ]),
            
        ], style={'width': '68%', 'padding': '20px', 'backgroundColor': '#222', 'borderRadius': '10px', 'color': 'white'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    html.Div([
        html.H4("Model Information", style={'marginTop': '30px', 'color': 'white'}),
        html.P(f"Model Status: {'Loaded successfully' if model_loaded else 'Not loaded'}", style={'color': 'white'}),
        html.P(f"Data Status: {'Loaded successfully' if data_loaded else 'Not loaded'}", style={'color': 'white'}),
        html.P(f"Curve Parameters: {'Loaded successfully' if curves_loaded else 'Not loaded'}", style={'color': 'white'}),
        html.P("Dashboard created based on IKEA historical marketing data and model instructions.", style={'color': 'white'}),
    ], style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#222', 'borderRadius': '10px'}),
    
], style={'padding': '20px', 'fontFamily': 'Arial', 'backgroundColor': '#111', 'minHeight': '100vh'})

# Callback to update budget display
@callback(
    Output('budget-display', 'children'),
    Input('budget-input', 'value')
)
def update_budget_display(budget):
    if budget is None:
        budget = 0
    return f"Selected Budget: ${budget:,}"

# Callback to update days display
@callback(
    Output('days-display', 'children'),
    Input('days-slider', 'value')
)
def update_days_display(days):
    return f"Forecast Period: {days} days"

# Callback to generate predictions and update all graphs
@callback(
    [Output('metrics-container', 'children'),
     Output('diminishing-returns-graph', 'figure'),
     Output('budget-allocation-graph', 'figure'),
     Output('platform-comparison-graph', 'figure'),
     Output('total-revenue-display', 'children'),
     Output('diminishing-returns-insight', 'children'),
     Output('allocation-table-container', 'children'),
     Output('platform-insights', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('budget-input', 'value'),
     State('funnel-level-dropdown', 'value'),
     State('days-slider', 'value')]
)
def update_predictions(n_clicks, budget, funnel_level, days):
    # Default empty outputs
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Click 'Generate Prediction' to see results",
        xaxis=dict(title=""),
        yaxis=dict(title=""),
        paper_bgcolor='#222',
        plot_bgcolor='#222',
        font=dict(color='white')
    )
    
    # Default metrics cards
    default_metrics = [
        html.Div([
            html.H4("Expected Revenue", style={'textAlign': 'center', 'color': 'white'}),
            html.H2("$0", style={'textAlign': 'center', 'color': '#4CAF50'})
        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px', 'minWidth': '200px'}),
        
        html.Div([
            html.H4("ROAS", style={'textAlign': 'center', 'color': 'white'}),
            html.H2("0.00", style={'textAlign': 'center', 'color': '#2196F3'})
        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px', 'minWidth': '200px'}),
        
        html.Div([
            html.H4("Predicted Clicks", style={'textAlign': 'center', 'color': 'white'}),
            html.H2("0", style={'textAlign': 'center', 'color': '#FF9800'})
        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px', 'minWidth': '200px'}),
        
        html.Div([
            html.H4("Predicted Impressions", style={'textAlign': 'center', 'color': 'white'}),
            html.H2("0", style={'textAlign': 'center', 'color': '#E91E63'})
        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px', 'minWidth': '200px'})
    ]
    
    if n_clicks is None:
        # Initial load - return empty figures
        return (default_metrics, empty_fig, empty_fig, empty_fig, 
                "Click 'Generate Prediction' to see total revenue", 
                "Click 'Generate Prediction' to see diminishing returns insights",
                "Click 'Generate Prediction' to see allocation table",
                "Click 'Generate Prediction' to see platform insights")
    
    # Set selected funnel level
    selected_funnel = funnel_level
    
    # Handle None budget value
    if budget is None or budget <= 0:
        budget = 1  # Set minimum budget to avoid division by zero
    
    # Optimize budget allocation
    # We'll create equal weights for all funnels but filter later by selected funnel
    funnel_weights = {'Lower': 1.0, 'Middle': 1.0, 'Upper': 1.0}
    optimal_allocation = optimize_budget(budget, curve_params, funnel_weights)
    
    # Filter optimal allocation by selected funnel level
    if len(optimal_allocation) > 0:
        optimal_allocation = optimal_allocation[optimal_allocation['Funnel'] == selected_funnel]
    
    # Generate daily forecast
    forecast = generate_daily_forecast(budget, days, curve_params, optimal_allocation)
    
    # 1. Revenue Forecast Metrics
    # Create metrics cards for the revenue forecast section
    total_revenue = forecast['Revenue'].sum()
    avg_daily_revenue = forecast['Revenue'].mean()
    total_budget = budget
    overall_roas = total_revenue / total_budget if total_budget > 0 else 0
    
    # Calculate total clicks and impressions based on historical ratios
    # Get average metrics from historical data
    avg_clicks_per_revenue = 0
    avg_impressions_per_revenue = 0
    
    if data_loaded:
        funnel_data = df[df['communication_type'] == selected_funnel]
        if not funnel_data.empty and funnel_data['revenue'].sum() > 0:
            avg_clicks_per_revenue = funnel_data['clicks'].sum() / funnel_data['revenue'].sum()
            avg_impressions_per_revenue = funnel_data['impressions'].sum() / funnel_data['revenue'].sum()
    
    # Calculate predicted metrics
    predicted_clicks = total_revenue * avg_clicks_per_revenue
    predicted_impressions = total_revenue * avg_impressions_per_revenue
    
    # Create metric cards
    metrics_cards = [
        html.Div([
            html.H4("Expected Revenue", style={'textAlign': 'center', 'color': 'white'}),
            html.H2(f"${total_revenue:,.2f}", style={'textAlign': 'center', 'color': '#4CAF50'})
        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px', 'minWidth': '200px'}),
        
        html.Div([
            html.H4("ROAS", style={'textAlign': 'center', 'color': 'white'}),
            html.H2(f"{overall_roas:.2f}", style={'textAlign': 'center', 'color': '#2196F3'})
        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px', 'minWidth': '200px'}),
        
        html.Div([
            html.H4("Predicted Clicks", style={'textAlign': 'center', 'color': 'white'}),
            html.H2(f"{predicted_clicks:,.0f}", style={'textAlign': 'center', 'color': '#FF9800'})
        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px', 'minWidth': '200px'}),
        
        html.Div([
            html.H4("Predicted Impressions", style={'textAlign': 'center', 'color': 'white'}),
            html.H2(f"{predicted_impressions:,.0f}", style={'textAlign': 'center', 'color': '#E91E63'})
        ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#333', 'borderRadius': '5px', 'minWidth': '200px'})
    ]
    
    # 2. Diminishing Returns Graph
    dim_returns_fig = go.Figure()
    
    # Create spend grid
    max_budget = budget * 1.5
    spend_grid = np.linspace(0, max_budget, 100)
    
    # Calculate total revenue at each spend level
    total_revenues = []
    marginal_roas_values = []
    
    for spend in spend_grid:
        # Distribute spend according to optimal proportions
        if len(optimal_allocation) > 0:
            proportions = optimal_allocation['Optimal_Budget'] / optimal_allocation['Optimal_Budget'].sum()
            
            # Calculate revenue for each platform-funnel
            spend_revenue = 0
            for i, row in optimal_allocation.iterrows():
                platform = row['Platform']
                funnel = row['Funnel']
                pf = f"{platform}_{funnel}"
                
                if pf in curve_params:
                    params = curve_params[pf]
                    platform_spend = spend * proportions.loc[i]
                    platform_revenue = hill_function(platform_spend, params['V_max'], params['k'], params['n'])
                    spend_revenue += platform_revenue
            
            total_revenues.append(spend_revenue)
        else:
            # Fallback if no optimal allocation
            total_revenues.append(spend * 5)
    
    # Calculate marginal ROAS (first derivative of revenue curve)
    for i in range(1, len(spend_grid)):
        if spend_grid[i] - spend_grid[i-1] > 0:
            marginal = (total_revenues[i] - total_revenues[i-1]) / (spend_grid[i] - spend_grid[i-1])
        else:
            marginal = 0
        marginal_roas_values.append(marginal)
    
    # Add a zero at the beginning to match length
    marginal_roas_values = [0] + marginal_roas_values
    
    # Add revenue curve
    dim_returns_fig.add_trace(go.Scatter(
        x=spend_grid,
        y=total_revenues,
        mode='lines',
        name='Total Revenue',
        line=dict(color='green', width=3)
    ))
    
    # Add marginal ROAS curve on secondary y-axis
    dim_returns_fig.add_trace(go.Scatter(
        x=spend_grid,
        y=marginal_roas_values,
        mode='lines',
        name='Marginal ROAS',
        line=dict(color='red', width=2),
        yaxis='y2'
    ))
    
    # Add current budget marker
    dim_returns_fig.add_trace(go.Scatter(
        x=[budget],
        y=[total_revenues[np.abs(spend_grid - budget).argmin()]],
        mode='markers',
        name='Current Budget',
        marker=dict(color='blue', size=12, symbol='star')
    ))
    
    # Add horizontal line at ROAS = 1
    dim_returns_fig.add_shape(
        type="line",
        x0=0,
        y0=1,
        x1=max_budget,
        y1=1,
        line=dict(color="gray", width=2, dash="dash"),
        yref='y2'
    )
    
    # Find optimal spend (where marginal ROAS = 1)
    optimal_idx = np.abs(np.array(marginal_roas_values) - 1).argmin()
    optimal_spend = spend_grid[optimal_idx]
    
    # Add vertical line at optimal spend
    if marginal_roas_values[optimal_idx] > 0.5:  # Only if we have a meaningful optimal point
        dim_returns_fig.add_shape(
            type="line",
            x0=optimal_spend,
            y0=0,
            x1=optimal_spend,
            y1=total_revenues[optimal_idx],
            line=dict(color="white", width=2, dash="dash")
        )
        
        # Add annotation for optimal spend
        dim_returns_fig.add_annotation(
            x=optimal_spend,
            y=total_revenues[optimal_idx] / 2,
            text=f"Optimal: ${optimal_spend:,.0f}",
            showarrow=False,
            textangle=-90,
            font=dict(color="white")
        )
    
    dim_returns_fig.update_layout(
        title='Diminishing Returns Analysis',
        xaxis=dict(title='Total Budget ($)', color='white'),
        yaxis=dict(title='Total Revenue ($)', side='left', showgrid=True, gridcolor='#444'),
        yaxis2=dict(title='Marginal ROAS', side='right', overlaying='y', showgrid=False, color='white'),
        legend=dict(x=0.01, y=0.99, font=dict(color='white')),
        hovermode='x unified',
        paper_bgcolor='#222',
        plot_bgcolor='#222'
    )
    
    # 3. Budget Allocation Graph
    allocation_fig = go.Figure()
    
    if len(optimal_allocation) > 0:
        # Create bar chart for selected funnel only
        platforms = optimal_allocation['Platform'].unique()
        
        # Add bars for the selected funnel only
        colors = {'Lower': 'rgb(49,130,189)', 'Middle': 'rgb(204,204,204)', 'Upper': 'rgb(222,45,38)'}
        
        # Only add the selected funnel
        allocation_fig.add_trace(go.Bar(
            x=platforms,
            y=optimal_allocation['Optimal_Budget'],
            name=selected_funnel,
            marker_color=colors[selected_funnel]
        ))
        
        allocation_fig.update_layout(
            title=f'Optimal Budget Allocation for {selected_funnel} Funnel',
            xaxis=dict(title='Platform', color='white'),
            yaxis=dict(title='Budget Allocation ($)', color='white'),
            legend=dict(x=0.01, y=0.99, font=dict(color='white')),
            hovermode='closest',
            paper_bgcolor='#222',
            plot_bgcolor='#222'
        )
    else:
        # Fallback if no optimal allocation
        allocation_fig.update_layout(
            title="No optimal allocation data available",
            xaxis=dict(title=""),
            yaxis=dict(title=""),
            paper_bgcolor='#222',
            plot_bgcolor='#222',
            font=dict(color='white')
        )
    
    # 4. Platform Comparison Graph
    platform_fig = go.Figure()
    
    if len(optimal_allocation) > 0:
        # Create scatter plot
        platform_fig.add_trace(go.Scatter(
            x=optimal_allocation['Optimal_Budget'],
            y=optimal_allocation['Expected_Revenue'],
            mode='markers+text',
            marker=dict(
                size=optimal_allocation['ROAS'] * 10,  # Size based on ROAS
                color=optimal_allocation['ROAS'],
                colorscale='Viridis',
                colorbar=dict(title='ROAS'),
                showscale=True
            ),
            text=optimal_allocation['Platform'],
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Budget: $%{x:,.2f}<br>Revenue: $%{y:,.2f}<br>ROAS: %{marker.color:.2f}<extra></extra>'
        ))
        
        platform_fig.update_layout(
            title='Platform Performance Comparison',
            xaxis=dict(title='Budget Allocation ($)', color='white'),
            yaxis=dict(title='Expected Revenue ($)', color='white'),
            hovermode='closest',
            paper_bgcolor='#222',
            plot_bgcolor='#222',
            font=dict(color='white')
        )
    else:
        # Fallback if no optimal allocation
        platform_fig.update_layout(
            title="No platform comparison data available",
            xaxis=dict(title=""),
            yaxis=dict(title=""),
            paper_bgcolor='#222',
            plot_bgcolor='#222',
            font=dict(color='white')
        )
    
    # Generate text insights
    total_revenue = forecast['Revenue'].sum()
    total_budget = budget
    overall_roas = total_revenue / total_budget if total_budget > 0 else 0
    
    total_revenue_text = f"Total Predicted Revenue: ${total_revenue:,.2f} | Overall ROAS: {overall_roas:.2f} | Selected Funnel: {selected_funnel}"
    
    # Diminishing returns insight
    if marginal_roas_values[optimal_idx] > 0.5:
        if budget < optimal_spend:
            dim_returns_insight = (
                f"Your current budget (${budget:,.0f}) is below the optimal point (${optimal_spend:,.0f}). "
                f"Increasing budget to the optimal point could improve overall returns."
            )
        elif budget > optimal_spend:
            dim_returns_insight = (
                f"Your current budget (${budget:,.0f}) is above the optimal point (${optimal_spend:,.0f}). "
                f"Consider reallocating excess budget to other marketing activities for better returns."
            )
        else:
            dim_returns_insight = f"Your current budget (${budget:,.0f}) is at the optimal point for maximum returns."
    else:
        dim_returns_insight = "Unable to determine optimal budget point from the current data."
    
    # Create allocation table
    if len(optimal_allocation) > 0:
        # Format the table
        table_data = optimal_allocation.copy()
        table_data['Optimal_Budget'] = table_data['Optimal_Budget'].map('${:,.2f}'.format)
        table_data['Expected_Revenue'] = table_data['Expected_Revenue'].map('${:,.2f}'.format)
        table_data['ROAS'] = table_data['ROAS'].map('{:.2f}'.format)
        
        # Create the table
        allocation_table = html.Table([
            html.Thead(
                html.Tr([html.Th(col, style={'color': 'white', 'backgroundColor': '#444', 'padding': '10px'}) for col in table_data.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(table_data.iloc[i][col], style={'color': 'white', 'padding': '8px', 'borderBottom': '1px solid #555'}) 
                    for col in table_data.columns
                ]) for i in range(len(table_data))
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse'})
    else:
        allocation_table = html.Div("No allocation data available", style={'color': 'white'})
    
    # Platform insights
    if len(optimal_allocation) > 0:
        # Get top and bottom performing platforms
        top_platform = optimal_allocation.iloc[0]  # Already sorted by ROAS
        bottom_platform = optimal_allocation.iloc[-1]
        
        platform_insights_text = (
            f"Top performing platform: {top_platform['Platform']} with ROAS of {top_platform['ROAS']:.2f}. "
            f"Lowest performing platform: {bottom_platform['Platform']} with ROAS of {bottom_platform['ROAS']:.2f}. "
            f"Consider shifting budget from lower to higher performing platforms for improved overall returns."
        )
    else:
        platform_insights_text = "No platform insights available."
    
    return (metrics_cards, dim_returns_fig, allocation_fig, platform_fig, 
            total_revenue_text, dim_returns_insight, allocation_table, platform_insights_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
