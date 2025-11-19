"""
Visualization functions for the CSAT dashboard.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import io
import base64

# Color palette for the dashboard
COLOR_PALETTE = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
    'groups': {
        'Grocery': '#1f77b4',  # Blue
        'Restaurant': '#ff7f0e',  # Orange
        'Plus': '#6a1b9a',  # Darker Purple
        'PAYG': '#00C7B7',  # Deliveroo Teal
        'Yes': '#2ca02c',  # Green
        'No': '#d62728'  # Red
    },
    'gradient': ['#1f77b4', '#42a5f5', '#64b5f6', '#90caf9', '#bbdefb']
}

def get_group_color(group_name: str) -> str:
    """Get color for a group name."""
    # Normalize group name
    group_str = str(group_name).strip()
    group_lower = group_str.lower()
    
    if 'grocery' in group_lower or 'odc' in group_lower:
        return COLOR_PALETTE['groups']['Grocery']
    elif 'restaurant' in group_lower:
        return COLOR_PALETTE['groups']['Restaurant']
    elif group_str == 'Plus' or group_str == 'Yes':
        return COLOR_PALETTE['groups']['Plus']
    elif group_str == 'PAYG' or group_str == 'No':
        return COLOR_PALETTE['groups']['PAYG']
    else:
        # Default to primary palette
        return COLOR_PALETTE['primary'][0]


def create_distribution_chart(df: pd.DataFrame, column: str, title: str, 
                             xlabel: str = None, ylabel: str = "Count") -> go.Figure:
    """Create a distribution chart for a numeric or categorical column."""
    fig = go.Figure()
    
    if df[column].dtype in ['int64', 'float64']:
        # Histogram for numeric data - use blue color
        data_min = df[column].dropna().min()
        data_max = df[column].dropna().max()
        
        # Check if data is on a 1-5 scale (like CSAT, Value, Bill Shock)
        is_scale_data = (data_min >= 1 and data_max <= 5 and (data_max - data_min) <= 4)
        
        fig.add_trace(go.Histogram(
            x=df[column].dropna(),
            nbinsx=20,
            marker=dict(
                color=COLOR_PALETTE['primary'][0],  # Use primary blue
                opacity=0.7
            )
        ))
        
        layout_dict = {
            'title': title,
            'xaxis_title': xlabel or column,
            'yaxis_title': ylabel,
            'bargap': 0.1
        }
        
        # For scale data (1-5), force integer-only ticks
        if is_scale_data:
            layout_dict['xaxis'] = {
                'dtick': 1,  # Show only integer ticks
                'tickmode': 'linear',
                'tick0': 1,
                'range': [0.5, 5.5]  # Slight padding for better visualization
            }
        
        fig.update_layout(**layout_dict)
    else:
        # Bar chart for categorical data
        value_counts = df[column].value_counts()
        num_categories = len(value_counts)
        # Use single color if more than 5 categories, otherwise use varied colors
        if num_categories > 5:
            bar_color = COLOR_PALETTE['primary'][0]  # Use primary blue
        else:
            colors = COLOR_PALETTE['primary'][:num_categories] if num_categories <= len(COLOR_PALETTE['primary']) else COLOR_PALETTE['primary'] * (num_categories // len(COLOR_PALETTE['primary']) + 1)
            bar_color = colors[:num_categories] if num_categories <= 5 else COLOR_PALETTE['primary'][0]
        
        fig.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            marker=dict(
                color=bar_color,
                opacity=0.8
            )
        ))
        fig.update_layout(
            title=title,
            xaxis_title=xlabel or column,
            yaxis_title=ylabel,
            xaxis={'categoryorder': 'total descending'}
        )
    
    return fig


def create_group_comparison_chart(df: pd.DataFrame, value_col: str, 
                                 group_col: str, chart_type: str = 'bar',
                                 title: str = None) -> go.Figure:
    """Create a grouped comparison chart."""
    if chart_type == 'bar':
        grouped = df.groupby(group_col)[value_col].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] > 0]  # Filter out groups with no data
        
        fig = go.Figure()
        # Use group-specific colors
        colors = [get_group_color(str(group)) for group in grouped[group_col]]
        fig.add_trace(go.Bar(
            x=grouped[group_col],
            y=grouped['mean'],
            text=grouped['mean'].round(2),
            textposition='auto',
            marker=dict(
                color=colors,
                opacity=0.8
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         'Average: %{y:.2f}<br>' +
                         'Count: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=grouped['count']
        ))
        # Clean up y-axis label for better readability
        if 'CSAT_numeric' in value_col:
            y_label = 'Average CSAT Score'
        elif 'Value_Worth_numeric' in value_col:
            y_label = 'Average Value Score'
        elif 'Bill_Shock_numeric' in value_col:
            y_label = 'Average Bill Shock Score'
        else:
            y_label = f'Average {value_col}'
        
        fig.update_layout(
            title=title or f'{value_col} by {group_col}',
            xaxis_title=group_col,
            yaxis_title=y_label,
            xaxis={'categoryorder': 'total descending'}
        )
    elif chart_type == 'box':
        fig = go.Figure()
        for group in df[group_col].dropna().unique():
            group_data = df[df[group_col] == group][value_col].dropna()
            if len(group_data) > 0:
                fig.add_trace(go.Box(
                    y=group_data,
                    name=str(group),
                    boxmean='sd'
                ))
        # Clean up y-axis label for better readability
        if 'CSAT_numeric' in value_col:
            y_label = 'CSAT Score'
        elif 'Value_Worth_numeric' in value_col:
            y_label = 'Value Score'
        elif 'Bill_Shock_numeric' in value_col:
            y_label = 'Bill Shock Score'
        else:
            y_label = value_col
        
        fig.update_layout(
            title=title or f'{value_col} Distribution by {group_col}',
            xaxis_title=group_col,
            yaxis_title=y_label
        )
    
    return fig


def create_sentiment_pie_chart(sentiment_counts: Dict[str, int], 
                               title: str = "Sentiment Distribution") -> go.Figure:
    """Create a pie chart for sentiment distribution."""
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    
    colors = {
        'Positive': '#2ecc71',
        'Negative': '#e74c3c',
        'Neutral': '#95a5a6'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=[colors.get(label, '#3498db') for label in labels]
    )])
    
    fig.update_layout(title=title)
    return fig


def create_wordcloud(text_data: pd.Series, max_words: int = 100) -> str:
    """Create a word cloud and return as base64 encoded image."""
    # Combine all text
    text = ' '.join(text_data.dropna().astype(str))
    
    if not text or text.strip() == '':
        return None
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        colormap='viridis'
    ).generate(text)
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    return img_str


def create_correlation_heatmap(df: pd.DataFrame, columns: List[str], 
                               title: str = "Correlation Heatmap") -> go.Figure:
    """Create a correlation heatmap."""
    # Select numeric columns
    numeric_df = df[columns].select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return go.Figure()
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        width=600,
        height=600
    )
    
    return fig


def create_driver_analysis_chart(df: pd.DataFrame, driver_col: str, 
                                 title: str = None) -> go.Figure:
    """Create a chart showing driver question responses."""
    value_counts = df[driver_col].value_counts()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=value_counts.index,
        y=value_counts.values,
        marker_color='steelblue',
        opacity=0.7,
        text=value_counts.values,
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title or f'Responses for {driver_col}',
        xaxis_title='Response',
        yaxis_title='Count',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig


def create_value_factors_chart(df: pd.DataFrame, prefix: str = 
                               'Which factors most influenced whether your order felt worth the money?') -> go.Figure:
    """Create a chart showing value factors selection."""
    factor_cols = [col for col in df.columns if col.startswith(prefix)]
    
    if not factor_cols:
        return go.Figure()
    
    factor_counts = {}
    for col in factor_cols:
        # Extract just the factor name by removing the full prefix
        factor_name = col
        # Remove the prefix (handle variations with newlines and formatting)
        if prefix in factor_name:
            # Split by prefix and take everything after it
            parts = factor_name.split(prefix)
            if len(parts) > 1:
                factor_name = parts[-1]
                # Remove common suffixes and formatting
                factor_name = factor_name.replace('Please select up to 3 options.', '')
                factor_name = factor_name.replace('\n\n', '').replace('\n', '')
                # Remove leading/trailing underscores, spaces, and common prefixes
                factor_name = factor_name.strip('_').strip()
                # If it still starts with underscore, remove it
                if factor_name.startswith('_'):
                    factor_name = factor_name[1:]
        
        # If extraction didn't work well, try getting the last part after underscore
        if not factor_name or len(factor_name) > 80:
            parts = col.split('_')
            if len(parts) > 1:
                factor_name = parts[-1]
        
        # Convert to numeric and sum, handling any non-numeric values
        numeric_col = pd.to_numeric(df[col], errors='coerce').fillna(0)
        count = numeric_col.sum()
        if count > 0:  # Only include factors that were selected
            factor_counts[factor_name] = count
    
    factor_df = pd.DataFrame({
        'Factor': list(factor_counts.keys()),
        'Count': list(factor_counts.values())
    }).sort_values('Count', ascending=False)
    
    fig = go.Figure()
    # Use single color if more than 5 factors, otherwise use varied colors
    num_factors = len(factor_df)
    if num_factors > 5:
        bar_color = COLOR_PALETTE['primary'][0]  # Use primary blue
    else:
        colors = COLOR_PALETTE['primary'][:num_factors] if num_factors <= len(COLOR_PALETTE['primary']) else COLOR_PALETTE['primary'] * (num_factors // len(COLOR_PALETTE['primary']) + 1)
        bar_color = colors[:num_factors] if num_factors <= 5 else COLOR_PALETTE['primary'][0]
    
    fig.add_trace(go.Bar(
        x=factor_df['Factor'],
        y=factor_df['Count'],
        marker=dict(
            color=bar_color,
            opacity=0.8
        ),
        text=factor_df['Count'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Value Factors Influence',
        xaxis_title='Factor',
        yaxis_title='Number of Selections',
        xaxis={'categoryorder': 'total descending', 'tickangle': -45}
    )
    
    return fig


def create_mission_chart(df: pd.DataFrame, prefix: str = 
                        'I ordered deliveroo for…') -> go.Figure:
    """Create a chart showing mission reasons."""
    mission_cols = [col for col in df.columns if col.startswith(prefix)]
    
    if not mission_cols:
        return go.Figure()
    
    mission_counts = {}
    for col in mission_cols:
        mission_name = col.replace(prefix + '_', '')
        # Convert to numeric and sum, handling any non-numeric values
        numeric_col = pd.to_numeric(df[col], errors='coerce').fillna(0)
        mission_counts[mission_name] = numeric_col.sum()
    
    mission_df = pd.DataFrame({
        'Mission': list(mission_counts.keys()),
        'Count': list(mission_counts.values())
    }).sort_values('Count', ascending=False)
    
    fig = go.Figure()
    # Use single color if more than 5 missions, otherwise use varied colors
    num_missions = len(mission_df)
    if num_missions > 5:
        bar_color = COLOR_PALETTE['primary'][0]  # Use primary blue
    else:
        colors = COLOR_PALETTE['primary'][:num_missions] if num_missions <= len(COLOR_PALETTE['primary']) else COLOR_PALETTE['primary'] * (num_missions // len(COLOR_PALETTE['primary']) + 1)
        bar_color = colors[:num_missions] if num_missions <= 5 else COLOR_PALETTE['primary'][0]
    
    fig.add_trace(go.Bar(
        x=mission_df['Mission'],
        y=mission_df['Count'],
        marker=dict(
            color=bar_color,
            opacity=0.8
        ),
        text=mission_df['Count'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Order Mission Reasons',
        xaxis_title='Mission',
        yaxis_title='Count',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

