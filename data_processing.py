"""
Data processing and cleaning functions for CSAT survey analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV data file."""
    df = pd.read_csv(file_path, low_memory=False)
    return df


def clean_categorical_column(df: pd.DataFrame, col_name: str, 
                             mapping: Optional[Dict] = None) -> pd.Series:
    """Clean and standardize categorical columns."""
    if col_name not in df.columns:
        return pd.Series(dtype=object)
    
    series = df[col_name].copy()
    
    # Handle missing values
    series = series.replace(['', 'nan', 'None', np.nan], None)
    
    # Apply mapping if provided
    if mapping:
        series = series.map(mapping).fillna(series)
    
    return series


def parse_multiselect_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Parse multi-select columns (comma-separated values) into binary columns."""
    if col_name not in df.columns:
        return pd.DataFrame()
    
    # Get unique values from the column
    unique_values = set()
    for val in df[col_name].dropna():
        if pd.notna(val) and val != '':
            # Split by comma and clean
            items = [item.strip() for item in str(val).split(',')]
            unique_values.update(items)
    
    # Create binary columns for each unique value
    result_df = pd.DataFrame(index=df.index)
    for value in unique_values:
        if value:  # Skip empty strings
            result_df[f"{col_name}_{value}"] = df[col_name].apply(
                lambda x: 1 if pd.notna(x) and value in str(x) else 0
            )
    
    return result_df


def convert_scale_to_numeric(series: pd.Series, scale_mapping: Dict) -> pd.Series:
    """Convert scale responses to numeric values."""
    return series.map(scale_mapping).astype('float64')


def standardize_tracker_responses(series: pd.Series) -> pd.Series:
    """Convert tracker question responses to numeric scale (1-5)."""
    mapping = {
        'Strongly Disagree': 1,
        'Disagree': 2,
        'Neither Agree Nor Disagree': 3,
        'Agree': 4,
        'Strongly Agree': 5
    }
    return series.map(mapping)


def standardize_bill_shock(series: pd.Series) -> pd.Series:
    """Convert bill shock responses to numeric scale."""
    mapping = {
        'Much less than I expected': 1,
        'Less than I expected': 2,
        'Just as much as I expected': 3,
        'More than I expected': 4,
        'Much more than I expected': 5
    }
    return series.map(mapping)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main preprocessing function."""
    processed_df = df.copy()
    
    # Standardize key categorical columns
    processed_df['Grocery_Restaurant'] = clean_categorical_column(
        df, 'Grocery / Restaurant order'
    )
    processed_df['Plus_Customer'] = clean_categorical_column(
        df, 'Plus Customer'
    )
    processed_df['Customer_Segment'] = clean_categorical_column(
        df, 'Customer Segment'
    )
    
    # Convert CSAT to numeric
    if 'Order Journey CSAT' in df.columns:
        processed_df['CSAT_numeric'] = pd.to_numeric(
            df['Order Journey CSAT'], errors='coerce'
        )
    
    # Convert bill shock to numeric
    if 'Bill expectation/shock' in df.columns:
        processed_df['Bill_Shock_numeric'] = standardize_bill_shock(
            df['Bill expectation/shock']
        )
    
    # Convert value question to numeric
    value_col = 'Thinking about your recent order, to what extent do you feel it was worth the money you spent?'
    if value_col in df.columns:
        processed_df['Value_Worth_numeric'] = pd.to_numeric(
            df[value_col], errors='coerce'
        )
    
    # Convert tracker questions to numeric
    tracker_questions = [
        'I trusted the updates were accurate',
        'I understood what was happening with my order',
        'I felt reassured while I was waiting for my order to arrive',
        'I had enough detail on my order progress',
        'I was aware of my order updates through the order tracker or notifications'
    ]
    
    for question in tracker_questions:
        if question in df.columns:
            col_name = question.replace(' ', '_').replace('?', '').replace("'", '')
            processed_df[f'{col_name}_numeric'] = standardize_tracker_responses(
                df[question]
            )
    
    # Parse mission column (multi-select)
    mission_col = 'I ordered deliveroo for…'
    if mission_col in df.columns:
        mission_df = parse_multiselect_column(df, mission_col)
        processed_df = pd.concat([processed_df, mission_df], axis=1)
    
    # Parse value factors (multi-select)
    value_factors_col = 'Which factors most influenced whether your order felt worth the money?\n\nPlease select up to 3 options.'
    if value_factors_col in df.columns:
        value_factors_df = parse_multiselect_column(df, value_factors_col)
        processed_df = pd.concat([processed_df, value_factors_df], axis=1)
    
    # Get text columns for sentiment analysis
    text_columns = [
        'Order CSAT reason',
        'Translation to English for: Order CSAT reason',
        'What, if anything, could make you feel more reassured about your order\'s progress?'
    ]
    
    for col in text_columns:
        if col in df.columns:
            processed_df[col] = df[col].fillna('')
    
    return processed_df


def get_driver_columns(df: pd.DataFrame) -> List[str]:
    """Get list of driver question columns."""
    drivers = [
        'The right temperature',
        'A good portion size',
        'Presented well',
        'Exactly what you ordered',
        'Great quality',
        'Quick & easy'
    ]
    return [d for d in drivers if d in df.columns]

