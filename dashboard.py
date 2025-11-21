"""
Main Streamlit dashboard for CSAT Survey Analysis.
"""
import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="CSAT Survey Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

# Import modules - NLTK download will happen lazily in sentiment_analysis when needed
# Wrap imports in try-except to prevent silent failures, but don't stop - let Streamlit start
_import_error = None
try:
    import data_processing as dp
    import sentiment_analysis as sa
    import visualizations as viz
    # Import color utilities
    from visualizations import COLOR_PALETTE, get_group_color
except Exception as e:
    _import_error = e

# Cache data loading
@st.cache_data(ttl=3600)  # Cache for 1 hour, but will refresh on code changes
def load_and_process_data(file_path: str):
    """Load and process the survey data."""
    df = dp.load_data(file_path)
    processed_df = dp.preprocess_data(df)
    
    # Perform sentiment analysis on open-ended responses
    # Prioritize Translation column if it has data, otherwise use original
    text_cols = [
        'Translation to English for: Order CSAT reason',
        'Order CSAT reason'
    ]
    
    for col in text_cols:
        if col in processed_df.columns:
            # Check for actual text content (not empty strings or NaN)
            # Convert to string and filter out empty/nan values
            text_series = processed_df[col].astype(str).str.strip()
            # Filter out empty strings, 'nan' (string), and actual NaN
            non_empty = text_series[(text_series != '') & (text_series != 'nan') & (text_series != 'None')]
            if len(non_empty) > 10:  # Need at least 10 responses to be meaningful
                processed_df = sa.analyze_sentiment(processed_df, col)
                processed_df = sa.extract_themes(processed_df, col)
                break  # Use the first column that has data
    
    return processed_df, df


def generate_insights(df: pd.DataFrame, processed_df: pd.DataFrame) -> List[str]:
    """Generate automated insights from the data."""
    insights = []
    
    # CSAT insights
    if 'CSAT_numeric' in processed_df.columns:
        avg_csat = processed_df['CSAT_numeric'].mean()
        insights.append(f"**Average CSAT Score:** {avg_csat:.2f} out of 5")
        
        # CSAT distribution insights
        high_csat = (processed_df['CSAT_numeric'] >= 4).sum()
        low_csat = (processed_df['CSAT_numeric'] <= 2).sum()
        total_with_csat = processed_df['CSAT_numeric'].notna().sum()
        if total_with_csat > 0:
            insights.append(f"**CSAT Distribution:** {high_csat/total_with_csat*100:.1f}% gave 4-5 stars, {low_csat/total_with_csat*100:.1f}% gave 1-2 stars")
        
        # Group comparisons
        if 'Grocery_Restaurant' in processed_df.columns:
            grocery_mask = processed_df['Grocery_Restaurant'].astype(str).str.contains('Grocery', na=False, case=False)
            restaurant_mask = processed_df['Grocery_Restaurant'].astype(str).str.contains('Restaurant', na=False, case=False)
            grocery_csat = processed_df[grocery_mask]['CSAT_numeric'].mean()
            restaurant_csat = processed_df[restaurant_mask]['CSAT_numeric'].mean()
            if pd.notna(grocery_csat) and pd.notna(restaurant_csat):
                diff = abs(grocery_csat - restaurant_csat)
                if diff > 0.2:
                    higher = "Grocery" if grocery_csat > restaurant_csat else "Restaurant"
                    insights.append(f"**Group Difference:** {higher} orders have {diff:.2f} points higher CSAT than the other group")
        
        # Plus customer comparison
        if 'Plus_Customer' in processed_df.columns:
            plus_csat = processed_df[processed_df['Plus_Customer'] == 'Yes']['CSAT_numeric'].mean()
            non_plus_csat = processed_df[processed_df['Plus_Customer'] == 'No']['CSAT_numeric'].mean()
            if pd.notna(plus_csat) and pd.notna(non_plus_csat):
                diff = plus_csat - non_plus_csat
                if abs(diff) > 0.1:
                    insights.append(f"**Plus Customer Impact:** Plus customers rate {diff:+.2f} points {'higher' if diff > 0 else 'lower'} than non-Plus customers")
    
    # Sentiment insights - find the actual column that was analyzed
    text_col_for_insights = None
    for col in ['Translation to English for: Order CSAT reason', 'Order CSAT reason']:
        if col in processed_df.columns:
            sent_col = f'{col}_sentiment'
            if sent_col in processed_df.columns and processed_df[sent_col].notna().sum() > 0:
                text_col_for_insights = col
                break
    
    if text_col_for_insights:
        sentiment_summary = sa.get_sentiment_summary(processed_df, text_col_for_insights)
        if sentiment_summary:
            insights.append(f"**Sentiment:** {sentiment_summary['positive_pct']:.1f}% positive, {sentiment_summary['negative_pct']:.1f}% negative feedback")
            
            # Top pain points
            theme_df = sa.get_top_themes(processed_df, text_col_for_insights, n=3)
            if not theme_df.empty and len(theme_df) > 0:
                top_negative_theme = theme_df.iloc[0]['Theme']
                insights.append(f"**Top Theme:** {top_negative_theme} appears in {theme_df.iloc[0]['Count']} responses")
    
    # Value insights
    if 'Value_Worth_numeric' in processed_df.columns:
        avg_value = processed_df['Value_Worth_numeric'].mean()
        insights.append(f"**Value Perception:** Average score of {avg_value:.2f} out of 5 for 'worth the money'")
        
        # Value correlation with CSAT
        if 'CSAT_numeric' in processed_df.columns:
            value_csat_corr = processed_df[['Value_Worth_numeric', 'CSAT_numeric']].corr().iloc[0, 1]
            if abs(value_csat_corr) > 0.3:
                insights.append(f"**Value-CSAT Correlation:** Strong correlation ({value_csat_corr:.2f}) between value perception and CSAT")
    
    # Tracker insights
    tracker_cols = [col for col in processed_df.columns if col.endswith('_numeric') and 'I_' in col]
    if tracker_cols:
        tracker_avg = processed_df[tracker_cols].mean().mean()
        insights.append(f"**Tracker Satisfaction:** Average score of {tracker_avg:.2f} out of 5 across all tracker questions")
        
        # Identify weakest tracker area
        tracker_means = processed_df[tracker_cols].mean()
        weakest = tracker_means.idxmin()
        weakest_score = tracker_means.min()
        if weakest_score < 3.5:
            insights.append(f"**Tracker Improvement Opportunity:** '{weakest.replace('_numeric', '').replace('I_', '')}' has lowest score ({weakest_score:.2f})")
    
    # Driver insights
    drivers = dp.get_driver_columns(df)
    if drivers and 'CSAT_numeric' in processed_df.columns:
        driver_impact = []
        for driver in drivers:
            if driver in df.columns:
                yes_csat = processed_df[df[driver] == 'Yes']['CSAT_numeric'].mean()
                no_csat = processed_df[df[driver] == 'No']['CSAT_numeric'].mean()
                if pd.notna(yes_csat) and pd.notna(no_csat):
                    impact = yes_csat - no_csat
                    driver_impact.append((driver, impact))
        
        if driver_impact:
            max_impact = max(driver_impact, key=lambda x: x[1])
            insights.append(f"**Key Driver:** '{max_impact[0]}' has the largest impact on CSAT ({max_impact[1]:+.2f} points difference)")
    
    return insights


def generate_value_summary(processed_df: pd.DataFrame, original_df: pd.DataFrame) -> List[str]:
    """Generate comprehensive summary for Value Analysis section."""
    summary = []
    
    # Value Perception Overview
    if 'Value_Worth_numeric' in processed_df.columns:
        avg_value = processed_df['Value_Worth_numeric'].mean()
        high_value = (processed_df['Value_Worth_numeric'] >= 4).sum()
        low_value = (processed_df['Value_Worth_numeric'] <= 2).sum()
        total_value = processed_df['Value_Worth_numeric'].notna().sum()
        
        summary.append("### Value Perception Overview")
        summary.append(f"- **Average Value Score:** {avg_value:.2f} out of 5")
        if total_value > 0:
            summary.append(f"- **High Value Perception (4-5):** {high_value/total_value*100:.1f}% of customers")
            summary.append(f"- **Low Value Perception (1-2):** {low_value/total_value*100:.1f}% of customers")
        
        # Value by Groups
        if 'Grocery_Restaurant' in processed_df.columns:
            groups = processed_df['Grocery_Restaurant'].dropna().unique()
            if len(groups) >= 2:
                group_values = {}
                for group in groups:
                    group_data = processed_df[processed_df['Grocery_Restaurant'] == group]['Value_Worth_numeric'].dropna()
                    if len(group_data) > 0:
                        group_values[group] = group_data.mean()
                
                if len(group_values) >= 2:
                    max_group = max(group_values, key=group_values.get)
                    min_group = min(group_values, key=group_values.get)
                    diff = group_values[max_group] - group_values[min_group]
                    if diff > 0.1:
                        summary.append(f"- **Group Difference:** {max_group} orders have {diff:.2f} points higher value perception than {min_group}")
        
        # Plus Customer comparison
        if 'Plus_Customer' in processed_df.columns:
            plus_value = processed_df[processed_df['Plus_Customer'] == 'Yes']['Value_Worth_numeric'].mean()
            non_plus_value = processed_df[processed_df['Plus_Customer'] == 'No']['Value_Worth_numeric'].mean()
            if pd.notna(plus_value) and pd.notna(non_plus_value):
                diff = plus_value - non_plus_value
                if abs(diff) > 0.1:
                    summary.append(f"- **Plus Customer Impact:** Plus customers rate value {diff:+.2f} points {'higher' if diff > 0 else 'lower'} than non-Plus customers")
        
        # Value vs CSAT Correlation
        if 'CSAT_numeric' in processed_df.columns:
            both_values = processed_df[['Value_Worth_numeric', 'CSAT_numeric']].dropna()
            if len(both_values) > 0:
                correlation = both_values.corr().iloc[0, 1]
                summary.append(f"- **Value-CSAT Relationship:** {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} {'positive' if correlation > 0 else 'negative'} correlation (r={correlation:.3f})")
    
    # Top Value Factors
    value_prefix = 'Which factors most influenced whether your order felt worth the money?'
    factor_cols = [col for col in processed_df.columns if col.startswith(value_prefix)]
    if factor_cols:
        factor_counts = {}
        for col in factor_cols:
            factor_name = col.replace(value_prefix + '_', '')
            factor_name = factor_name.replace('Please select up to 3 options.', '').strip('_').strip()
            if factor_name:
                numeric_col = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
                factor_counts[factor_name] = numeric_col.sum()
        
        if factor_counts:
            top_factor = max(factor_counts, key=factor_counts.get)
            summary.append(f"\n### Top Value Factors")
            summary.append(f"- **Most Influential Factor:** {top_factor} (selected by {factor_counts[top_factor]:.0f} customers)")
            # Get top 3 factors
            sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if len(sorted_factors) > 1:
                summary.append(f"- **Top 3 Factors:** {', '.join([f'{f[0]} ({f[1]:.0f})' for f in sorted_factors])}")
    
    # Bill Shock Analysis (only 4-5 ratings - bill shock is ONLY 4-5, not 1-3)
    if 'Bill_Shock_numeric' in processed_df.columns:
        # Filter to only bill shock (4-5) - ratings 1-3 are NOT bill shock
        bill_shock_data = processed_df[processed_df['Bill_Shock_numeric'] >= 4].copy()
        high_shock = len(bill_shock_data)
        total_bill = processed_df['Bill_Shock_numeric'].notna().sum()
        
        summary.append(f"\n### Bill Expectation Analysis")
        if high_shock > 0:
            avg_bill_shock = bill_shock_data['Bill_Shock_numeric'].mean()
            summary.append(f"- **Average Bill Shock Score (4-5 only):** {avg_bill_shock:.2f} out of 5")
        if total_bill > 0:
            summary.append(f"- **Bill Shock (4-5 ONLY):** {high_shock/total_bill*100:.1f}% of customers experienced bills higher than expected (ratings 1-3 are NOT bill shock)")
        
        # Bill Shock vs CSAT (only 4-5 ratings)
        if 'CSAT_numeric' in processed_df.columns and high_shock > 0:
            both_values = bill_shock_data[['Bill_Shock_numeric', 'CSAT_numeric']].dropna()
            if len(both_values) > 0:
                correlation = both_values.corr().iloc[0, 1]
                if correlation < -0.3:
                    summary.append(f"- **Impact on Satisfaction:** Negative correlation (r={correlation:.3f}) suggests bill shock reduces CSAT")
    
    # Driver Performance
    drivers = dp.get_driver_columns(original_df)
    if drivers and 'CSAT_numeric' in processed_df.columns:
        driver_performance = []
        driver_impact = []
        
        for driver in drivers:
            if driver in original_df.columns:
                # Performance (% Yes)
                value_counts = original_df[driver].value_counts()
                total = len(original_df[driver].dropna())
                if total > 0:
                    yes_pct = (value_counts.get('Yes', 0) / total * 100)
                    driver_performance.append((driver, yes_pct))
                
                # Impact on CSAT
                yes_csat = processed_df[original_df[driver] == 'Yes']['CSAT_numeric'].mean()
                no_csat = processed_df[original_df[driver] == 'No']['CSAT_numeric'].mean()
                if pd.notna(yes_csat) and pd.notna(no_csat):
                    impact = yes_csat - no_csat
                    driver_impact.append((driver, impact))
        
        if driver_performance:
            summary.append(f"\n### Experience Drivers Performance")
            # Find lowest performing driver
            lowest_driver = min(driver_performance, key=lambda x: x[1])
            if lowest_driver[1] < 70:
                summary.append(f"- **Improvement Opportunity:** {lowest_driver[0]} has lowest performance ({lowest_driver[1]:.1f}% Yes responses)")
            
            # Find highest impact driver
            if driver_impact:
                max_impact = max(driver_impact, key=lambda x: x[1])
                summary.append(f"- **Highest Impact Driver:** {max_impact[0]} has largest CSAT impact ({max_impact[1]:+.2f} points difference between Yes/No)")
    
    return summary


def generate_tracker_summary(processed_df: pd.DataFrame) -> List[str]:
    """Generate comprehensive summary for Order Tracker Analysis section."""
    summary = []
    
    # Tracker Questions Analysis
    tracker_questions = [
        ('I trusted the updates were accurate', 'I_trusted_the_updates_were_accurate_numeric'),
        ('I understood what was happening with my order', 'I_understood_what_was_happening_with_my_order_numeric'),
        ('I felt reassured while I was waiting for my order to arrive', 'I_felt_reassured_while_I_was_waiting_for_my_order_to_arrive_numeric'),
        ('I had enough detail on my order progress', 'I_had_enough_detail_on_my_order_progress_numeric'),
        ('I was aware of my order updates through the order tracker or notifications', 'I_was_aware_of_my_order_updates_through_the_order_tracker_or_notifications_numeric')
    ]
    
    tracker_data = []
    for question, col_name in tracker_questions:
        if col_name in processed_df.columns:
            avg_score = processed_df[col_name].mean()
            tracker_data.append((question, col_name, avg_score))
    
    if tracker_data:
        summary.append("### Tracker Questions Performance")
        overall_avg = np.mean([score for _, _, score in tracker_data])
        summary.append(f"- **Overall Average Score:** {overall_avg:.2f} out of 5 across all tracker questions")
        
        # Find highest and lowest performing questions
        highest = max(tracker_data, key=lambda x: x[2])
        lowest = min(tracker_data, key=lambda x: x[2])
        
        summary.append(f"- **Highest Performing:** {highest[0][:50]}... (Score: {highest[2]:.2f})")
        summary.append(f"- **Lowest Performing:** {lowest[0][:50]}... (Score: {lowest[2]:.2f})")
        
        if highest[2] - lowest[2] > 0.3:
            summary.append(f"- **Performance Gap:** {highest[2] - lowest[2]:.2f} points difference between best and worst performing questions")
        
        # Tracker by Groups
        if 'Grocery_Restaurant' in processed_df.columns:
            groups = processed_df['Grocery_Restaurant'].dropna().unique()
            if len(groups) >= 2:
                group_tracker_avg = {}
                for group in groups:
                    group_tracker_cols = [col for _, col, _ in tracker_data if col in processed_df.columns]
                    if group_tracker_cols:
                        group_avg = processed_df[processed_df['Grocery_Restaurant'] == group][group_tracker_cols].mean().mean()
                        group_tracker_avg[group] = group_avg
                
                if len(group_tracker_avg) >= 2:
                    max_group = max(group_tracker_avg, key=group_tracker_avg.get)
                    min_group = min(group_tracker_avg, key=group_tracker_avg.get)
                    diff = group_tracker_avg[max_group] - group_tracker_avg[min_group]
                    if diff > 0.2:
                        summary.append(f"- **Group Difference:** {max_group} orders have {diff:.2f} points higher tracker satisfaction than {min_group}")
        
        # Tracker vs CSAT Relationship
        if 'CSAT_numeric' in processed_df.columns:
            tracker_cols = [col for _, col, _ in tracker_data if col in processed_df.columns]
            if tracker_cols:
                # Calculate average tracker score without modifying the original dataframe
                avg_tracker_scores = processed_df[tracker_cols].mean(axis=1)
                both_values = pd.DataFrame({
                    'Avg_Tracker_Score': avg_tracker_scores,
                    'CSAT_numeric': processed_df['CSAT_numeric']
                }).dropna()
                if len(both_values) > 0:
                    correlation = both_values.corr().iloc[0, 1]
                    summary.append(f"\n### Tracker Impact on Satisfaction")
                    summary.append(f"- **Tracker-CSAT Relationship:** {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} {'positive' if correlation > 0 else 'negative'} correlation (r={correlation:.3f})")
                    if abs(correlation) > 0.4:
                        summary.append(f"- **Key Insight:** Better tracker experience {'strongly' if abs(correlation) > 0.7 else 'moderately'} influences overall customer satisfaction")
    
    # Reassurance Feedback
    reassurance_col = "What, if anything, could make you feel more reassured about your order's progress?"
    if reassurance_col in processed_df.columns:
        feedback_count = processed_df[reassurance_col].notna().sum()
        if feedback_count > 0:
            summary.append(f"\n### Reassurance Feedback")
            summary.append(f"- **Feedback Received:** {feedback_count} customers provided suggestions for improvement")
    
    return summary


@st.cache_data
def apply_filters(_processed_df, _original_df, order_type, plus_customer, customer_segment):
    """Apply filters to the dataframes. Cached to avoid re-filtering on every render.
    The filter parameters (order_type, plus_customer, customer_segment) are part of the cache key,
    so changing them will invalidate the cache."""
    filtered_processed = _processed_df.copy()
    filtered_original = _original_df.copy()
    
    # Create a boolean mask for all filters
    mask = pd.Series([True] * len(filtered_processed), index=filtered_processed.index)
    
    if order_type != 'All' and 'Grocery_Restaurant' in filtered_processed.columns:
        order_mask = filtered_processed['Grocery_Restaurant'] == order_type
        mask = mask & order_mask
    
    if plus_customer != 'All' and 'Plus_Customer' in filtered_processed.columns:
        plus_mask = filtered_processed['Plus_Customer'] == plus_customer
        mask = mask & plus_mask
    
    if customer_segment != 'All' and 'Customer_Segment' in filtered_processed.columns:
        segment_mask = filtered_processed['Customer_Segment'] == customer_segment
        mask = mask & segment_mask
    
    # Apply the combined mask to ensure all columns (including mission columns) are filtered consistently
    filtered_processed = filtered_processed[mask].copy()
    # Apply the same mask to filtered_original to ensure indices match
    # Use reindex to handle cases where indices might not align perfectly
    try:
        filtered_original = filtered_original.loc[filtered_processed.index].copy()
    except KeyError:
        # If indices don't match, try to align by resetting index
        filtered_processed_reset = filtered_processed.reset_index(drop=True)
        filtered_original_reset = filtered_original.reset_index(drop=True)
        filtered_original = filtered_original_reset.loc[filtered_processed_reset.index].copy()
        filtered_processed = filtered_processed_reset
    
    return filtered_processed, filtered_original


@st.cache_data
def compute_driver_summary(_original_df, drivers, _filter_key):
    """Compute driver performance summary. Cached to avoid recalculation on scroll.
    _filter_key includes hash of dataframe index to ensure cache invalidates when filters change."""
    driver_summary = []
    for driver in drivers:
        if driver in _original_df.columns:
            value_counts = _original_df[driver].value_counts()
            total = len(_original_df[driver].dropna())
            if total > 0:
                driver_summary.append({
                    'Driver': driver,
                    'Yes %': (value_counts.get('Yes', 0) / total * 100),
                    'No %': (value_counts.get('No', 0) / total * 100),
                    'Not Sure %': (value_counts.get('Not Sure', 0) / total * 100)
                })
    return pd.DataFrame(driver_summary) if driver_summary else pd.DataFrame()


@st.cache_data
def compute_driver_csat(_processed_df, _original_df, drivers, _filter_key):
    """Compute driver impact on CSAT. Cached to avoid recalculation on scroll.
    _filter_key includes hash of dataframe index to ensure cache invalidates when filters change."""
    driver_csat = []
    if 'CSAT_numeric' not in _processed_df.columns:
        return pd.DataFrame()
    
    # Ensure indices align between _original_df and _processed_df
    # Both should have the same indices after filtering, but align them explicitly
    aligned_original = _original_df.loc[_processed_df.index] if len(_processed_df) > 0 else _original_df
    
    for driver in drivers:
        if driver in aligned_original.columns:
            # Create masks using aligned indices
            yes_mask = aligned_original[driver] == 'Yes'
            no_mask = aligned_original[driver] == 'No'
            # Apply masks to _processed_df (indices should now match)
            yes_csat = _processed_df[yes_mask]['CSAT_numeric'].mean() if yes_mask.sum() > 0 else None
            no_csat = _processed_df[no_mask]['CSAT_numeric'].mean() if no_mask.sum() > 0 else None
            if yes_csat is not None or no_csat is not None:
                driver_csat.append({
                    'Driver': driver,
                    'Yes': yes_csat if yes_csat is not None else 0,
                    'No': no_csat if no_csat is not None else 0
                })
    return pd.DataFrame(driver_csat) if driver_csat else pd.DataFrame()


@st.cache_data
def compute_comparison_data(_processed_df, comparison_metrics, group_col, _filter_key):
    """Compute comparison data for group comparisons. Cached to avoid recalculation.
    _filter_key includes hash of dataframe index to ensure cache invalidates when filters change."""
    comparison_data = []
    if group_col not in _processed_df.columns:
        return pd.DataFrame()
    
    for metric_col, metric_name in comparison_metrics:
        for group in _processed_df[group_col].dropna().unique():
            group_data = _processed_df[_processed_df[group_col] == group][metric_col].dropna()
            if len(group_data) > 0:
                comparison_data.append({
                    'Metric': metric_name,
                    'Group': group,
                    'Mean': group_data.mean(),
                    'Count': len(group_data)
                })
    return pd.DataFrame(comparison_data) if comparison_data else pd.DataFrame()


@st.cache_data
def compute_statistical_tests(_processed_df, comparison_metrics, group_col, _filter_key):
    """Compute statistical tests. Cached to avoid recalculation on scroll.
    _filter_key is used to ensure cache invalidation when filters change."""
    results = []
    if group_col not in _processed_df.columns:
        return results
    
    for metric_col, metric_name in comparison_metrics:
        groups = _processed_df[group_col].dropna().unique()
        if len(groups) >= 2:
            group1_data = _processed_df[_processed_df[group_col] == groups[0]][metric_col].dropna()
            group2_data = _processed_df[_processed_df[group_col] == groups[1]][metric_col].dropna()
            if len(group1_data) > 0 and len(group2_data) > 0:
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                results.append({
                    'metric_name': metric_name,
                    't_stat': t_stat,
                    'p_value': p_value
                })
    return results


def main():
    """Main dashboard application."""
    # Check for import errors first
    if _import_error is not None:
        st.error(f"❌ Error importing required modules: {str(_import_error)}")
        st.exception(_import_error)
        st.info("Please check that all required Python files are in the repository.")
        return
    
    st.title("📊 Order Journey CSAT Survey Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    data_file = "OJ CSAT Trial.csv"
    
    # Check if file exists first
    import os
    if not os.path.exists(data_file):
        st.error(f"❌ Data file '{data_file}' not found in the current directory.")
        st.info(f"Current directory: {os.getcwd()}")
        st.info("Please ensure the CSV file is in the same directory as dashboard.py")
        st.info("**Available files in directory:**")
        try:
            files = os.listdir('.')
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                st.write("CSV files found:", ', '.join(csv_files))
            else:
                st.write("No CSV files found in directory")
        except Exception:
            pass
        return  # Return early instead of stopping
    
    try:
        with st.spinner("Loading and processing data..."):
            processed_df, original_df = load_and_process_data(data_file)
        
        st.sidebar.header("Filters")
        
        # Sidebar filters - get selections first
        selected_order_type = 'All'
        selected_plus = 'All'
        selected_segment = 'All'
        
        if 'Grocery_Restaurant' in processed_df.columns:
            order_types = ['All'] + list(processed_df['Grocery_Restaurant'].dropna().unique())
            selected_order_type = st.sidebar.selectbox("Order Type", order_types)
        
        if 'Plus_Customer' in processed_df.columns:
            plus_options = ['All'] + list(processed_df['Plus_Customer'].dropna().unique())
            selected_plus = st.sidebar.selectbox("Plus Customer", plus_options)
        
        if 'Customer_Segment' in processed_df.columns:
            segments = ['All'] + list(processed_df['Customer_Segment'].dropna().unique())
            selected_segment = st.sidebar.selectbox("Customer Segment", segments)
        
        # Apply filters using cached function
        processed_df, original_df = apply_filters(
            processed_df, original_df, 
            selected_order_type, selected_plus, selected_segment
        )
        
        # Single scrollable page with sections
        
        # ============================================
        # OVERVIEW SECTION
        # ============================================
        st.markdown("---")
        st.markdown("# 📋 OVERVIEW")
        st.markdown("---")
        st.markdown("**Purpose:** High-level overview of the survey data providing context for the Value and Order Tracker analyses below.")
        st.markdown("")
        
        # Executive Summary
        st.subheader("Executive Summary")
        st.markdown("*Key metrics and insights from the survey data.*")
        
        col1, col2, col3, col4 = st.columns(4)
            
        with col1:
            if 'CSAT_numeric' in processed_df.columns:
                avg_csat = processed_df['CSAT_numeric'].mean()
                st.metric("Average CSAT Score", f"{avg_csat:.2f}", delta=None)
        
        with col2:
            total_responses = len(processed_df)
            st.metric("Total Responses", f"{total_responses:,}")
        
        with col3:
            if 'Value_Worth_numeric' in processed_df.columns:
                avg_value = processed_df['Value_Worth_numeric'].mean()
                st.metric("Avg Value Score", f"{avg_value:.2f}", delta=None)
        
        with col4:
            # Find sentiment column
            sentiment_col = None
            for col in ['Translation to English for: Order CSAT reason', 'Order CSAT reason']:
                if col in processed_df.columns:
                    sent_col = f'{col}_sentiment'
                    if sent_col in processed_df.columns and processed_df[sent_col].notna().sum() > 0:
                        sentiment_col = sent_col
                        text_col_for_count = col
                        break
            
            if sentiment_col and sentiment_col in processed_df.columns:
                total_with_text = processed_df[text_col_for_count].notna().sum()
                if total_with_text > 0:
                    positive_pct = (processed_df[sentiment_col] == 'Positive').sum() / total_with_text * 100
                    st.metric("Positive Sentiment", f"{positive_pct:.1f}%", delta=None)
        
        st.markdown("### Key Insights")
        st.markdown("*Automatically generated insights highlighting important patterns and differences in the data.*")
        insights = generate_insights(original_df, processed_df)
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Sentiment summary
        st.markdown("### Sentiment Summary")
        st.markdown("*Analysis of open-ended feedback showing the overall sentiment (positive, negative, or neutral) of customer comments.*")
        # Find the actual sentiment column that was created
        sentiment_col = None
        text_col_for_summary = None
        for col in ['Translation to English for: Order CSAT reason', 'Order CSAT reason']:
            if col in processed_df.columns:
                sent_col = f'{col}_sentiment'
                if sent_col in processed_df.columns and processed_df[sent_col].notna().sum() > 0:
                    sentiment_col = sent_col
                    text_col_for_summary = col
                    break
        
        if sentiment_col and sentiment_col in processed_df.columns:
            sentiment_counts = processed_df[sentiment_col].value_counts().to_dict()
            if sentiment_counts:
                col1, col2 = st.columns(2)
                with col1:
                    fig = viz.create_sentiment_pie_chart(sentiment_counts, "Overall Sentiment Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    sentiment_summary = sa.get_sentiment_summary(processed_df, text_col_for_summary)
                    if sentiment_summary:
                        st.markdown(f"""
                        **Sentiment Breakdown:**
                        - Positive: {sentiment_summary['positive_count']} ({sentiment_summary['positive_pct']:.1f}%)
                        - Negative: {sentiment_summary['negative_count']} ({sentiment_summary['negative_pct']:.1f}%)
                        - Neutral: {sentiment_summary['neutral_count']} ({sentiment_summary['neutral_pct']:.1f}%)
                        """)
        
        # Overall CSAT Analysis
        st.subheader("Overall CSAT Analysis")
        st.markdown("*Basic CSAT distribution and statistics to provide context for the detailed analyses below.*")
        if 'CSAT_numeric' in processed_df.columns:
            # Overall distribution
            st.markdown("**CSAT Distribution:**")
            fig = viz.create_distribution_chart(
                processed_df, 'CSAT_numeric',
                "Order Journey CSAT Distribution",
                "CSAT Score"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("**Statistical Summary:**")
            summary_stats = processed_df['CSAT_numeric'].describe()
            st.dataframe(summary_stats.to_frame().T)
        
        st.markdown("---")
        
        # ============================================
        # SECTION 1: VALUE ANALYSIS
        # ============================================
        st.markdown("# 💰 VALUE ANALYSIS")
        st.markdown("---")
        st.markdown("**Purpose:** Comprehensive analysis of how customers perceive the value of their orders, including value perception, bill expectations, experience drivers, and their impact on satisfaction.")
        st.markdown("")
        if 'Value_Worth_numeric' in processed_df.columns:
            # Value distribution
            st.subheader("Value Perception Distribution")
            st.markdown("*Shows how customers rated whether their order was 'worth the money' on a 5-point scale.*")
            fig = viz.create_distribution_chart(
                processed_df, 'Value_Worth_numeric',
                "Worth the Money Score Distribution",
                "Score (1-5)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Value factors
            st.subheader("Value Factors Influence")
            st.markdown("*Identifies which factors (e.g., food quality, price, time saved) most influence customers' perception of value.*")
            value_prefix = 'Which factors most influenced whether your order felt worth the money?'
            fig = viz.create_value_factors_chart(processed_df, value_prefix)
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
            
            # Value by groups
            st.subheader("Value Perception by Groups")
            st.markdown("*Compares value perception scores across different customer groups to see if certain segments feel they get better value.*")
            col1, col2 = st.columns(2)
            with col1:
                if 'Grocery_Restaurant' in processed_df.columns:
                    fig = viz.create_group_comparison_chart(
                        processed_df, 'Value_Worth_numeric', 'Grocery_Restaurant',
                        title="Value by Order Type"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'Plus_Customer' in processed_df.columns:
                    fig = viz.create_group_comparison_chart(
                        processed_df, 'Value_Worth_numeric', 'Plus_Customer',
                        title="Value by Plus Customer"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation with CSAT
            st.subheader("Value vs CSAT Relationship")
            st.markdown("*This scatter plot shows how customers' perception of value (whether the order was 'worth the money') relates to their overall satisfaction (CSAT). Each point represents one customer's responses. A positive correlation means customers who feel they got good value tend to give higher CSAT scores.*")
            if 'CSAT_numeric' in processed_df.columns:
                # Filter to only rows with both values
                both_values = processed_df[['Value_Worth_numeric', 'CSAT_numeric']].dropna()
                if len(both_values) > 0:
                    correlation = both_values.corr().iloc[0, 1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Correlation Coefficient", f"{correlation:.3f}", 
                                 help="Range: -1 to +1. Values closer to +1 indicate a strong positive relationship.")
                    with col2:
                        st.metric("Sample Size", f"{len(both_values):,}", 
                                 help="Number of customers who answered both questions")
                    
                    # Create scatter plot with trend line
                    fig = go.Figure()
                    
                    # Sample data if too many points for better performance
                    if len(both_values) > 1000:
                        sample_size = 1000
                        both_values_sampled = both_values.sample(n=sample_size, random_state=42)
                    else:
                        both_values_sampled = both_values
                    
                    # Add scatter points
                    fig.add_trace(go.Scatter(
                        x=both_values_sampled['Value_Worth_numeric'],
                        y=both_values_sampled['CSAT_numeric'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.5,
                            color='steelblue',
                            line=dict(width=0)
                        ),
                        name='Customer Responses',
                        hovertemplate='Value: %{x}<br>CSAT: %{y}<extra></extra>',
                        showlegend=False
                    ))
                    
                    # Add trend line
                    x_vals = both_values['Value_Worth_numeric'].values
                    y_vals = both_values['CSAT_numeric'].values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                    line_x = np.linspace(x_vals.min(), x_vals.max(), 100)
                    line_y = slope * line_x + intercept
                    
                    fig.add_trace(go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='Trend: CSAT = %{y:.2f} for Value = %{x:.1f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Value Perception vs Customer Satisfaction",
                        xaxis_title="Value Score (1 = Not worth it, 5 = Very worth it)",
                        yaxis_title="CSAT Score (1 = Very Dissatisfied, 5 = Very Satisfied)",
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    if abs(correlation) > 0.7:
                        strength = "strong"
                    elif abs(correlation) > 0.4:
                        strength = "moderate"
                    else:
                        strength = "weak"
                    
                    direction = "positive" if correlation > 0 else "negative"
                    
                    st.info(f"**Interpretation:** There is a {strength} {direction} relationship (r={correlation:.3f}) between value perception and CSAT. This suggests that customers' perception of value {'strongly influences' if abs(correlation) > 0.7 else 'influences'} their overall satisfaction.")
        
        # Bill Shock & Mission Analysis (part of Value)
        st.subheader("Bill Expectation & Mission Analysis")
        st.markdown("*Examines whether customers' bill expectations were met and what reasons they had for placing their order - factors that influence value perception.*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Bill Expectation/Shock:**")
            st.markdown("*Bill shock is ONLY defined as customers who rated 4-5 (bills higher than expected). Ratings 1-3 are NOT bill shock and are excluded from this analysis.*")
            if 'Bill_Shock_numeric' in processed_df.columns:
                # Filter to only show 4-5 ratings (bill shock is ONLY 4-5, not 1-3)
                bill_shock_filtered = processed_df[processed_df['Bill_Shock_numeric'] >= 4].copy()
                if len(bill_shock_filtered) > 0:
                    # Create a custom chart that only shows 4 and 5
                    value_counts = bill_shock_filtered['Bill_Shock_numeric'].value_counts().sort_index()
                    # Only keep 4 and 5
                    value_counts = value_counts[value_counts.index >= 4]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=value_counts.index.astype(int),
                            y=value_counts.values,
                            marker=dict(
                                color=COLOR_PALETTE['primary'][0],
                                opacity=0.8
                            ),
                            text=value_counts.values,
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Bill Shock Distribution (4-5 Only)",
                        xaxis_title="Bill Shock Score (4-5 = Bills higher than expected)",
                        yaxis_title="Count",
                        xaxis=dict(
                            dtick=1,
                            tickmode='linear',
                            tick0=4,
                            range=[3.5, 5.5]
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bill shock by groups
                    if 'Grocery_Restaurant' in bill_shock_filtered.columns:
                        fig = viz.create_group_comparison_chart(
                            bill_shock_filtered, 'Bill_Shock_numeric', 'Grocery_Restaurant',
                            title="Bill Shock by Order Type (4-5 Only)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No customers with bill shock (ratings 4-5) found in the filtered data.")
        
        with col2:
            st.markdown("**Order Mission Reasons:**")
            st.markdown("*Shows the most common reasons customers placed orders (e.g., quick meal, treat, family time, topping up groceries).*")
            mission_prefix = 'I ordered deliveroo for…'
            fig = viz.create_mission_chart(processed_df, mission_prefix)
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
        
        # Relationship analysis
        st.markdown("**Bill Shock vs CSAT:**")
        st.markdown("*This scatter plot shows how bill shock (ratings 4-5 ONLY) relates to overall satisfaction. Ratings 1-3 are NOT bill shock and are excluded. Each point represents one customer with bill shock. A negative correlation suggests that unexpected costs reduce satisfaction.*")
        if 'Bill_Shock_numeric' in processed_df.columns and 'CSAT_numeric' in processed_df.columns:
            # Filter to only rows with both values and bill shock (4-5)
            bill_shock_data = processed_df[processed_df['Bill_Shock_numeric'] >= 4].copy()
            both_values = bill_shock_data[['Bill_Shock_numeric', 'CSAT_numeric']].dropna()
            if len(both_values) > 0:
                correlation = both_values.corr().iloc[0, 1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Correlation Coefficient", f"{correlation:.3f}",
                             help="Range: -1 to +1. Negative values indicate bill shock reduces satisfaction.")
                with col2:
                    st.metric("Sample Size", f"{len(both_values):,}",
                             help="Number of customers who answered both questions")
                
                # Create scatter plot with trend line
                fig = go.Figure()
                
                # Sample data if too many points for better performance
                if len(both_values) > 1000:
                    sample_size = 1000
                    both_values_sampled = both_values.sample(n=sample_size, random_state=42)
                else:
                    both_values_sampled = both_values
                
                # Add scatter points
                fig.add_trace(go.Scatter(
                    x=both_values_sampled['Bill_Shock_numeric'],
                    y=both_values_sampled['CSAT_numeric'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        opacity=0.5,
                        color='steelblue',
                        line=dict(width=0)
                    ),
                    name='Customer Responses',
                    hovertemplate='Bill Shock: %{x}<br>CSAT: %{y}<extra></extra>',
                    showlegend=False
                ))
                
                # Add trend line
                x_vals = both_values['Bill_Shock_numeric'].values
                y_vals = both_values['CSAT_numeric'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                line_x = np.linspace(x_vals.min(), x_vals.max(), 100)
                line_y = slope * line_x + intercept
                
                fig.add_trace(go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='Trend: CSAT = %{y:.2f} for Bill Shock = %{x:.1f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Bill Expectation vs Customer Satisfaction (Bill Shock 4-5 Only)",
                    xaxis_title="Bill Shock Score (4-5 = Bills higher than expected)",
                    yaxis_title="CSAT Score (1 = Very Dissatisfied, 5 = Very Satisfied)",
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    xaxis=dict(
                        dtick=1,
                        tickmode='linear',
                        tick0=4,
                        range=[3.5, 5.5]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                if abs(correlation) > 0.7:
                    strength = "strong"
                elif abs(correlation) > 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                direction = "positive" if correlation > 0 else "negative"
                impact = "increases" if correlation > 0 else "decreases"
                
                st.info(f"**Interpretation:** There is a {strength} {direction} relationship (r={correlation:.3f}) between bill expectations and CSAT. This suggests that when bills are {'higher' if correlation < 0 else 'lower'} than expected, customer satisfaction {impact}.")
        
        # Drivers Analysis (part of Value)
        st.subheader("Experience Drivers Analysis")
        st.markdown("*Analyzes key experience drivers (temperature, portion size, quality, etc.) and their impact on value perception and overall satisfaction.*")
        drivers = dp.get_driver_columns(original_df)
        
        if drivers:
            st.markdown("**Driver Performance:**")
            st.markdown("*Shows the percentage of customers who answered 'Yes' for each driver (e.g., right temperature, good portion size, great quality).*")
            
            # Create driver summary using cached function
            # Create a filter key to ensure cache updates when filters change
            # Include hash of dataframe index to ensure cache invalidates when filtered data changes
            try:
                # Use pandas hash function for dataframe index
                original_hash = hash(tuple(original_df.index)) if len(original_df) > 0 else 0
                processed_hash = hash(tuple(processed_df.index)) if len(processed_df) > 0 else 0
            except (TypeError, ValueError):
                # Fallback to length if hash fails
                original_hash = len(original_df)
                processed_hash = len(processed_df)
            filter_key = f"{selected_order_type}_{selected_plus}_{selected_segment}_{original_hash}_{processed_hash}"
            driver_df = compute_driver_summary(original_df, drivers, filter_key)
            
            # Yes percentage chart
            if not driver_df.empty:
                # Use single color if more than 5 drivers, otherwise use varied colors
                num_drivers = len(driver_df)
                if num_drivers > 5:
                    bar_color = COLOR_PALETTE['primary'][0]  # Use primary blue
                else:
                    colors = COLOR_PALETTE['primary'][:num_drivers] if num_drivers <= len(COLOR_PALETTE['primary']) else COLOR_PALETTE['primary'] * (num_drivers // len(COLOR_PALETTE['primary']) + 1)
                    bar_color = colors[:num_drivers] if num_drivers <= 5 else COLOR_PALETTE['primary'][0]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=driver_df['Driver'],
                        y=driver_df['Yes %'],
                        marker=dict(
                            color=bar_color,
                            opacity=0.8
                        ),
                        text=driver_df['Yes %'].round(1),
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Driver Performance (% Yes Responses)",
                    xaxis_title="Driver",
                    yaxis_title="% Yes",
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Driver impact on CSAT
            st.markdown("**Driver Impact on CSAT:**")
            st.markdown("*Compares average CSAT scores for customers who answered 'Yes' vs 'No' for each driver, showing which drivers have the biggest impact on satisfaction.*")
            driver_csat_df = compute_driver_csat(processed_df, original_df, drivers, filter_key)
            
            if not driver_csat_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Yes',
                    x=driver_csat_df['Driver'],
                    y=driver_csat_df['Yes'],
                    marker_color=COLOR_PALETTE['groups']['Yes']
                ))
                fig.add_trace(go.Bar(
                    name='No',
                    x=driver_csat_df['Driver'],
                    y=driver_csat_df['No'],
                    marker_color=COLOR_PALETTE['groups']['No']
                ))
                fig.update_layout(
                    title="Average CSAT by Driver Response",
                    xaxis_title="Driver",
                    yaxis_title="Average CSAT Score",
                    xaxis_tickangle=-45,
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Value-related Group Comparisons
        st.subheader("Value-Related Group Comparisons")
        st.markdown("*Statistical comparisons across customer segments for value-related metrics (Value Score, Bill Shock). Note: Bill Shock is ONLY ratings 4-5 - ratings 1-3 are NOT bill shock and are excluded.*")
        
        # Comparison metrics for value
        value_comparison_metrics = []
        if 'Value_Worth_numeric' in processed_df.columns:
            value_comparison_metrics.append(('Value_Worth_numeric', 'Value Score'))
        if 'Bill_Shock_numeric' in processed_df.columns:
            value_comparison_metrics.append(('Bill_Shock_numeric', 'Bill Shock Score'))
        
        # Filter processed_df for bill shock comparisons (only 4-5)
        # Create a copy where bill shock values < 4 are set to NaN so they're excluded from comparisons
        # Note: processed_df is already filtered by apply_filters, so comparison_df will also be filtered
        comparison_df = processed_df.copy()
        if 'Bill_Shock_numeric' in comparison_df.columns:
            # For bill shock comparisons, only include 4-5 ratings by setting others to NaN
            comparison_df.loc[comparison_df['Bill_Shock_numeric'] < 4, 'Bill_Shock_numeric'] = np.nan
        
        if value_comparison_metrics:
            # Create filter key for cache invalidation
            # Include hash of dataframe index to ensure cache invalidates when filtered data changes
            try:
                comparison_hash = hash(tuple(comparison_df.index)) if len(comparison_df) > 0 else 0
            except (TypeError, ValueError):
                # Fallback to length if hash fails
                comparison_hash = len(comparison_df)
            filter_key = f"{selected_order_type}_{selected_plus}_{selected_segment}_{comparison_hash}"
            
            # Group by Grocery/Restaurant
            # Only show this comparison if filter is 'All' (otherwise we're comparing within a single group)
            if 'Grocery_Restaurant' in comparison_df.columns:
                # Check if we have multiple groups in the filtered data
                unique_groups = comparison_df['Grocery_Restaurant'].dropna().unique()
                if len(unique_groups) > 1 or selected_order_type == 'All':
                    st.markdown("**Grocery vs Restaurant:**")
                    comp_df = compute_comparison_data(comparison_df, value_comparison_metrics, 'Grocery_Restaurant', filter_key)
                    
                    if not comp_df.empty:
                        color_map = {group: get_group_color(group) for group in comp_df['Group'].unique()}
                        fig = px.bar(
                            comp_df,
                            x='Metric',
                            y='Mean',
                            color='Group',
                            title="Value Metrics Comparison by Order Type",
                            barmode='group',
                            text='Mean',
                            color_discrete_map=color_map
                        )
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical test (only if we have 2+ groups)
                        if len(unique_groups) >= 2:
                            test_results = compute_statistical_tests(comparison_df, value_comparison_metrics, 'Grocery_Restaurant', filter_key)
                            if test_results:
                                st.markdown("**Statistical Significance:**")
                                for result in test_results:
                                    st.markdown(f"- **{result['metric_name']}:** t-statistic={result['t_stat']:.3f}, p-value={result['p_value']:.4f} {'(significant)' if result['p_value'] < 0.05 else '(not significant)'}")
                else:
                    # Only one group in filtered data - show single group metrics
                    st.markdown(f"**Value Metrics for {unique_groups[0]}:**")
                    single_group_data = []
                    for metric_col, metric_name in value_comparison_metrics:
                        if metric_col in comparison_df.columns:
                            metric_data = comparison_df[metric_col].dropna()
                            if len(metric_data) > 0:
                                single_group_data.append({
                                    'Metric': metric_name,
                                    'Mean': metric_data.mean(),
                                    'Count': len(metric_data)
                                })
                    
                    if single_group_data:
                        single_df = pd.DataFrame(single_group_data)
                        fig = px.bar(
                            single_df,
                            x='Metric',
                            y='Mean',
                            title=f"Value Metrics for {unique_groups[0]}",
                            text='Mean',
                            color_discrete_sequence=[get_group_color(unique_groups[0])]
                        )
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
            
            # Plus Customer comparison
            if 'Plus_Customer' in comparison_df.columns:
                # Check if we have multiple groups in the filtered data
                unique_plus_groups = comparison_df['Plus_Customer'].dropna().unique()
                if len(unique_plus_groups) > 1 or selected_plus == 'All':
                    st.markdown("**Plus vs Non-Plus Customers:**")
                    comp_df_raw = compute_comparison_data(comparison_df, value_comparison_metrics, 'Plus_Customer', filter_key)
                    
                    if not comp_df_raw.empty:
                        # Map Yes/No to Plus/PAYG for display
                        group_label_map = {'Yes': 'Plus', 'No': 'PAYG'}
                        comp_df = comp_df_raw.copy()
                        comp_df['Group'] = comp_df['Group'].map(group_label_map).fillna(comp_df['Group'])
                        color_map = {group: get_group_color(group) for group in comp_df['Group'].unique()}
                        fig = px.bar(
                            comp_df,
                            x='Metric',
                            y='Mean',
                            color='Group',
                            title="Value Metrics Comparison by Plus Customer Status",
                            barmode='group',
                            text='Mean',
                            color_discrete_map=color_map
                        )
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical test (only if we have 2+ groups)
                        if len(unique_plus_groups) >= 2:
                            test_results = compute_statistical_tests(comparison_df, value_comparison_metrics, 'Plus_Customer', filter_key)
                            if test_results:
                                st.markdown("**Statistical Significance:**")
                                for result in test_results:
                                    st.markdown(f"- **{result['metric_name']}:** t-statistic={result['t_stat']:.3f}, p-value={result['p_value']:.4f} {'(significant)' if result['p_value'] < 0.05 else '(not significant)'}")
                else:
                    # Only one group in filtered data - show single group metrics
                    group_label = 'Plus' if unique_plus_groups[0] == 'Yes' else 'PAYG'
                    st.markdown(f"**Value Metrics for {group_label} Customers:**")
                    single_group_data = []
                    for metric_col, metric_name in value_comparison_metrics:
                        if metric_col in comparison_df.columns:
                            metric_data = comparison_df[metric_col].dropna()
                            if len(metric_data) > 0:
                                single_group_data.append({
                                    'Metric': metric_name,
                                    'Mean': metric_data.mean(),
                                    'Count': len(metric_data)
                                })
                    
                    if single_group_data:
                        single_df = pd.DataFrame(single_group_data)
                        fig = px.bar(
                            single_df,
                            x='Metric',
                            y='Mean',
                            title=f"Value Metrics for {group_label} Customers",
                            text='Mean',
                            color_discrete_sequence=[get_group_color(unique_plus_groups[0])]
                        )
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
        
        # Value-related Sentiment Themes
        st.subheader("Value-Related Sentiment Themes")
        st.markdown("*Key themes in customer feedback related to value, price, and quality.*")
        # Find which text column was actually analyzed
        text_col = None
        for col in ['Translation to English for: Order CSAT reason', 'Order CSAT reason']:
            if col in processed_df.columns:
                sentiment_col = f'{col}_sentiment'
                if sentiment_col in processed_df.columns:
                    if processed_df[sentiment_col].notna().sum() > 0:
                        text_col = col
                        break
        
        if text_col and text_col in processed_df.columns:
            # Filter themes to value-related ones
            theme_df = sa.get_top_themes(processed_df, text_col, n=15)
            if not theme_df.empty:
                # Filter for value-related themes
                value_themes = ['Value Perception', 'Food Quality Issues', 'Missing Items', 'Wrong Order', 'Packaging']
                value_theme_df = theme_df[theme_df['Theme'].isin(value_themes)]
                if value_theme_df.empty:
                    # If no exact matches, show top themes that might relate to value
                    value_theme_df = theme_df.head(5)
                
                if not value_theme_df.empty:
                    num_themes = len(value_theme_df)
                    # Use single color if more than 5 themes, otherwise use varied colors
                    if num_themes > 5:
                        bar_color = COLOR_PALETTE['primary'][0]  # Use primary blue
                    else:
                        colors = COLOR_PALETTE['primary'][:num_themes] if num_themes <= len(COLOR_PALETTE['primary']) else COLOR_PALETTE['primary'] * (num_themes // len(COLOR_PALETTE['primary']) + 1)
                        bar_color = colors[:num_themes] if num_themes <= 5 else COLOR_PALETTE['primary'][0]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=value_theme_df['Theme'],
                            y=value_theme_df['Count'],
                            marker=dict(
                                color=bar_color,
                                opacity=0.8
                            ),
                            text=value_theme_df['Count'],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Value-Related Themes in Feedback",
                        xaxis_title="Theme",
                        yaxis_title="Number of Mentions",
                        xaxis={'categoryorder': 'total descending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Value Analysis Summary
        st.subheader("Value Analysis Summary")
        st.markdown("*Key insights and findings from the Value Analysis section.*")
        value_summary = generate_value_summary(processed_df, original_df)
        for line in value_summary:
            st.markdown(line)
        
        st.markdown("---")
        
        # ============================================
        # SECTION 2: ORDER TRACKER ANALYSIS
        # ============================================
        st.markdown("# 📍 ORDER TRACKER ANALYSIS")
        st.markdown("---")
        st.markdown("**Purpose:** Comprehensive analysis of customer satisfaction with order tracking features, including tracker questions, reassurance, and their impact on overall satisfaction.")
        st.markdown("")
        
        # Define tracker questions
        tracker_questions = [
            ('I trusted the updates were accurate', 'I_trusted_the_updates_were_accurate_numeric'),
            ('I understood what was happening with my order', 'I_understood_what_was_happening_with_my_order_numeric'),
            ('I felt reassured while I was waiting for my order to arrive', 'I_felt_reassured_while_I_was_waiting_for_my_order_to_arrive_numeric'),
            ('I had enough detail on my order progress', 'I_had_enough_detail_on_my_order_progress_numeric'),
            ('I was aware of my order updates through the order tracker or notifications', 'I_was_aware_of_my_order_updates_through_the_order_tracker_or_notifications_numeric')
        ]
        
        # Tracker Questions Overview
        st.subheader("Tracker Questions Overview")
        st.markdown("*Shows the percentage of customers who selected each rating (1-5) for each tracker question.*")
        
        # Rating Distribution Chart
        st.markdown("**Rating Distribution by Question:**")
        st.markdown("*Shows the percentage of customers who selected each rating (1-5) for each tracker question.*")
        
        # Collect rating distribution data
        rating_dist_data = []
        for question, col_name in tracker_questions:
            if col_name in processed_df.columns:
                # Get value counts for each rating (1-5)
                value_counts = processed_df[col_name].value_counts().sort_index()
                total = processed_df[col_name].notna().sum()
                
                # Calculate percentages for each rating
                for rating in range(1, 6):
                    count = value_counts.get(rating, 0)
                    pct = (count / total * 100) if total > 0 else 0
                    rating_dist_data.append({
                        'Question': question[:50] + '...' if len(question) > 50 else question,  # Truncate long questions
                        'Rating': rating,
                        'Percentage': pct,
                        'Count': count
                    })
        
        if rating_dist_data:
            rating_dist_df = pd.DataFrame(rating_dist_data)
            
            # Create grouped bar chart
            fig = go.Figure()
            
            # Add a bar for each rating (1-5)
            ratings = sorted(rating_dist_df['Rating'].unique())
            colors_map = {1: '#d62728', 2: '#ff7f0e', 3: '#bcbd22', 4: '#2ca02c', 5: '#1f77b4'}  # Red to Blue gradient
            
            for rating in ratings:
                rating_data = rating_dist_df[rating_dist_df['Rating'] == rating]
                fig.add_trace(go.Bar(
                    name=f'Rating {rating}',
                    x=rating_data['Question'],
                    y=rating_data['Percentage'],
                    marker=dict(color=colors_map.get(rating, COLOR_PALETTE['primary'][0])),
                    text=[f'{p:.1f}%<br>({c})' for p, c in zip(rating_data['Percentage'], rating_data['Count'])],
                    textposition='auto',
                    hovertemplate='Question: %{x}<br>Rating: ' + str(rating) + '<br>Percentage: %{y:.1f}%<br>Count: %{customdata}<extra></extra>',
                    customdata=rating_data['Count']
                ))
            
            fig.update_layout(
                title="Rating Distribution by Tracker Question",
                xaxis_title="Question",
                yaxis_title="Percentage of Responses (%)",
                barmode='group',
                xaxis_tickangle=-45,
                legend=dict(title="Rating")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tracker by Groups
        st.subheader("Tracker Scores by Groups")
        st.markdown("*Compares tracker satisfaction scores between different customer groups to identify if tracking experience differs by segment.*")
        if 'Grocery_Restaurant' in processed_df.columns and tracker_questions:
            group_tracker = []
            for question, col_name in tracker_questions:
                if col_name in processed_df.columns:
                    for group in processed_df['Grocery_Restaurant'].dropna().unique():
                        group_avg = processed_df[processed_df['Grocery_Restaurant'] == group][col_name].mean()
                        group_tracker.append({
                            'Question': question[:30] + '...',
                            'Group': group,
                            'Average Score': group_avg
                        })
            
            if group_tracker:
                group_tracker_df = pd.DataFrame(group_tracker)
                color_map = {group: get_group_color(group) for group in group_tracker_df['Group'].unique()}
                fig = px.bar(
                    group_tracker_df,
                    x='Question',
                    y='Average Score',
                    color='Group',
                    title="Tracker Scores by Order Type",
                    barmode='group',
                    color_discrete_map=color_map
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Reassurance Improvement Feedback
        st.subheader("Reassurance Improvement Feedback")
        st.markdown("*Customer suggestions on how to improve the order tracking experience and make them feel more reassured during delivery.*")
        reassurance_col = "What, if anything, could make you feel more reassured about your order's progress?"
        if reassurance_col in processed_df.columns:
            feedback_texts = processed_df[reassurance_col].dropna().head(10)
            for idx, text in enumerate(feedback_texts, 1):
                if text and str(text).strip():
                    st.markdown(f"**{idx}.** {text}")
        
        # Tracker vs CSAT Relationship
        st.subheader("Tracker Satisfaction vs CSAT")
        st.markdown("*Examines how tracker satisfaction relates to overall CSAT scores.*")
        if tracker_questions and 'CSAT_numeric' in processed_df.columns:
            # Calculate average tracker score per customer
            tracker_cols = [col for _, col in tracker_questions if col in processed_df.columns]
            if tracker_cols:
                processed_df['Avg_Tracker_Score'] = processed_df[tracker_cols].mean(axis=1)
                both_values = processed_df[['Avg_Tracker_Score', 'CSAT_numeric']].dropna()
                if len(both_values) > 0:
                    correlation = both_values.corr().iloc[0, 1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Correlation Coefficient", f"{correlation:.3f}",
                                 help="Range: -1 to +1. Positive values indicate tracker satisfaction increases CSAT.")
                    with col2:
                        st.metric("Sample Size", f"{len(both_values):,}",
                                 help="Number of customers who answered both tracker questions and CSAT")
                    
                    # Create scatter plot with trend line
                    fig = go.Figure()
                    
                    # Sample data if too many points for better performance
                    if len(both_values) > 1000:
                        sample_size = 1000
                        both_values_sampled = both_values.sample(n=sample_size, random_state=42)
                    else:
                        both_values_sampled = both_values
                    
                    # Add scatter points
                    fig.add_trace(go.Scatter(
                        x=both_values_sampled['Avg_Tracker_Score'],
                        y=both_values_sampled['CSAT_numeric'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.5,
                            color='steelblue',
                            line=dict(width=0)
                        ),
                        name='Customer Responses',
                        hovertemplate='Tracker: %{x:.2f}<br>CSAT: %{y}<extra></extra>',
                        showlegend=False
                    ))
                    
                    # Add trend line
                    x_vals = both_values['Avg_Tracker_Score'].values
                    y_vals = both_values['CSAT_numeric'].values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                    line_x = np.linspace(x_vals.min(), x_vals.max(), 100)
                    line_y = slope * line_x + intercept
                    
                    fig.add_trace(go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='Trend: CSAT = %{y:.2f} for Tracker = %{x:.1f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Tracker Satisfaction vs Customer Satisfaction",
                        xaxis_title="Average Tracker Score (1-5)",
                        yaxis_title="CSAT Score (1 = Very Dissatisfied, 5 = Very Satisfied)",
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    if abs(correlation) > 0.7:
                        strength = "strong"
                    elif abs(correlation) > 0.4:
                        strength = "moderate"
                    else:
                        strength = "weak"
                    
                    direction = "positive" if correlation > 0 else "negative"
                    
                    st.info(f"**Interpretation:** There is a {strength} {direction} relationship (r={correlation:.3f}) between tracker satisfaction and CSAT. This suggests that better tracker experience {'strongly influences' if abs(correlation) > 0.7 else 'influences'} overall satisfaction.")
        
        # Tracker-related Sentiment Themes
        st.subheader("Tracker-Related Sentiment Themes")
        st.markdown("*Key themes in customer feedback related to tracking, updates, and reassurance.*")
        # Find which text column was actually analyzed
        text_col = None
        for col in ['Translation to English for: Order CSAT reason', 'Order CSAT reason']:
            if col in processed_df.columns:
                sentiment_col = f'{col}_sentiment'
                if sentiment_col in processed_df.columns:
                    if processed_df[sentiment_col].notna().sum() > 0:
                        text_col = col
                        break
        
        if text_col and text_col in processed_df.columns:
            # Filter themes to tracker-related ones
            theme_df = sa.get_top_themes(processed_df, text_col, n=15)
            if not theme_df.empty:
                # Filter for tracker-related themes
                tracker_themes = ['Tracker/Updates', 'Delivery Speed']
                tracker_theme_df = theme_df[theme_df['Theme'].isin(tracker_themes)]
                if tracker_theme_df.empty:
                    # If no exact matches, show top themes
                    tracker_theme_df = theme_df.head(5)
                
                if not tracker_theme_df.empty:
                    num_themes = len(tracker_theme_df)
                    # Use single color if more than 5 themes, otherwise use varied colors
                    if num_themes > 5:
                        bar_color = COLOR_PALETTE['primary'][0]  # Use primary blue
                    else:
                        colors = COLOR_PALETTE['primary'][:num_themes] if num_themes <= len(COLOR_PALETTE['primary']) else COLOR_PALETTE['primary'] * (num_themes // len(COLOR_PALETTE['primary']) + 1)
                        bar_color = colors[:num_themes] if num_themes <= 5 else COLOR_PALETTE['primary'][0]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=tracker_theme_df['Theme'],
                            y=tracker_theme_df['Count'],
                            marker=dict(
                                color=bar_color,
                                opacity=0.8
                            ),
                            text=tracker_theme_df['Count'],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Tracker-Related Themes in Feedback",
                        xaxis_title="Theme",
                        yaxis_title="Number of Mentions",
                        xaxis={'categoryorder': 'total descending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Order Tracker Analysis Summary
        st.subheader("Order Tracker Analysis Summary")
        st.markdown("*Key insights and findings from the Order Tracker Analysis section.*")
        tracker_summary = generate_tracker_summary(processed_df)
        for line in tracker_summary:
            st.markdown(line)
    
    except FileNotFoundError:
        st.error(f"Data file '{data_file}' not found. Please ensure the file is in the correct location.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)


# Streamlit runs the entire script, so call main() directly
# (not wrapped in if __name__ == "__main__" which doesn't work on Streamlit Cloud)
# Use a function wrapper to ensure Streamlit can start even if main() fails
def run_app():
    try:
        main()
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        st.info("Please ensure all required files are in the repository.")
        st.exception(e)
    except Exception as e:
        st.error(f"An error occurred while starting the dashboard: {str(e)}")
        st.exception(e)

# Call the wrapper function
run_app()

