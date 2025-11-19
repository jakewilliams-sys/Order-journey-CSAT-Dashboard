"""
Sentiment analysis functions for open-ended survey responses.
"""
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List, Tuple
import nltk

# Initialize VADER sentiment analyzer lazily
_analyzer = None

def get_analyzer():
    """Get or create the VADER sentiment analyzer."""
    global _analyzer
    if _analyzer is None:
        # Ensure NLTK data is downloaded
        try:
            nltk.download('vader_lexicon', quiet=True)
        except Exception:
            try:
                nltk.download('vader_lexicon')
            except Exception:
                pass
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def get_sentiment_score(text: str) -> Dict[str, float]:
    """Get sentiment scores for a text using VADER."""
    if pd.isna(text) or text == '':
        return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    try:
        analyzer = get_analyzer()
        return analyzer.polarity_scores(str(text))
    except Exception:
        # If sentiment analysis fails, return neutral scores
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}


def classify_sentiment(compound_score: float) -> str:
    """Classify sentiment based on compound score."""
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_sentiment(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Analyze sentiment for a text column."""
    result_df = df.copy()
    
    # Get sentiment scores
    sentiment_scores = result_df[text_column].apply(get_sentiment_score)
    
    # Extract individual scores
    result_df[f'{text_column}_compound'] = sentiment_scores.apply(lambda x: x['compound'])
    result_df[f'{text_column}_pos'] = sentiment_scores.apply(lambda x: x['pos'])
    result_df[f'{text_column}_neu'] = sentiment_scores.apply(lambda x: x['neu'])
    result_df[f'{text_column}_neg'] = sentiment_scores.apply(lambda x: x['neg'])
    result_df[f'{text_column}_sentiment'] = result_df[f'{text_column}_compound'].apply(
        classify_sentiment
    )
    
    return result_df


def extract_keywords(text: str, keywords: List[str]) -> List[str]:
    """Extract keywords from text."""
    if pd.isna(text) or text == '':
        return []
    
    text_lower = str(text).lower()
    found_keywords = []
    for keyword in keywords:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    return found_keywords


def extract_themes(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Extract themes from text using keyword matching."""
    result_df = df.copy()
    
    # Define theme keywords
    themes = {
        'Food Quality Issues': ['cold', 'warm', 'hot', 'temperature', 'fresh', 'freshness', 
                               'quality', 'taste', 'flavor', 'flavour', 'tasty', 'delicious'],
        'Missing Items': ['missing', 'didn\'t get', 'did not get', 'forgot', 'left out', 
                         'omitted', 'absent'],
        'Wrong Order': ['wrong', 'incorrect', 'mistake', 'error', 'not what i ordered', 
                       'different', 'substituted'],
        'Delivery Speed': ['quick', 'fast', 'slow', 'delayed', 'late', 'time', 'speed', 
                          'minutes', 'hours', 'waiting'],
        'Value Perception': ['worth', 'value', 'expensive', 'cheap', 'price', 'cost', 
                           'money', 'paid', 'refund'],
        'Tracker/Updates': ['update', 'tracker', 'tracking', 'notification', 'informed', 
                           'reassured', 'progress', 'status'],
        'Rider Experience': ['driver', 'rider', 'delivery person', 'friendly', 'polite', 
                            'attitude', 'communication'],
        'Packaging': ['packaging', 'packed', 'container', 'bag', 'box', 'wrapped']
    }
    
    # Extract themes for each row
    for theme_name, keywords in themes.items():
        result_df[f'{text_column}_theme_{theme_name}'] = result_df[text_column].apply(
            lambda x: 1 if any(kw in str(x).lower() for kw in keywords) else 0
        )
    
    return result_df


def get_sentiment_summary(df: pd.DataFrame, text_column: str) -> Dict:
    """Get summary statistics for sentiment analysis."""
    sentiment_col = f'{text_column}_sentiment'
    
    if sentiment_col not in df.columns:
        return {}
    
    total = len(df[df[text_column].notna() & (df[text_column] != '')])
    
    if total == 0:
        return {}
    
    summary = {
        'total_responses': total,
        'positive_count': len(df[df[sentiment_col] == 'Positive']),
        'negative_count': len(df[df[sentiment_col] == 'Negative']),
        'neutral_count': len(df[df[sentiment_col] == 'Neutral']),
        'positive_pct': len(df[df[sentiment_col] == 'Positive']) / total * 100,
        'negative_pct': len(df[df[sentiment_col] == 'Negative']) / total * 100,
        'neutral_pct': len(df[df[sentiment_col] == 'Neutral']) / total * 100,
        'avg_compound': df[f'{text_column}_compound'].mean()
    }
    
    return summary


def get_top_themes(df: pd.DataFrame, text_column: str, n: int = 10) -> pd.DataFrame:
    """Get top themes from text analysis."""
    theme_cols = [col for col in df.columns if col.startswith(f'{text_column}_theme_')]
    
    if not theme_cols:
        return pd.DataFrame()
    
    theme_counts = {}
    for col in theme_cols:
        theme_name = col.replace(f'{text_column}_theme_', '')
        theme_counts[theme_name] = df[col].sum()
    
    theme_df = pd.DataFrame({
        'Theme': list(theme_counts.keys()),
        'Count': list(theme_counts.values())
    }).sort_values('Count', ascending=False).head(n)
    
    return theme_df

