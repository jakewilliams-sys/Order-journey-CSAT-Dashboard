# CSAT Survey Analysis Dashboard

A comprehensive interactive dashboard for analyzing Order Journey CSAT survey data with sentiment analysis, group comparisons, and key insights visualization.

## Features

- **Executive Summary**: Key metrics, insights, and sentiment overview
- **CSAT Analysis**: Distribution analysis and group comparisons
- **Sentiment Deep Dive**: Automated sentiment analysis with VADER, theme extraction, and sample quotes
- **Value Analysis**: Value perception metrics and factors influencing value
- **Tracker Analysis**: Order tracking satisfaction across 5 key questions
- **Drivers Analysis**: Performance of key drivers (temperature, portion size, quality, etc.)
- **Bill Shock & Mission**: Bill expectation analysis and order mission reasons
- **Group Comparisons**: Statistical comparisons across Grocery/Restaurant, Plus Customer, and Customer Segment

## Installation

1. Install Python 3.8 or higher

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for sentiment analysis):
```python
import nltk
nltk.download('vader_lexicon')
```

## Usage

1. Ensure your CSV data file is named `OJ CSAT Trial.csv` and is in the same directory as `dashboard.py`

2. Run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

3. The dashboard will open in your default web browser at `http://localhost:8501`

## Data Requirements

The dashboard expects a CSV file with the following key columns:

- `Order Journey CSAT`: 5-point scale CSAT score
- `Order CSAT reason`: Open-ended feedback text
- `Translation to English for: Order CSAT reason`: English translation of feedback
- `Bill expectation/shock`: Bill shock scale
- `Grocery / Restaurant order`: Order type classification
- `Plus Customer`: Plus customer status (Yes/No)
- `Customer Segment`: Customer segment classification
- `The right temperature`, `A good portion size`, `Presented well`, `Exactly what you ordered`, `Great quality`, `Quick & easy`: Driver questions (Yes/No/Not Sure)
- `I ordered deliveroo for…`: Mission multi-select
- `Thinking about your recent order, to what extent do you feel it was worth the money you spent?`: Value score (1-5)
- `Which factors most influenced whether your order felt worth the money?`: Value factors multi-select
- `I trusted the updates were accurate`, `I understood what was happening with my order`, `I felt reassured while I was waiting for my order to arrive`, `I had enough detail on my order progress`, `I was aware of my order updates through the order tracker or notifications`: Tracker questions (Strongly Disagree to Strongly Agree)
- `What, if anything, could make you feel more reassured about your order's progress?`: Open-ended tracker feedback

## Dashboard Features

### Filters
Use the sidebar filters to:
- Filter by Order Type (Grocery/Restaurant)
- Filter by Plus Customer status
- Filter by Customer Segment

### Tabs Overview

1. **Executive Summary**: High-level metrics, key insights, and sentiment summary
2. **CSAT Analysis**: Detailed CSAT distribution and group comparisons
3. **Sentiment Deep Dive**: Sentiment analysis, themes, and sample quotes
4. **Value Analysis**: Value perception scores and influencing factors
5. **Tracker Analysis**: Order tracking satisfaction metrics
6. **Drivers Analysis**: Performance of key experience drivers
7. **Bill Shock & Mission**: Bill expectation and order mission analysis
8. **Group Comparisons**: Statistical comparisons with significance testing

## Technical Details

### Sentiment Analysis
- Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment scoring
- Classifies feedback as Positive, Negative, or Neutral
- Extracts themes using keyword matching

### Statistical Analysis
- Performs t-tests for group comparisons
- Calculates correlations between key metrics
- Provides significance testing results

### Performance
- Data is cached using Streamlit's caching mechanism
- Optimized for datasets with ~44K+ responses
- Efficient processing of multi-select columns

## File Structure

```
.
├── dashboard.py              # Main Streamlit application
├── data_processing.py        # Data cleaning and preprocessing
├── sentiment_analysis.py     # Sentiment analysis functions
├── visualizations.py         # Visualization functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── OJ CSAT Trial.csv        # Survey data (user-provided)
```

## Dependencies

- pandas >= 2.0.0
- streamlit >= 1.28.0
- plotly >= 5.17.0
- nltk >= 3.8.1
- wordcloud >= 1.9.2
- scipy >= 1.11.0
- numpy >= 1.24.0
- vaderSentiment >= 3.3.2

## Troubleshooting

### Issue: "Data file not found"
- Ensure `OJ CSAT Trial.csv` is in the same directory as `dashboard.py`
- Check the file name matches exactly (case-sensitive)

### Issue: "Module not found"
- Run `pip install -r requirements.txt` to install all dependencies
- Ensure you're using the correct Python environment

### Issue: Slow loading
- The first load may take time as data is processed and cached
- Subsequent loads will be faster due to Streamlit caching

## License

This project is provided as-is for internal analysis purposes.

