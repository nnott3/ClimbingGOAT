# Climbing Competition Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing climbing competition data with ELO rankings, athlete performance tracking, and country statistics.

## Key Fixes Applied

1. **Environment Variable Colors**: All colors are now configurable via environment variables
2. **ELO Rankings First**: The ELO Rankings tab is now the first tab as requested
3. **Improved Error Handling**: Better error handling for missing data and files
4. **Streamlit Cloud Compatibility**: Optimized for deployment on Streamlit Cloud

## Environment Variables

Set these environment variables in your Streamlit Cloud deployment to customize colors:

### Discipline Colors
- `BOULDER_COLOR` (default: `#3498DB`)
- `LEAD_COLOR` (default: `#E74C3C`)
- `SPEED_COLOR` (default: `#27AE60`)
- `COMBINED_COLOR` (default: `#9B59B6`)

### Gender Colors
- `MEN_COLOR` (default: `#2C3E50`)
- `WOMEN_COLOR` (default: `#E91E63`)

### Chart Colors
- `CHART_COLOR_1` through `CHART_COLOR_10` (defaults provided)

## File Structure

```
project/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── utils/
│   ├── __init__.py
│   ├── analysis.py       # Main analysis class
│   └── elo_scoring.py    # ELO calculation utilities
└── DATA/
    ├── data_summary/
    │   ├── data_summary.csv
    │   ├── era_statistics.csv
    │   ├── event_metadata.csv
    │   └── era_files_summary.csv
    └── raw_data/
        ├── aggregated_results.csv
        └── [era-specific files].csv
```

## Deployment Steps

### 1. Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Streamlit Cloud Deployment

1. **Repository Setup**:
   - Push all files to a GitHub repository
   - Include the DATA folder with your climbing data

2. **Environment Variables** (in Streamlit Cloud settings):
   ```
   BOULDER_COLOR=#3498DB
   LEAD_COLOR=#E74C3C
   SPEED_COLOR=#27AE60
   MEN_COLOR=#2C3E50
   WOMEN_COLOR=#E91E63
   ```

3. **Deploy**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Add environment variables in Advanced Settings

## Features

### 🎯 ELO Rankings (Primary Tab)
- Interactive discipline and gender selection
- Historical ELO progression charts
- Current leaderboards
- ELO rating distributions

### 📈 Overview
- Competition statistics
- Participation trends over time
- Discipline and gender distributions

### 🏆 Athletes
- Performance analysis by discipline/gender
- Individual athlete timelines
- Win rates and statistics

### 🌍 Countries
- Country performance rankings
- Performance heatmaps by discipline
- Activity statistics

### 🔍 Deep Dive
- Placeholder for advanced analytics
- Future feature development

## Data Requirements

The application expects CSV files with the following columns:
- `name`: Athlete name
- `country`: Country code
- `discipline`: Boulder, Lead, or Speed
- `gender`: Men or Women
- `round_rank`: Competition rank
- `year`: Competition year
- `event_name`: Competition name
- `location`: Competition location
- `start_date`: Competition date
- `round`: Competition round (Qualification, Final, etc.)

## Troubleshooting

1. **Data Not Loading**: Check that the DATA folder structure matches the expected format
2. **Color Issues**: Verify environment variables are set correctly
3. **Performance Issues**: Consider reducing the dataset size for faster loading

## Color Customization

To change colors after deployment:
1. Go to your Streamlit Cloud app settings
2. Navigate to "Advanced settings"
3. Add or modify environment variables
4. Redeploy the app

The color system allows easy theme changes without code modifications.