import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import os
from datetime import datetime
import altair as alt
from utils.elo_scoring import ELOCalculator

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from utils.analysis import ClimbingAnalyzer
except ImportError:
    st.error("Could not import ClimbingAnalyzer. Please ensure utils/analysis.py exists in the correct location.")
    st.stop()

warnings.filterwarnings('ignore')

# Get colors from environment variables with defaults
DISCIPLINE_COLORS = {
    'Lead': os.getenv('LEAD_COLOR', "#C33727"), 
    'Boulder': os.getenv('BOULDER_COLOR', "#2777AC"), 
    'Speed': os.getenv('SPEED_COLOR', '#1C7C44'), 
    'Combined': os.getenv('COMBINED_COLOR', '#833F9E')
}

GENDER_COLORS = {
    'Men': os.getenv('MEN_COLOR', '#2C3E50'), 
    'Women': os.getenv('WOMEN_COLOR', '#E91E63')
}

CHART_COLORS = [
    os.getenv('CHART_COLOR_1', "#D24130"),
    os.getenv('CHART_COLOR_2', "#2885C3"),
    os.getenv('CHART_COLOR_3', "#1C7C44"),
    os.getenv('CHART_COLOR_4', "#A156BE"),
    os.getenv('CHART_COLOR_5', "#D98B0F"),
    os.getenv('CHART_COLOR_6', '#1ABC9C'),
    os.getenv('CHART_COLOR_7', '#E67E22'),
    os.getenv('CHART_COLOR_8', '#95A5A6'),
    os.getenv('CHART_COLOR_9', '#34495E'),
    os.getenv('CHART_COLOR_10', '#8E44AD')
]

# Page configuration
st.set_page_config(
    page_title="Climbing Competition Analysis",
    page_icon="🧗‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
}
.participation-trends {
    margin-top: 3rem;
    margin-bottom: 3rem;
}
.filter-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.button-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

/* Segmented Control Styles */
.segmented-control {
    display: flex;
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    overflow: hidden;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.segment {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 12px 8px;
    background: #f8f9fa;
    border-right: 1px solid #e0e0e0;
    cursor: pointer;
    transition: all 0.2s ease;
    min-height: 70px;
}

.segment:last-child {
    border-right: none;
}

.segment:hover {
    background: #e9ecef;
}

.segment.active {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
}

.segment-name {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 4px;
}

.segment-years {
    font-size: 11px;
    opacity: 0.8;
}

/* Radio button styling */
.stRadio > div {
    flex-direction: row !important;
}

.stRadio label {
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-right: 2rem !important;
}

.stRadio > div > label > div {
    font-size: 24px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# Country emoji mapping
COUNTRY_FLAGS = {
    'USA': '🇺🇸', 'FRA': '🇫🇷', 'GER': '🇩🇪', 'JPN': '🇯🇵', 'GBR': '🇬🇧',
    'AUT': '🇦🇹', 'CAN': '🇨🇦', 'ITA': '🇮🇹', 'RUS': '🇷🇺', 'CHE': '🇨🇭',
    'ESP': '🇪🇸', 'BEL': '🇧🇪', 'NED': '🇳🇱', 'POL': '🇵🇱', 'CZE': '🇨🇿',
    'SLO': '🇸🇮', 'KOR': '🇰🇷', 'AUS': '🇦🇺', 'NOR': '🇳🇴', 'SWE': '🇸🇪'
}

@st.cache_data
def load_analyzer():
    """Load the climbing analyzer with caching."""
    try:
        analyzer = ClimbingAnalyzer()
        return analyzer
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please check that the DATA folder exists with the correct structure")
        return None

def filter_disciplines(overview):
    """Filter out Boulder&Lead and Combined disciplines"""
    if overview.get('disciplines'):
        filtered_disciplines = {k: v for k, v in overview['disciplines'].items() 
                              if k not in ['Boulder&Lead', 'Combined']}
        overview['disciplines'] = filtered_disciplines
    return overview

def add_country_flags(df, country_col='country'):
    """Add emoji flags to country names"""
    if country_col in df.columns:
        df[country_col] = df[country_col].apply(lambda x: f"{COUNTRY_FLAGS.get(x, '')} {x}")
    return df

def create_athlete_filter_buttons(discipline_default='Boulder', gender_default='Men'):
    """Create discipline and gender filter buttons for athlete analysis."""
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    
    # Discipline selection
    st.markdown("### Select Discipline")
    disciplines = ['Boulder', 'Lead', 'Speed']
    col1, col2, col3 = st.columns(3)
    
    selected_discipline = discipline_default
    
    with col1:
        if st.button(f"🧗 Boulder", key="athlete_boulder", width='stretch', 
                    type="primary" if discipline_default == "Boulder" else "secondary"):
            selected_discipline = "Boulder"
    
    with col2:
        if st.button(f"🪨 Lead", key="athlete_lead", width='stretch',
                    type="primary" if discipline_default == "Lead" else "secondary"):
            selected_discipline = "Lead"
    
    with col3:
        if st.button(f"⚡ Speed", key="athlete_speed", width='stretch',
                    type="primary" if discipline_default == "Speed" else "secondary"):
            selected_discipline = "Speed"
    
    # Gender selection  
    st.markdown("### Select Gender")
    col1, col2 = st.columns(2)
    
    selected_gender = gender_default
    
    with col1:
        if st.button(f"👨 Men", key="athlete_men", width='stretch',
                    type="primary" if gender_default == "Men" else "secondary"):
            selected_gender = "Men"
    
    with col2:
        if st.button(f"👩 Women", key="athlete_women", width='stretch',
                    type="primary" if gender_default == "Women" else "secondary"):
            selected_gender = "Women"
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return selected_discipline, selected_gender

def create_country_charts(country_stats, metric='total_wins'):
    """Create visual charts for country statistics instead of tables."""
    if country_stats.empty:
        return None
    
    # Get top 15 countries
    if metric == 'total_wins':
        top_countries = country_stats.nlargest(15, 'total_wins')
        title = "🏆 Most Successful Countries (Total Wins)"
        y_label = "Total Wins"
        color_col = 'total_wins'
        color_scale = 'Reds'
    elif metric == 'total_podiums':
        top_countries = country_stats.nlargest(15, 'total_podiums')
        title = "🥉 Most Successful Countries (Total Podiums)"
        y_label = "Total Podiums"
        color_col = 'total_podiums'
        color_scale = 'Blues'
    else:
        top_countries = country_stats.nlargest(15, 'total_athletes')
        title = "👥 Most Active Countries (Total Athletes)"
        y_label = "Total Athletes"
        color_col = 'total_athletes'
        color_scale = 'Greens'
    
    fig = px.bar(
        top_countries,
        x=color_col,
        y='country',
        orientation='h',
        title=title,
        labels={color_col: y_label, 'country': 'Country'},
        color=color_col,
        color_continuous_scale=color_scale,
        text=color_col
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        showlegend=False
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🧗‍♂️ Climbing Competition Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data with error handling
    analyzer = load_analyzer()
    if analyzer is None:
        st.error("Failed to load data. Please check your data files and folder structure.")
        st.markdown("""
        Expected folder structure:
        ```
        DATA/
        ├── data_summary/
        │   ├── data_summary.csv
        │   ├── era_statistics.csv
        │   ├── event_metadata.csv
        │   └── era_files_summary.csv
        └── aggregate_data/
            ├── aggregated_results.csv
            └── [era-specific files].csv
        ```
        """)
        return
    
    overview = analyzer.get_data_overview()
    
    if not overview:
        st.warning("No data found. Please check your data files.")
        return
    
    # Filter out unwanted disciplines
    overview = filter_disciplines(overview)
    
    available_disciplines = overview['disciplines'].keys()
    
    filters = {
        'year_range': overview.get('year_range'),
        'disciplines': available_disciplines,
        'genders': list(overview.get('genders', {}).keys()) if overview.get('genders') else [],
        'countries': None,
        'rounds': list(overview.get('rounds', {}).keys()) if overview.get('rounds') else []
    }
    
    # Main content tabs - ELO first
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 ELO Rankings", 
        "📈 Overview", 
        "🏆 Athletes", 
        "🌍 Countries", 
        "🔍 Deep Dive"
    ])
    
    with tab1:
        
        # Initialize session state
        if 'selected_elo_discipline' not in st.session_state:
            st.session_state.selected_elo_discipline = 'Lead'
        if 'selected_elo_gender' not in st.session_state:
            st.session_state.selected_elo_gender = 'Men'
        if 'selected_elo_era' not in st.session_state:
            st.session_state.selected_elo_era = None
        if 'selected_era_idx' not in st.session_state:
            st.session_state.selected_era_idx = 0
        
        disciplines = ['Boulder', 'Lead', 'Speed']
        genders = ['Men', 'Women']
        
        # Updated discipline eras starting from 2007 only
        DISCIPLINE_ERAS = {
            "Lead": [
                ("IFSC_Modern", 2007, 2025),
                ],
            "Boulder": [
                ("IFSC_ZoneTop", 2007, 2024),
                ("IFSC_AddedPoints", 2025, 2025),
                ],
            "Speed": [
                ("IFSC_Score", 2007, 2008),
                ("IFSC_Time", 2009, 2025),
                ]
            }

        col_left, col_right = st.columns([1, 2.2])

        with col_left:
            # Discipline selection with larger font
            st.markdown("### Discipline")
            discipline_option = st.radio(
                "", 
                ["🧗 Lead", "🪨 Boulder", "⚡ Speed"],
                key="discipline_radio",
                horizontal=True,
                label_visibility="collapsed"
            )
            selected_discipline = discipline_option.split()[1]  # Extract discipline name
            st.session_state.selected_elo_discipline = selected_discipline

            # Gender selection with larger font
            st.markdown("### Gender")
            gender_option = st.radio(
                "", 
                ["👨 Men", "👩 Women"], 
                key="gender_radio",
                horizontal=True,
                label_visibility="collapsed"
            )
            selected_gender = gender_option.split()[1]  # Extract gender name
            st.session_state.selected_elo_gender = selected_gender
            
            eras = DISCIPLINE_ERAS.get(selected_discipline, [])
            
            if eras:
                st.markdown("**Choose Period:**")
                era_cols = st.columns(len(eras))
                
                for i, (era_name, start_year, end_year) in enumerate(eras):
                    with era_cols[i]:
                        if st.button(
                            f"{era_name.split('_')[0]}\n\n{era_name.split('_')[1]}\n\n{start_year}–{end_year}", 
                            key=f"era_{i}_{selected_discipline}",
                            width='stretch',
                            type="primary" if st.session_state.selected_era_idx == i else "secondary"
                        ):
                            st.session_state.selected_era_idx = i
                            st.session_state.selected_elo_era = f"{selected_discipline.lower()}_{start_year}_{end_year}"
            
            # Show current selection
            if eras and st.session_state.selected_era_idx < len(eras):
                era_name, start_year, end_year = eras[st.session_state.selected_era_idx]
                st.info(f"**Current Selection:** \n\n {selected_discipline} | {selected_gender} | {era_name} | {start_year}–{end_year}")

        with col_right:
            if eras and st.session_state.selected_era_idx < len(eras):
                era_name, start_year, end_year = eras[st.session_state.selected_era_idx]
            
                # Use the correct era key format
                era_key = f"{selected_discipline}_{era_name}_{start_year}-{end_year}_{selected_gender}"
                
                # Calculate ELO for selected criteria
                with st.spinner("Calculating ELO ratings..."):
                    try:
                        # Initialize ELO calculator
                        elo_calc = ELOCalculator(analyzer.era_files)
                        
                        leaderboard = elo_calc.get_current_leaderboard(era_key, discipline=selected_discipline, gender=selected_gender)
                        
                        if not leaderboard.empty:
                            # Top 5 athletes historical ELO chart
                            st.markdown("### Top 5 Athletes - Historical ELO")
                            top_5 = leaderboard.head(5)
                            
                            # Create historical chart
                            historical_fig = elo_calc.create_historical_elo_chart(top_5['name'].tolist(), era_key, discipline=selected_discipline, gender=selected_gender)
                            
                            if historical_fig and historical_fig.data:
                                st.plotly_chart(historical_fig, width='stretch')
                            else:
                                st.info("No ELO history data available for visualization")
                            
                            # Current ELO Leaderboard - Top 10 in right column
                            st.markdown("### Current ELO Leaderboard - Top 10")
                            top_elo = leaderboard.head(10)[['name', 'current_elo', 'peak_elo', 'competitions', 'avg_rank']]
                            top_elo.columns = ['Athlete', 'Current ELO', 'Peak ELO', 'Competitions', 'Avg Rank']
                            # Format the numeric columns
                            top_elo['Current ELO'] = top_elo['Current ELO'].round(0).astype(int)
                            top_elo['Peak ELO'] = top_elo['Peak ELO'].round(0).astype(int)
                            top_elo['Competitions'] = top_elo['Competitions'].astype(int)
                            top_elo['Avg Rank'] = top_elo['Avg Rank'].round(1)
                            st.dataframe(top_elo, width='stretch', hide_index=True)
                            
                        else:
                            st.warning(f"No ELO data available for {era_key}")
                            st.write("This could be due to:")
                            st.write("- No competition data for this discipline/gender/era combination")
                            st.write("- Data formatting issues in the source files")
                            st.write("- Missing era-specific data files")
                            
                    except Exception as e:
                        st.error(f"Error calculating ELO: {str(e)}")

    with tab2:
        st.markdown("## 📈 Competition Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records",
                f"{overview.get('total_records', 0):,}",
                help="Total competition results in database"
            )
        
        with col2:
            st.metric(
                "Unique Athletes",
                f"{overview.get('unique_athletes', 0):,}",
                help="Total number of different athletes"
            )
        
        with col3:
            st.metric(
                "Countries",
                f"{overview.get('unique_countries', 0):,}",
                help="Countries represented in competitions"
            )
        
        with col4:
            if overview.get('year_range'):
                year_span = overview['year_range'][1] - overview['year_range'][0] + 1
                st.metric(
                    "Years Covered",
                    f"{year_span}",
                    help=f"From {overview['year_range'][0]} to {overview['year_range'][1]}"
                )
        
        # Discipline and gender distribution with consistent colors
        col1, col2 = st.columns(2)
        
        with col1:
            if overview.get('disciplines'):
                colors = [DISCIPLINE_COLORS.get(disc, '#95A5A6') for disc in overview['disciplines'].keys()]
                fig = px.pie(
                    values=list(overview['disciplines'].values()),
                    names=list(overview['disciplines'].keys()),
                    title="Distribution by Discipline",
                    color_discrete_sequence=colors
                )
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            if overview.get('genders'):
                colors = [GENDER_COLORS.get(gender, '#95A5A6') for gender in overview['genders'].keys()]
                fig = px.pie(
                    values=list(overview['genders'].values()),
                    names=list(overview['genders'].keys()),
                    title="Distribution by Gender",
                    color_discrete_sequence=colors
                )
                st.plotly_chart(fig, width='stretch')
        
        # Participation trends
        st.markdown('<div class="participation-trends">', unsafe_allow_html=True)
        st.markdown("### 📈 Participation Trends Over Time")
        trends_fig = analyzer.create_participation_trends(filters)
        if trends_fig and trends_fig.data:
            st.plotly_chart(trends_fig, width='stretch')
        else:
            st.info("No data available for current filters")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## 🏆 Athlete Analysis")
        
        # Initialize session state for selections
        if 'selected_athlete_discipline' not in st.session_state:
            st.session_state.selected_athlete_discipline = 'Boulder'
        if 'selected_athlete_gender' not in st.session_state:
            st.session_state.selected_athlete_gender = 'Men'
        
        # Filter controls
        selected_discipline, selected_gender = create_athlete_filter_buttons(
            st.session_state.selected_athlete_discipline,
            st.session_state.selected_athlete_gender
        )
        
        # Update session state
        st.session_state.selected_athlete_discipline = selected_discipline
        st.session_state.selected_athlete_gender = selected_gender
        
        # Create filtered data
        athlete_filters = {
            'disciplines': [selected_discipline],
            'genders': [selected_gender],
            'year_range': filters['year_range'],
            'rounds': filters['rounds']
        }
        
        # Get athlete statistics
        athlete_stats = analyzer.get_athlete_stats(athlete_filters)
        
        if not athlete_stats.empty:
            st.markdown(f"**Showing:** {selected_discipline} - {selected_gender}")
            
            # Top performers charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🥇 Most Successful Athletes (Wins)")
                top_winners = athlete_stats.nlargest(10, 'wins')
                if not top_winners.empty:
                    fig = px.bar(
                        top_winners,
                        x='wins',
                        y='name',
                        orientation='h',
                        title=f'Top 10 {selected_discipline} {selected_gender} by Wins',
                        labels={'wins': 'Number of Wins', 'name': 'Athlete'},
                        color='wins',
                        color_continuous_scale='Reds',
                        text='wins'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No wins data available for current selection")
            
            with col2:
                st.markdown("### 🏅 Most Successful Athletes (Podiums)")
                top_podiums = athlete_stats.nlargest(10, 'podiums')
                if not top_podiums.empty:
                    fig = px.bar(
                        top_podiums,
                        x='podiums',
                        y='name',
                        orientation='h',
                        title=f'Top 10 {selected_discipline} {selected_gender} by Podiums',
                        labels={'podiums': 'Number of Podiums', 'name': 'Athlete'},
                        color='podiums',
                        color_continuous_scale='Blues',
                        text='podiums'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No podium data available for current selection")
            
            # Most active athletes chart
            st.markdown("### 🏃‍♀️ Most Active Athletes")
            active_athletes = athlete_stats.nlargest(10, 'total_competitions')
            if not active_athletes.empty:
                fig = px.bar(
                    active_athletes,
                    x='total_competitions',
                    y='name',
                    orientation='h',
                    title=f'Top 10 Most Active {selected_discipline} {selected_gender}',
                    labels={'total_competitions': 'Number of Competitions', 'name': 'Athlete'},
                    color='total_competitions',
                    color_continuous_scale='Greens',
                    text='total_competitions'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No activity data available for current selection")
            
            # Individual athlete analysis
            st.markdown("### 🎯 Individual Athlete Deep Dive")
            
            # Athlete selector
            athlete_names = sorted(athlete_stats['name'].unique())
            selected_athlete = st.selectbox("Select Athlete", athlete_names)
            
            if selected_athlete:
                # Athlete timeline
                timeline_fig = analyzer.create_athlete_timeline(selected_athlete)
                if timeline_fig and timeline_fig.data:
                    st.plotly_chart(timeline_fig, width='stretch')
                
                # Athlete detailed stats
                athlete_detail = athlete_stats[athlete_stats['name'] == selected_athlete].iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Competitions", int(athlete_detail['total_competitions']))
                with col2:
                    st.metric("Wins", int(athlete_detail['wins']))
                with col3:
                    st.metric("Podiums", int(athlete_detail['podiums']))
                with col4:
                    st.metric("Win Rate", f"{athlete_detail['win_rate']:.1f}%")
        
        else:
            st.warning("No athlete data available for current selection")
    
    with tab4:
        st.markdown("## 🌍 Country Analysis")
        
        # Country statistics
        country_stats = analyzer.get_country_stats(filters)
        
        if not country_stats.empty:
            # Visual charts for country performance
            col1, col2 = st.columns(2)
            
            with col1:
                wins_chart = create_country_charts(country_stats, 'total_wins')
                if wins_chart:
                    st.plotly_chart(wins_chart, width='stretch')
            
            with col2:
                podiums_chart = create_country_charts(country_stats, 'total_podiums')
                if podiums_chart:
                    st.plotly_chart(podiums_chart, width='stretch')
            
            # Most active countries chart
            active_chart = create_country_charts(country_stats, 'total_athletes')
            if active_chart:
                st.plotly_chart(active_chart, width='stretch')
            
            # Performance heatmaps by discipline
            st.markdown("### 🗺️ Performance by Country and Discipline")
            
            col1, col2 = st.columns(2)
            
            with col1:
                wins_heatmap = analyzer.create_country_discipline_heatmap(filters, metric='wins')
                if wins_heatmap and wins_heatmap.data:
                    st.plotly_chart(wins_heatmap, width='stretch')
            
            with col2:
                podiums_heatmap = analyzer.create_country_discipline_heatmap(filters, metric='podiums')
                if podiums_heatmap and podiums_heatmap.data:
                    st.plotly_chart(podiums_heatmap, width='stretch')
                    
        else:
            st.warning("No country data available for current filters")
    
    with tab5:
        st.markdown("## 🔍 Deep Dive Analysis")
        
        analysis_options = [
            "Yearly Trends",
            "Career Progression", 
            "Head-to-Head Comparison",
            "Location Performance",
            "Discipline Crossover"
        ]
        
        selected_analysis = st.selectbox("Select Analysis Type", analysis_options)
        
        st.info(f"🚧 {selected_analysis} analysis coming soon! We're working on bringing you detailed insights.")
        
        # Placeholder for future features
        st.markdown("""
        ### Planned Features:
        - **Yearly Trends**: Track how climbing performance and participation evolved over time
        - **Career Progression**: Analyze individual athlete development and peak performance periods  
        - **Head-to-Head Comparison**: Compare two athletes across multiple metrics and competitions
        - **Location Performance**: See how different venues and countries affect performance
        - **Discipline Crossover**: Analyze athletes who compete across multiple climbing disciplines
        """)

if __name__ == "__main__":
    main()