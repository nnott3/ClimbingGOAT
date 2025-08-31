

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
        st.markdown("## 🎯 ELO Rankings")
        
        # ELO system selection with buttons
        era_files = overview.get('era_files', [])

        if era_files:
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
            
            DISCIPLINE_ERAS = {
                "Lead": [
                    ("UIAA_Legacy", 1991, 2006),
                    ("IFSC_Modern", 2007, 2025),
                    ],
                "Boulder": [
                    ("UIAA_Legacy", 1991, 2006),
                    ("IFSC_ZoneTop", 2007, 2024),
                    ("IFSC_AddedPoints", 2025, 2025),
                    ],
                "Speed": [
                    ("UIAA_Legacy", 1991, 2006),
                    ("IFSC_Score", 2007, 2008),
                    ("IFSC_ShortestTime", 2009, 2025),
                    ]
                }

            col_left, col_right = st.columns([1, 2.2])

            with col_left:
                # Discipline selection with larger font
                st.radio(
                    "Discipline", 
                    ["🧗 Lead", "🪨 Boulder", "⚡ Speed"],
                    key="discipline_radio",
                    horizontal=True
                )
                selected_discipline = st.session_state.discipline_radio.split()[1]  # Extract discipline name
                st.session_state.selected_elo_discipline = selected_discipline

                # Gender selection with larger font
                st.radio(
                    "Gender", 
                    ["👨 Men", "👩 Women"], 
                    key="gender_radio",
                    horizontal=True
                )
                selected_gender = st.session_state.gender_radio.split()[1]  # Extract gender name
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
                                use_container_width=True,
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
                

                    # Map era selection to actual era files
                    era_key = f"{selected_discipline}_{era_name}_{start_year}-{end_year}_{selected_gender}"
                    
                    # Calculate ELO for selected criteria
                    with st.spinner("Calculating ELO ratings..."):
                        try:
                            leaderboard = analyzer.get_elo_leaderboard(era_key, discipline=selected_discipline, gender=selected_gender)
                            
                            if not leaderboard.empty:
                                # Top 5 athletes historical ELO chart
                                st.markdown("### Top 5 Athletes - Historical ELO")
                                top_5 = leaderboard.head(5)
                                
                                historical_fig = analyzer.create_historical_elo_chart(top_5['name'].tolist(), era_key)
                                if historical_fig and historical_fig.data:
                                    st.plotly_chart(historical_fig, use_container_width=True)
                                else:
                                    st.info("No ELO history data available for visualization")
                                
                                # Current ELO Leaderboard - Top 10 in right column
                                st.markdown("### Current ELO Leaderboard - Top 10")
                                top_elo = leaderboard.head(10)[['name', 'current_elo', 'peak_elo', 'competitions', 'avg_rank']]
                                top_elo.columns = ['Athlete', 'Current ELO', 'Peak ELO', 'Competitions', 'Avg Rank']
                                st.dataframe(top_elo, use_container_width=True, hide_index=True)
                                
                            else:
                                st.warning("No ELO data available for selected criteria")
                                
                        except Exception as e:
                            st.error(f"Error calculating ELO: {str(e)}")
                            # Fallback to sample chart
                            st.markdown("### Sample ELO Chart (Demo)")
                            years = np.arange(1991, 2026)
                            athletes_data = []
                            for i, athlete in enumerate(['Sample A', 'Sample B', 'Sample C']):
                                base_elo = 1400 + i * 50
                                noise = np.random.RandomState(i).randn(len(years)) * 30
                                trend = (years - 1991) * (1 + i * 0.5) + noise
                                elo_values = base_elo + trend.cumsum()
                                
                                for year, elo in zip(years, elo_values):
                                    athletes_data.append({'Year': year, 'ELO': elo, 'Athlete': athlete})

                            df = pd.DataFrame(athletes_data)
                            chart = (
                                alt.Chart(df)
                                .mark_line(point=True, strokeWidth=3)
                                .encode(
                                    x=alt.X('Year:O', title='Year'),
                                    y=alt.Y('ELO:Q', title='ELO Rating'),
                                    color=alt.Color('Athlete:N', title='Athlete'),
                                    tooltip=['Year', 'ELO', 'Athlete']
                                )
                                .interactive()
                            )
                            st.altair_chart(chart, use_container_width=True)

            # Additional ELO Analysis Features (Full Width)
            if eras and st.session_state.selected_era_idx < len(eras):
                era_name, start_year, end_year = eras[st.session_state.selected_era_idx]
                era_key = f"{selected_discipline.lower()}_{start_year}_{end_year}"
                
                try:
                    leaderboard = analyzer.get_elo_leaderboard(era_key, discipline=selected_discipline, gender=selected_gender)
                    
                    if not leaderboard.empty:
                        # Dropdown to add more athletes to chart
                        st.markdown("### Add More Athletes to Chart")
                        all_athletes = sorted(leaderboard['name'].tolist())
                        top_5_names = leaderboard.head(5)['name'].tolist()
                        
                        additional_athletes = st.multiselect(
                            "Select additional athletes to add to the chart:",
                            [athlete for athlete in all_athletes if athlete not in top_5_names],
                            max_selections=5,
                            help="Add up to 5 more athletes to compare with the top 5"
                        )
                        
                        if additional_athletes:
                            combined_athletes = top_5_names + additional_athletes
                            combined_fig = analyzer.create_historical_elo_chart(combined_athletes, era_key)
                            if combined_fig and combined_fig.data:
                                st.plotly_chart(combined_fig, use_container_width=True)
                        
                        # Extended ELO Leaderboard
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("### Complete ELO Leaderboard")
                            
                            # Show more athletes option
                            show_count = st.selectbox(
                                "Number of athletes to display:",
                                [20, 50, 100, "All"],
                                index=0
                            )
                            
                            if show_count == "All":
                                display_leaderboard = leaderboard
                            else:
                                display_leaderboard = leaderboard.head(show_count)
                            
                            # Format the leaderboard for display
                            display_df = display_leaderboard[['name', 'current_elo', 'peak_elo', 'competitions', 'avg_rank']].copy()
                            display_df.columns = ['Athlete', 'Current ELO', 'Peak ELO', 'Competitions', 'Avg Rank']
                            display_df['Current ELO'] = display_df['Current ELO'].round(0).astype(int)
                            display_df['Peak ELO'] = display_df['Peak ELO'].round(0).astype(int)
                            display_df['Competitions'] = display_df['Competitions'].astype(int)
                            display_df['Avg Rank'] = display_df['Avg Rank'].round(1)
                            
                            # Add rank column
                            display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
                            
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            st.markdown("### ELO Statistics")
                            
                            # Summary statistics
                            avg_elo = leaderboard['current_elo'].mean()
                            median_elo = leaderboard['current_elo'].median()
                            max_elo = leaderboard['current_elo'].max()
                            min_elo = leaderboard['current_elo'].min()
                            
                            st.metric("Average ELO", f"{avg_elo:.0f}")
                            st.metric("Median ELO", f"{median_elo:.0f}")
                            st.metric("Highest ELO", f"{max_elo:.0f}")
                            st.metric("Lowest ELO", f"{min_elo:.0f}")
                            
                            # Most active athlete
                            most_active = leaderboard.loc[leaderboard['competitions'].idxmax()]
                            st.metric(
                                "Most Active Athlete", 
                                most_active['name'], 
                                delta=f"{most_active['competitions']} competitions"
                            )
                        
                        # ELO Distribution Chart
                        st.markdown("### ELO Rating Distribution")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram
                            fig_hist = px.histogram(
                                leaderboard,
                                x='current_elo',
                                nbins=25,
                                title=f'ELO Distribution - {selected_discipline} {selected_gender}',
                                labels={'current_elo': 'Current ELO Rating', 'count': 'Number of Athletes'},
                                color_discrete_sequence=[DISCIPLINE_COLORS.get(selected_discipline, '#3498DB')]
                            )
                            fig_hist.update_layout(showlegend=False)
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Box plot for ELO ranges
                            fig_box = px.box(
                                leaderboard,
                                y='current_elo',
                                title=f'ELO Range - {selected_discipline} {selected_gender}',
                                labels={'current_elo': 'ELO Rating'},
                                color_discrete_sequence=[DISCIPLINE_COLORS.get(selected_discipline, '#3498DB')]
                            )
                            fig_box.update_layout(showlegend=False)
                            st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Activity vs Performance scatter plot
                        st.markdown("### Activity vs Performance Analysis")
                        
                        fig_scatter = px.scatter(
                            leaderboard,
                            x='competitions',
                            y='current_elo',
                            size='peak_elo',
                            hover_data=['name', 'avg_rank'],
                            title=f'Competition Activity vs ELO Rating - {selected_discipline} {selected_gender}',
                            labels={
                                'competitions': 'Number of Competitions',
                                'current_elo': 'Current ELO Rating',
                                'peak_elo': 'Peak ELO (size)'
                            },
                            color='avg_rank',
                            color_continuous_scale='RdYlBu_r'
                        )
                        
                        fig_scatter.update_layout(
                            height=500,
                            coloraxis_colorbar=dict(title="Average Rank")
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Individual Athlete Deep Dive
                        st.markdown("### Individual Athlete Analysis")
                        
                        selected_athlete = st.selectbox(
                            "Select an athlete for detailed analysis:",
                            leaderboard['name'].tolist(),
                            help="Choose an athlete to see their detailed ELO progression and statistics"
                        )
                        
                        if selected_athlete:
                            # Get athlete's detailed history
                            athlete_fig = analyzer.create_historical_elo_chart([selected_athlete], era_key)
                            if athlete_fig and athlete_fig.data:
                                st.plotly_chart(athlete_fig, use_container_width=True)
                            
                            # Athlete statistics
                            athlete_stats = leaderboard[leaderboard['name'] == selected_athlete].iloc[0]
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("Current ELO", f"{athlete_stats['current_elo']:.0f}")
                            with col2:
                                st.metric("Peak ELO", f"{athlete_stats['peak_elo']:.0f}")
                            with col3:
                                st.metric("Competitions", f"{athlete_stats['competitions']:.0f}")
                            with col4:
                                st.metric("Average Rank", f"{athlete_stats['avg_rank']:.1f}")
                            with col5:
                                rank_in_leaderboard = leaderboard[leaderboard['name'] == selected_athlete].index[0] + 1
                                st.metric("Current Rank", f"#{rank_in_leaderboard}")
                
                except Exception as e:
                    st.error(f"Error in ELO analysis: {str(e)}")
                    st.info("This might be due to missing data or incompatible era files. Please check your data structure.")

           
        else:
            st.info("No era-specific files found for ELO calculation")

