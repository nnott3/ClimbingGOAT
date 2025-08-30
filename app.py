# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import warnings
# import sys
# import os

# # Add the utils directory to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# try:
#     from utils.analysis import ClimbingAnalyzer
# except ImportError:
#     st.error("Could not import ClimbingAnalyzer. Please ensure utils/analysis.py exists in the correct location.")
#     st.stop()

# warnings.filterwarnings('ignore')

# # Consistent color palette
# DISCIPLINE_COLORS = {'Lead': '#E74C3C', 'Boulder': '#3498DB', 'Speed': '#27AE60', 'Combined': '#9B59B6'}
# GENDER_COLORS = {'Men': '#2C3E50', 'Women': '#E91E63'}
# CHART_COLORS = ['#E74C3C', '#3498DB', '#27AE60', '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22', '#95A5A6', '#34495E', '#8E44AD']

# # Page configuration
# st.set_page_config(
#     page_title="Climbing Competition Analysis",
#     page_icon="🧗‍♂️",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
# .main-header {
#     font-size: 3rem;
#     color: #1f77b4;
#     text-align: center;
#     margin-bottom: 2rem;
# }
# .metric-container {
#     background-color: #f0f2f6;
#     padding: 1rem;
#     border-radius: 0.5rem;
#     margin: 0.5rem 0;
# }
# .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
#     font-size: 1.2rem;
#     font-weight: bold;
# }
# .participation-trends {
#     margin-top: 3rem;
#     margin-bottom: 3rem;
# }
# .filter-section {
#     background-color: #f8f9fa;
#     padding: 1rem;
#     border-radius: 0.5rem;
#     margin-bottom: 1rem;
# }
# .button-row {
#     display: flex;
#     gap: 0.5rem;
#     margin-bottom: 1rem;
# }
# </style>
# """, unsafe_allow_html=True)

# # Country emoji mapping
# COUNTRY_FLAGS = {
#     'USA': '🇺🇸', 'FRA': '🇫🇷', 'GER': '🇩🇪', 'JPN': '🇯🇵', 'GBR': '🇬🇧',
#     'AUT': '🇦🇹', 'CAN': '🇨🇦', 'ITA': '🇮🇹', 'RUS': '🇷🇺', 'CHE': '🇨🇭',
#     'ESP': '🇪🇸', 'BEL': '🇧🇪', 'NED': '🇳🇱', 'POL': '🇵🇱', 'CZE': '🇨🇿',
#     'SLO': '🇸🇮', 'KOR': '🇰🇷', 'AUS': '🇦🇺', 'NOR': '🇳🇴', 'SWE': '🇸🇪'
# }

# @st.cache_data
# def load_analyzer():
#     """Load the climbing analyzer with caching."""
#     try:
#         analyzer = ClimbingAnalyzer()
#         return analyzer
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         st.error("Please check that the DATA folder exists with the correct structure")
#         return None

# def filter_disciplines(overview):
#     """Filter out Boulder&Lead and Combined disciplines"""
#     if overview.get('disciplines'):
#         filtered_disciplines = {k: v for k, v in overview['disciplines'].items() 
#                               if k not in ['Boulder&Lead', 'Combined']}
#         overview['disciplines'] = filtered_disciplines
#     return overview

# def add_country_flags(df, country_col='country'):
#     """Add emoji flags to country names"""
#     if country_col in df.columns:
#         df[country_col] = df[country_col].apply(lambda x: f"{COUNTRY_FLAGS.get(x, '')} {x}")
#     return df

# def create_athlete_filter_buttons(discipline_default='Boulder', gender_default='Men'):
#     """Create discipline and gender filter buttons for athlete analysis."""
#     st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    
#     # Discipline selection
#     st.markdown("### Select Discipline")
#     disciplines = ['Boulder', 'Lead', 'Speed']
#     col1, col2, col3 = st.columns(3)
    
#     selected_discipline = discipline_default
    
#     with col1:
#         if st.button(f"🧗 Boulder", key="athlete_boulder", use_container_width=True, 
#                     type="primary" if discipline_default == "Boulder" else "secondary"):
#             selected_discipline = "Boulder"
    
#     with col2:
#         if st.button(f"🪨 Lead", key="athlete_lead", use_container_width=True,
#                     type="primary" if discipline_default == "Lead" else "secondary"):
#             selected_discipline = "Lead"
    
#     with col3:
#         if st.button(f"⚡ Speed", key="athlete_speed", use_container_width=True,
#                     type="primary" if discipline_default == "Speed" else "secondary"):
#             selected_discipline = "Speed"
    
#     # Gender selection  
#     st.markdown("### Select Gender")
#     col1, col2 = st.columns(2)
    
#     selected_gender = gender_default
    
#     with col1:
#         if st.button(f"👨 Men", key="athlete_men", use_container_width=True,
#                     type="primary" if gender_default == "Men" else "secondary"):
#             selected_gender = "Men"
    
#     with col2:
#         if st.button(f"👩 Women", key="athlete_women", use_container_width=True,
#                     type="primary" if gender_default == "Women" else "secondary"):
#             selected_gender = "Women"
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     return selected_discipline, selected_gender

# def create_country_charts(country_stats, metric='total_wins'):
#     """Create visual charts for country statistics instead of tables."""
#     if country_stats.empty:
#         return None
    
#     # Get top 15 countries
#     if metric == 'total_wins':
#         top_countries = country_stats.nlargest(15, 'total_wins')
#         title = "🏆 Most Successful Countries (Total Wins)"
#         y_label = "Total Wins"
#         color_col = 'total_wins'
#         color_scale = 'Reds'
#     elif metric == 'total_podiums':
#         top_countries = country_stats.nlargest(15, 'total_podiums')
#         title = "🥉 Most Successful Countries (Total Podiums)"
#         y_label = "Total Podiums"
#         color_col = 'total_podiums'
#         color_scale = 'Blues'
#     else:
#         top_countries = country_stats.nlargest(15, 'total_athletes')
#         title = "👥 Most Active Countries (Total Athletes)"
#         y_label = "Total Athletes"
#         color_col = 'total_athletes'
#         color_scale = 'Greens'
    
#     fig = px.bar(
#         top_countries,
#         x=color_col,
#         y='country',
#         orientation='h',
#         title=title,
#         labels={color_col: y_label, 'country': 'Country'},
#         color=color_col,
#         color_continuous_scale=color_scale,
#         text=color_col
#     )
    
#     fig.update_layout(
#         yaxis={'categoryorder': 'total ascending'},
#         height=500,
#         showlegend=False
#     )
    
#     fig.update_traces(texttemplate='%{text}', textposition='outside')
    
#     return fig

# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🧗‍♂️ Climbing Competition Analysis Dashboard</h1>', unsafe_allow_html=True)
    
#     # Load data with error handling
#     analyzer = load_analyzer()
#     if analyzer is None:
#         st.error("Failed to load data. Please check your data files and folder structure.")
#         st.markdown("""
#         Expected folder structure:
#         ```
#         DATA/
#         ├── data_summary/
#         │   ├── data_summary.csv
#         │   ├── era_statistics.csv
#         │   ├── event_metadata.csv
#         │   └── era_files_summary.csv
#         └── raw_data/
#             ├── aggregated_results.csv
#             └── [era-specific files].csv
#         ```
#         """)
#         return
    
#     overview = analyzer.get_data_overview()
    
#     if not overview:
#         st.warning("No data found. Please check your data files.")
#         return
    
#     # Filter out unwanted disciplines
#     overview = filter_disciplines(overview)
    
#     # Create basic filters
#     if overview.get('disciplines'):
#         available_disciplines = [d for d in overview['disciplines'].keys() 
#                                if d not in ['Boulder&Lead', 'Combined']]
#     else:
#         available_disciplines = []
    
#     filters = {
#         'year_range': overview.get('year_range'),
#         'disciplines': available_disciplines,
#         'genders': list(overview.get('genders', {}).keys()) if overview.get('genders') else [],
#         'countries': None,
#         'rounds': list(overview.get('rounds', {}).keys()) if overview.get('rounds') else []
#     }
    
#     # Main content tabs
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "📈 Overview", 
#         "🏆 Athletes", 
#         "🌍 Countries", 
#         "🎯 ELO Rankings", 
#         "🔍 Deep Dive"
#     ])
    
#     with tab1:
#         st.markdown("## 📈 Competition Overview")
        
#         # Key metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric(
#                 "Total Records",
#                 f"{overview.get('total_records', 0):,}",
#                 help="Total competition results in database"
#             )
        
#         with col2:
#             st.metric(
#                 "Unique Athletes",
#                 f"{overview.get('unique_athletes', 0):,}",
#                 help="Total number of different athletes"
#             )
        
#         with col3:
#             st.metric(
#                 "Countries",
#                 f"{overview.get('unique_countries', 0):,}",
#                 help="Countries represented in competitions"
#             )
        
#         with col4:
#             if overview.get('year_range'):
#                 year_span = overview['year_range'][1] - overview['year_range'][0] + 1
#                 st.metric(
#                     "Years Covered",
#                     f"{year_span}",
#                     help=f"From {overview['year_range'][0]} to {overview['year_range'][1]}"
#                 )
        
#         # Discipline and gender distribution with consistent colors
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if overview.get('disciplines'):
#                 colors = [DISCIPLINE_COLORS.get(disc, '#95A5A6') for disc in overview['disciplines'].keys()]
#                 fig = px.pie(
#                     values=list(overview['disciplines'].values()),
#                     names=list(overview['disciplines'].keys()),
#                     title="Distribution by Discipline",
#                     color_discrete_sequence=colors
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             if overview.get('genders'):
#                 colors = [GENDER_COLORS.get(gender, '#95A5A6') for gender in overview['genders'].keys()]
#                 fig = px.pie(
#                     values=list(overview['genders'].values()),
#                     names=list(overview['genders'].keys()),
#                     title="Distribution by Gender",
#                     color_discrete_sequence=colors
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
        
#         # Participation trends
#         st.markdown('<div class="participation-trends">', unsafe_allow_html=True)
#         st.markdown("### 📈 Participation Trends Over Time")
#         trends_fig = analyzer.create_participation_trends(filters)
#         if trends_fig.data:
#             st.plotly_chart(trends_fig, use_container_width=True)
#         else:
#             st.info("No data available for current filters")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with tab2:
#         st.markdown("## 🏆 Athlete Analysis")
        
#         # Initialize session state for selections
#         if 'selected_athlete_discipline' not in st.session_state:
#             st.session_state.selected_athlete_discipline = 'Boulder'
#         if 'selected_athlete_gender' not in st.session_state:
#             st.session_state.selected_athlete_gender = 'Men'
        
#         # Filter controls
#         selected_discipline, selected_gender = create_athlete_filter_buttons(
#             st.session_state.selected_athlete_discipline,
#             st.session_state.selected_athlete_gender
#         )
        
#         # Update session state
#         st.session_state.selected_athlete_discipline = selected_discipline
#         st.session_state.selected_athlete_gender = selected_gender
        
#         # Create filtered data
#         athlete_filters = {
#             'disciplines': [selected_discipline],
#             'genders': [selected_gender],
#             'year_range': filters['year_range'],
#             'rounds': filters['rounds']
#         }
        
#         # Get athlete statistics
#         athlete_stats = analyzer.get_athlete_stats(athlete_filters)
        
#         if not athlete_stats.empty:
#             st.markdown(f"**Showing:** {selected_discipline} - {selected_gender}")
            
#             # Top performers charts
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("### 🥇 Most Successful Athletes (Wins)")
#                 top_winners = athlete_stats.nlargest(10, 'wins')
#                 if not top_winners.empty:
#                     fig = px.bar(
#                         top_winners,
#                         x='wins',
#                         y='name',
#                         orientation='h',
#                         title=f'Top 10 {selected_discipline} {selected_gender} by Wins',
#                         labels={'wins': 'Number of Wins', 'name': 'Athlete'},
#                         color='wins',
#                         color_continuous_scale='Reds',
#                         text='wins'
#                     )
#                     fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
#                     fig.update_traces(texttemplate='%{text}', textposition='outside')
#                     st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.info("No wins data available for current selection")
            
#             with col2:
#                 st.markdown("### 🏅 Most Successful Athletes (Podiums)")
#                 top_podiums = athlete_stats.nlargest(10, 'podiums')
#                 if not top_podiums.empty:
#                     fig = px.bar(
#                         top_podiums,
#                         x='podiums',
#                         y='name',
#                         orientation='h',
#                         title=f'Top 10 {selected_discipline} {selected_gender} by Podiums',
#                         labels={'podiums': 'Number of Podiums', 'name': 'Athlete'},
#                         color='podiums',
#                         color_continuous_scale='Blues',
#                         text='podiums'
#                     )
#                     fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
#                     fig.update_traces(texttemplate='%{text}', textposition='outside')
#                     st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.info("No podium data available for current selection")
            
#             # Most active athletes chart
#             st.markdown("### 🏃‍♀️ Most Active Athletes")
#             active_athletes = athlete_stats.nlargest(10, 'total_competitions')
#             if not active_athletes.empty:
#                 fig = px.bar(
#                     active_athletes,
#                     x='total_competitions',
#                     y='name',
#                     orientation='h',
#                     title=f'Top 10 Most Active {selected_discipline} {selected_gender}',
#                     labels={'total_competitions': 'Number of Competitions', 'name': 'Athlete'},
#                     color='total_competitions',
#                     color_continuous_scale='Greens',
#                     text='total_competitions'
#                 )
#                 fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
#                 fig.update_traces(texttemplate='%{text}', textposition='outside')
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No activity data available for current selection")
            
#             # Individual athlete analysis
#             st.markdown("### 🎯 Individual Athlete Deep Dive")
            
#             # Athlete selector
#             athlete_names = sorted(athlete_stats['name'].unique())
#             selected_athlete = st.selectbox("Select Athlete", athlete_names)
            
#             if selected_athlete:
#                 # Athlete timeline
#                 timeline_fig = analyzer.create_athlete_timeline(selected_athlete)
#                 if timeline_fig.data:
#                     st.plotly_chart(timeline_fig, use_container_width=True)
                
#                 # Athlete detailed stats
#                 athlete_detail = athlete_stats[athlete_stats['name'] == selected_athlete].iloc[0]
                
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Total Competitions", int(athlete_detail['total_competitions']))
#                 with col2:
#                     st.metric("Wins", int(athlete_detail['wins']))
#                 with col3:
#                     st.metric("Podiums", int(athlete_detail['podiums']))
#                 with col4:
#                     st.metric("Win Rate", f"{athlete_detail['win_rate']:.1f}%")
        
#         else:
#             st.warning("No athlete data available for current selection")
    
#     with tab3:
#         st.markdown("## 🌍 Country Analysis")
        
#         # Country statistics
#         country_stats = analyzer.get_country_stats(filters)
        
#         if not country_stats.empty:
#             # Visual charts for country performance
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 wins_chart = create_country_charts(country_stats, 'total_wins')
#                 if wins_chart:
#                     st.plotly_chart(wins_chart, use_container_width=True)
            
#             with col2:
#                 podiums_chart = create_country_charts(country_stats, 'total_podiums')
#                 if podiums_chart:
#                     st.plotly_chart(podiums_chart, use_container_width=True)
            
#             # Most active countries chart
#             active_chart = create_country_charts(country_stats, 'total_athletes')
#             if active_chart:
#                 st.plotly_chart(active_chart, use_container_width=True)
            
#             # Performance heatmaps by discipline
#             st.markdown("### 🗺️ Performance by Country and Discipline")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 wins_heatmap = analyzer.create_country_discipline_heatmap(filters, metric='wins')
#                 if wins_heatmap.data:
#                     st.plotly_chart(wins_heatmap, use_container_width=True)
            
#             with col2:
#                 podiums_heatmap = analyzer.create_country_discipline_heatmap(filters, metric='podiums')
#                 if podiums_heatmap.data:
#                     st.plotly_chart(podiums_heatmap, use_container_width=True)
                    
#         else:
#             st.warning("No country data available for current filters")
    
#     with tab4:
#         st.markdown("## 🎯 ELO Rankings")
        
#         # ELO system selection with buttons
#         era_files = overview.get('era_files', [])
#         if era_files:
#             # Initialize session state
#             if 'selected_elo_discipline' not in st.session_state:
#                 st.session_state.selected_elo_discipline = 'Speed'
#             if 'selected_elo_gender' not in st.session_state:
#                 st.session_state.selected_elo_gender = 'Men'
#             if 'selected_elo_era' not in st.session_state:
#                 st.session_state.selected_elo_era = None
            
#             disciplines = ['Speed', 'Boulder', 'Lead']
#             genders = ['Men', 'Women']
            
#             # Discipline selection
#             st.markdown("### Select Discipline")
#             discipline_cols = st.columns(len(disciplines))
            
#             for i, discipline in enumerate(disciplines):
#                 with discipline_cols[i]:
#                     if st.button(f"🧗 {discipline}", key=f"elo_disc_{discipline}", use_container_width=True):
#                         st.session_state.selected_elo_discipline = discipline
            
#             # Gender selection
#             st.markdown("### Select Gender")
#             gender_cols = st.columns(len(genders))
            
#             for i, gender in enumerate(genders):
#                 with gender_cols[i]:
#                     emoji = "👨" if gender == "Men" else "👩"
#                     if st.button(f"{emoji} {gender}", key=f"elo_gender_{gender}", use_container_width=True):
#                         st.session_state.selected_elo_gender = gender
            
#             # Era selection based on discipline and gender
#             selected_discipline = st.session_state.selected_elo_discipline
#             selected_gender = st.session_state.selected_elo_gender
            
#             relevant_eras = [era for era in era_files if selected_discipline.lower() in era.lower()]
            
#             if relevant_eras:
#                 st.markdown("### Select Era/System")
#                 era_cols = st.columns(min(len(relevant_eras), 4))
                
#                 for i, era in enumerate(relevant_eras[:4]):
#                     with era_cols[i % 4]:
#                         if st.button(f"📅 {era}", key=f"elo_era_{era}", use_container_width=True):
#                             st.session_state.selected_elo_era = era
                
#                 # Default to first era if none selected
#                 if st.session_state.selected_elo_era is None:
#                     st.session_state.selected_elo_era = relevant_eras[0]
                
#                 selected_era = st.session_state.selected_elo_era
#                 st.markdown(f"**Current Selection:** {selected_discipline} - {selected_gender} - {selected_era}")
                
#                 # Calculate ELO
#                 with st.spinner("Calculating ELO ratings..."):
#                     leaderboard = analyzer.get_elo_leaderboard(selected_era, discipline=selected_discipline, gender=selected_gender)
                
#                 if not leaderboard.empty:
#                     # Top 5 athletes historical ELO chart
#                     st.markdown("### 📊 Top 5 Athletes - Historical ELO")
#                     top_5 = leaderboard.head(5)
                    
#                     historical_fig = analyzer.create_historical_elo_chart(top_5['name'].tolist(), selected_era)
#                     if historical_fig.data:
#                         st.plotly_chart(historical_fig, use_container_width=True)
                    
#                     # Dropdown to add more athletes
#                     st.markdown("### 🔍 Add More Athletes to Chart")
#                     all_athletes = sorted(leaderboard['name'].tolist())
#                     additional_athletes = st.multiselect(
#                         "Select additional athletes to add to the chart",
#                         [athlete for athlete in all_athletes if athlete not in top_5['name'].tolist()],
#                         max_selections=5
#                     )
                    
#                     if additional_athletes:
#                         combined_athletes = top_5['name'].tolist() + additional_athletes
#                         combined_fig = analyzer.create_historical_elo_chart(combined_athletes, selected_era)
#                         if combined_fig.data:
#                             st.plotly_chart(combined_fig, use_container_width=True)
                    
#                     # Current ELO Leaderboard
#                     st.markdown("### 🏆 Current ELO Leaderboard - Top 20")
#                     top_elo = leaderboard.head(20)[['name', 'current_elo', 'peak_elo', 'competitions', 'avg_rank']]
#                     st.dataframe(top_elo, use_container_width=True)
                    
#                     # ELO distribution
#                     st.markdown("### 📈 ELO Rating Distribution")
#                     fig = px.histogram(
#                         leaderboard,
#                         x='current_elo',
#                         nbins=30,
#                         title=f'ELO Rating Distribution - {selected_discipline} {selected_gender}',
#                         labels={'current_elo': 'Current ELO Rating', 'count': 'Number of Athletes'},
#                         color_discrete_sequence=['#3498DB']
#                     )
#                     st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.warning("No ELO data available for selected criteria")
#             else:
#                 st.warning(f"No era files found for {selected_discipline}")
#         else:
#             st.info("No era-specific files found for ELO calculation")
    
#     with tab5:
#         st.markdown("## 🔍 Deep Dive Analysis")
        
#         analysis_options = [
#             "Yearly Trends",
#             "Career Progression", 
#             "Head-to-Head Comparison",
#             "Location Performance",
#             "Discipline Crossover"
#         ]
        
#         selected_analysis = st.selectbox("Select Analysis Type", analysis_options)
        
#         st.info(f"🚧 {selected_analysis} analysis coming soon! We're working on bringing you detailed insights.")
        
#         # Placeholder for future features
#         st.markdown("""
#         ### Planned Features:
#         - **Yearly Trends**: Track how climbing performance and participation evolved over time
#         - **Career Progression**: Analyze individual athlete development and peak performance periods  
#         - **Head-to-Head Comparison**: Compare two athletes across multiple metrics and competitions
#         - **Location Performance**: See how different venues and countries affect performance
#         - **Discipline Crossover**: Analyze athletes who compete across multiple climbing disciplines
#         """)

# if __name__ == "__main__":
#     main()


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
    'Lead': os.getenv('LEAD_COLOR', '#E74C3C'), 
    'Boulder': os.getenv('BOULDER_COLOR', '#3498DB'), 
    'Speed': os.getenv('SPEED_COLOR', '#27AE60'), 
    'Combined': os.getenv('COMBINED_COLOR', '#9B59B6')
}

GENDER_COLORS = {
    'Men': os.getenv('MEN_COLOR', '#2C3E50'), 
    'Women': os.getenv('WOMEN_COLOR', '#E91E63')
}

CHART_COLORS = [
    os.getenv('CHART_COLOR_1', '#E74C3C'),
    os.getenv('CHART_COLOR_2', '#3498DB'),
    os.getenv('CHART_COLOR_3', '#27AE60'),
    os.getenv('CHART_COLOR_4', '#9B59B6'),
    os.getenv('CHART_COLOR_5', '#F39C12'),
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
        if st.button(f"🧗 Boulder", key="athlete_boulder", use_container_width=True, 
                    type="primary" if discipline_default == "Boulder" else "secondary"):
            selected_discipline = "Boulder"
    
    with col2:
        if st.button(f"🪨 Lead", key="athlete_lead", use_container_width=True,
                    type="primary" if discipline_default == "Lead" else "secondary"):
            selected_discipline = "Lead"
    
    with col3:
        if st.button(f"⚡ Speed", key="athlete_speed", use_container_width=True,
                    type="primary" if discipline_default == "Speed" else "secondary"):
            selected_discipline = "Speed"
    
    # Gender selection  
    st.markdown("### Select Gender")
    col1, col2 = st.columns(2)
    
    selected_gender = gender_default
    
    with col1:
        if st.button(f"👨 Men", key="athlete_men", use_container_width=True,
                    type="primary" if gender_default == "Men" else "secondary"):
            selected_gender = "Men"
    
    with col2:
        if st.button(f"👩 Women", key="athlete_women", use_container_width=True,
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
        └── raw_data/
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
    
    # Create basic filters
    if overview.get('disciplines'):
        available_disciplines = [d for d in overview['disciplines'].keys() 
                               if d not in ['Boulder&Lead', 'Combined']]
    else:
        available_disciplines = []
    
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
        print(era_files)
        if era_files:
            # Initialize session state
            if 'selected_elo_discipline' not in st.session_state:
                st.session_state.selected_elo_discipline = 'Lead'
            if 'selected_elo_gender' not in st.session_state:
                st.session_state.selected_elo_gender = 'Men'
            if 'selected_elo_era' not in st.session_state:
                st.session_state.selected_elo_era = None
            
            disciplines = ['Speed', 'Boulder', 'Lead']
            genders = ['Men', 'Women']
            
            # Discipline selection
            st.markdown("### Select Discipline")
            discipline_cols = st.columns(len(disciplines))
            
            for i, discipline in enumerate(disciplines):
                with discipline_cols[i]:
                    if st.button(f"🧗 {discipline}", key=f"elo_disc_{discipline}", use_container_width=True):
                        st.session_state.selected_elo_discipline = discipline
            
            # Gender selection
            st.markdown("### Select Gender")
            gender_cols = st.columns(len(genders))
            
            for i, gender in enumerate(genders):
                with gender_cols[i]:
                    emoji = "👨" if gender == "Men" else "👩"
                    if st.button(f"{emoji} {gender}", key=f"elo_gender_{gender}", use_container_width=True):
                        st.session_state.selected_elo_gender = gender
            
            # Era selection based on discipline and gender
            selected_discipline = st.session_state.selected_elo_discipline
            selected_gender = st.session_state.selected_elo_gender
            
            relevant_eras = [era for era in era_files if selected_discipline.lower() in era.lower()]
            
            if relevant_eras:
                st.markdown("### Select Era/System")
                era_cols = st.columns(min(len(relevant_eras), 4))
                
                for i, era in enumerate(relevant_eras[:4]):
                    with era_cols[i % 4]:
                        if st.button(f"📅 {era}", key=f"elo_era_{era}", use_container_width=True):
                            st.session_state.selected_elo_era = era
                
                # Default to first era if none selected
                if st.session_state.selected_elo_era is None:
                    st.session_state.selected_elo_era = relevant_eras[0]
                
                selected_era = st.session_state.selected_elo_era
                st.markdown(f"**Current Selection:** {selected_discipline} - {selected_gender} - {selected_era}")
                
                # Calculate ELO
                with st.spinner("Calculating ELO ratings..."):
                    leaderboard = analyzer.get_elo_leaderboard(selected_era, discipline=selected_discipline, gender=selected_gender)
                
                if not leaderboard.empty:
                    # Top 5 athletes historical ELO chart
                    st.markdown("### 📊 Top 5 Athletes - Historical ELO")
                    top_5 = leaderboard.head(5)
                    
                    historical_fig = analyzer.create_historical_elo_chart(top_5['name'].tolist(), selected_era)
                    if historical_fig.data:
                        st.plotly_chart(historical_fig, use_container_width=True)
                    
                    # Dropdown to add more athletes
                    st.markdown("### 🔍 Add More Athletes to Chart")
                    all_athletes = sorted(leaderboard['name'].tolist())
                    additional_athletes = st.multiselect(
                        "Select additional athletes to add to the chart",
                        [athlete for athlete in all_athletes if athlete not in top_5['name'].tolist()],
                        max_selections=5
                    )
                    
                    if additional_athletes:
                        combined_athletes = top_5['name'].tolist() + additional_athletes
                        combined_fig = analyzer.create_historical_elo_chart(combined_athletes, selected_era)
                        if combined_fig.data:
                            st.plotly_chart(combined_fig, use_container_width=True)
                    
                    # Current ELO Leaderboard
                    st.markdown("### 🏆 Current ELO Leaderboard - Top 20")
                    top_elo = leaderboard.head(20)[['name', 'current_elo', 'peak_elo', 'competitions', 'avg_rank']]
                    st.dataframe(top_elo, use_container_width=True)
                    
                    # ELO distribution
                    st.markdown("### 📈 ELO Rating Distribution")
                    fig = px.histogram(
                        leaderboard,
                        x='current_elo',
                        nbins=30,
                        title=f'ELO Rating Distribution - {selected_discipline} {selected_gender}',
                        labels={'current_elo': 'Current ELO Rating', 'count': 'Number of Athletes'},
                        color_discrete_sequence=[DISCIPLINE_COLORS.get(selected_discipline, '#3498DB')]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No ELO data available for selected criteria")
            else:
                st.warning(f"No era files found for {selected_discipline}")
        else:
            st.info("No era-specific files found for ELO calculation")

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
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if overview.get('genders'):
                colors = [GENDER_COLORS.get(gender, '#95A5A6') for gender in overview['genders'].keys()]
                fig = px.pie(
                    values=list(overview['genders'].values()),
                    names=list(overview['genders'].keys()),
                    title="Distribution by Gender",
                    color_discrete_sequence=colors
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Participation trends
        st.markdown('<div class="participation-trends">', unsafe_allow_html=True)
        st.markdown("### 📈 Participation Trends Over Time")
        trends_fig = analyzer.create_participation_trends(filters)
        if trends_fig.data:
            st.plotly_chart(trends_fig, use_container_width=True)
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
                    st.plotly_chart(fig, use_container_width=True)
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
                    st.plotly_chart(fig, use_container_width=True)
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
                st.plotly_chart(fig, use_container_width=True)
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
                if timeline_fig.data:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                
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
                    st.plotly_chart(wins_chart, use_container_width=True)
            
            with col2:
                podiums_chart = create_country_charts(country_stats, 'total_podiums')
                if podiums_chart:
                    st.plotly_chart(podiums_chart, use_container_width=True)
            
            # Most active countries chart
            active_chart = create_country_charts(country_stats, 'total_athletes')
            if active_chart:
                st.plotly_chart(active_chart, use_container_width=True)
            
            # Performance heatmaps by discipline
            st.markdown("### 🗺️ Performance by Country and Discipline")
            
            col1, col2 = st.columns(2)
            
            with col1:
                wins_heatmap = analyzer.create_country_discipline_heatmap(filters, metric='wins')
                if wins_heatmap.data:
                    st.plotly_chart(wins_heatmap, use_container_width=True)
            
            with col2:
                podiums_heatmap = analyzer.create_country_discipline_heatmap(filters, metric='podiums')
                if podiums_heatmap.data:
                    st.plotly_chart(podiums_heatmap, use_container_width=True)
                    
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