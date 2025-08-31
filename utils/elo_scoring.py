# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from typing import Dict, List, Tuple, Optional
# import warnings
# import os
# warnings.filterwarnings('ignore')

# class ELOCalculator:
#     """Dedicated ELO calculation and analysis for climbing competitions."""
    
#     def __init__(self, era_files: Dict[str, pd.DataFrame]):
#         self.era_files = era_files
#         self.elo_cache = {}  # Cache calculated ELO histories
        
#         # Get colors from environment variables - consistent with app.py
#         self.discipline_colors = {
#             'Boulder': os.getenv('BOULDER_COLOR', '#2777AC'),
#             'Lead': os.getenv('LEAD_COLOR', '#C33727'), 
#             'Speed': os.getenv('SPEED_COLOR', '#1C7C44')
#         }
        
#         self.chart_colors = [
#             os.getenv('CHART_COLOR_1', '#D24130'),
#             os.getenv('CHART_COLOR_2', '#2885C3'),
#             os.getenv('CHART_COLOR_3', '#1C7C44'),
#             os.getenv('CHART_COLOR_4', '#A156BE'),
#             os.getenv('CHART_COLOR_5', '#D98B0F'),
#             os.getenv('CHART_COLOR_6', '#1ABC9C'),
#             os.getenv('CHART_COLOR_7', '#E67E22'),
#             os.getenv('CHART_COLOR_8', '#95A5A6'),
#             os.getenv('CHART_COLOR_9', '#34495E'),
#             os.getenv('CHART_COLOR_10', '#8E44AD')
#         ]
    
#     def calculate_elo_ratings(self, era_file: str, k_factor: int = 32, discipline: str = None, gender: str = None) -> pd.DataFrame:
#         """Calculate ELO ratings for athletes in a specific era with optional filters."""
#         cache_key = f"{era_file}_{discipline}_{gender}_{k_factor}"
        
#         if cache_key in self.elo_cache:
#             return self.elo_cache[cache_key]
        
#         if era_file not in self.era_files:
#             return pd.DataFrame()
        
#         # Load and clean data
#         df = pd.read_csv(f"./Data/aggregate_data/{era_file}.csv")
#         df = df.dropna(subset=['round_rank'])
        
#         # Filter by discipline and gender
#         if discipline:
#             df = df[df['discipline'].str.lower() == discipline.lower()]
#         if gender:
#             df = df[df['gender'].str.lower() == gender.lower()]
        
#         if df.empty:
#             return pd.DataFrame()
        
#         df['start_date'] = pd.to_datetime(df['start_date'])
#         df = df.sort_values(['year', 'start_date', 'event_name'])
        
        
#         # Initialize ELO ratings and history
#         elo_ratings = {}
#         elo_history = []
#         last_competition_date = {}  # Track last competition for each athlete
        
#         # Get all unique athletes and dates for off-season tracking
#         all_athletes = set(df['name'].unique())
#         all_dates = sorted(df['start_date'].unique())
        
#         # Group by competition
#         for (event, year, disc, gend, round_type), event_data in df.groupby(['event_name', 'year', 'discipline', 'gender', 'round']):
#             event_data = event_data.sort_values('round_rank')
#             athletes = event_data['name'].tolist()
#             ranks = event_data['round_rank'].tolist()
#             start_date = event_data['start_date'].iloc[0]

#             # Initialize new athletes with 1500 rating
#             for athlete in athletes:
#                 if athlete not in elo_ratings:
#                     elo_ratings[athlete] = 1500
#                     # Add initial rating record
#                     elo_history.append({
#                         'name': athlete,
#                         'event': 'Initial Rating',
#                         'year': year,
#                         'date': start_date,
#                         'discipline': disc,
#                         'gender': gend,
#                         'round': 'N/A',
#                         'rank': None,
#                         'elo_before': 1500,
#                         'elo_after': 1500,
#                         'elo_change': 0,
#                         'competed': False
#                     })
            
            
#             # Calculate ELO changes for competing athletes
#             n_athletes = len(athletes)
#             for i, athlete_a in enumerate(athletes):
#                 rank_a = ranks[i]
#                 rating_a = elo_ratings[athlete_a]
                
#                 # Compare against all other athletes
#                 total_change = 0
#                 for j, athlete_b in enumerate(athletes):
#                     if i != j:
#                         rank_b = ranks[j]
#                         rating_b = elo_ratings[athlete_b]
                        
#                         # Expected score (better rank = higher expected score)
#                         expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                        
#                         # Actual score (better rank wins)
#                         actual_a = 1 if rank_a < rank_b else 0 if rank_a > rank_b else 0.5
                        
#                         # ELO change
#                         change = k_factor * (actual_a - expected_a) / (n_athletes - 1)
#                         total_change += change
                
#                 # Update rating
#                 elo_ratings[athlete_a] += total_change
#                 last_competition_date[athlete_a] = start_date
                
#                 # Record history
#                 elo_history.append({
#                     'name': athlete_a,
#                     'event': event,
#                     'year': year,
#                     'date': start_date,
#                     'discipline': disc,
#                     'gender': gend,
#                     'round': round_type,
#                     'rank': rank_a,
#                     'elo_before': rating_a,
#                     'elo_after': elo_ratings[athlete_a],
#                     'elo_change': total_change,
#                     'competed': True
#                 })
        
#         result = pd.DataFrame(elo_history)
#         self.elo_cache[cache_key] = result
#         return result

#     def get_current_leaderboard(self, era_file: str, discipline: str = None, gender: str = None) -> pd.DataFrame:
#         """Get current ELO leaderboard for specified criteria."""
#         print('aaaaa')
#         elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
#         if elo_history.empty:
#             return pd.DataFrame()
        
#         # Get latest rating for each athlete (only from actual competitions)
#         competing_history = elo_history[elo_history['competed'] == True]
        
#         if competing_history.empty:
#             return pd.DataFrame()
        
#         latest_ratings = competing_history.groupby('name').last()[['elo_after']].reset_index()
#         latest_ratings = latest_ratings.rename(columns={'elo_after': 'current_elo'})
        
#         # Add additional stats (only from competitions)
#         athlete_stats = competing_history.groupby('name').agg({
#             'elo_change': ['count', 'mean', 'std'],
#             'rank': 'mean',
#             'elo_after': ['min', 'max']
#         }).round(2)
        
#         athlete_stats.columns = ['competitions', 'avg_elo_change', 'elo_volatility', 'avg_rank', 'min_elo', 'peak_elo']
#         athlete_stats = athlete_stats.reset_index()
        
#         # Merge with current ratings
#         leaderboard = latest_ratings.merge(athlete_stats, on='name')
#         leaderboard = leaderboard.sort_values('current_elo', ascending=False)
        
#         return leaderboard

    #### BY DATE ####
    # def create_historical_elo_chart(self, athletes: List[str], era_file: str, discipline: str = None, gender: str = None) -> go.Figure:
    #     """Create historical ELO progression chart for specified athletes."""
    #     elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
    #     if elo_history.empty:
    #         return go.Figure()
        
    #     # Ensure date column is datetime
    #     elo_history['date'] = pd.to_datetime(elo_history['date'], errors='coerce')
    #     elo_history = elo_history.dropna(subset=['date'])
        
    #     # Filter for specified athletes (case-insensitive)
    #     athlete_data = elo_history[elo_history['name'].str.lower().isin([a.lower() for a in athletes])]
        
    #     if athlete_data.empty:
    #         return go.Figure()
        
    #     fig = go.Figure()
        
    #     for i, athlete in enumerate(athletes):
    #         athlete_elo = athlete_data[athlete_data['name'].str.lower() == athlete.lower()].copy()
    #         if not athlete_elo.empty:
    #             athlete_elo = athlete_elo.sort_values(['date', 'event'])
                
    #             # Get actual name with correct case
    #             actual_name = athlete_elo['name'].iloc[0]
                
    #             # Separate competing and non-competing points for different styling
    #             competing_data = athlete_elo[athlete_elo['competed'] == True]
    #             off_season_data = athlete_elo[athlete_elo['competed'] == False]
                
    #             color = self.chart_colors[i % len(self.chart_colors)]
                
    #             # Plot competing points with full styling
    #             if not competing_data.empty:
    #                 hover_text_comp = []
    #                 for _, row in competing_data.iterrows():
    #                     hover_text_comp.append(
    #                         f"<b>{actual_name}</b><br>"
    #                         f"Event: {row['event']}<br>"
    #                         f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
    #                         f"Rank: {row['rank']}<br>"
    #                         f"ELO: {row['elo_after']:.0f}<br>"
    #                         f"Change: {row['elo_change']:+.1f}"
    #                     )
                    
    #                 fig.add_trace(go.Scatter(
    #                     x=competing_data['date'],
    #                     y=competing_data['elo_after'],
    #                     mode='lines+markers',
    #                     name=actual_name,
    #                     line=dict(color=color, width=2),
    #                     marker=dict(size=6, color=color),
    #                     hovertemplate='%{hovertext}<extra></extra>',
    #                     hovertext=hover_text_comp,
    #                     showlegend=True
    #                 ))
        
    #     # Title
    #     title_parts = ["Historical ELO Progression"]
    #     if discipline:
    #         title_parts.insert(0, discipline)
    #     if gender:
    #         title_parts.insert(-1, gender)
        
    #     fig.update_layout(
    #         title=" - ".join(title_parts),
    #         xaxis_title="Date",
    #         yaxis_title="ELO Rating",
    #         hovermode='closest',
    #         height=500,
    #         showlegend=True,
    #         legend=dict(
    #             yanchor="top",
    #             y=0.99,
    #             xanchor="left",
    #             x=0.01
    #         )
    #     )
        
    #     return fig
    
    # # def calculate_elo_ratings(self, era_file: str, k_factor: int = 32, discipline: str = None, gender: str = None) -> pd.DataFrame:
    # #     """Calculate ELO ratings for athletes in a specific era with optional filters."""
    # #     cache_key = f"{era_file}_{discipline}_{gender}_{k_factor}"
        
    # #     if cache_key in self.elo_cache:
    # #         return self.elo_cache[cache_key]
        
    # #     if era_file not in self.era_files:
    # #         return pd.DataFrame()
        
    # #     ###### PROBLEM ###### what is df for ?
    # #     df = pd.read_csv(f"./Data/aggregate_data/{era_file}.csv")
    # #     df = df.dropna(subset=['round_rank'])
        
    # #     # Filter by discipline and gender
    # #     if discipline:
    # #         df = df[df['discipline'].str.lower() == discipline.lower()]
    # #     if gender:
    # #         df = df[df['gender'].str.lower() == gender.lower()]
        
    # #     if df.empty:
    # #         return pd.DataFrame()
        
    # #     df = df.sort_values(['year', 'start_date', 'event_name'])
        
        
    # #     # Initialize ELO ratings
    # #     elo_ratings = {}
    # #     elo_history = []
        
    # #     # Group by competition
    # #     for (event, year, disc, gend, round_type), event_data in df.groupby(['event_name', 'year', 'discipline', 'gender', 'round']):
    # #         event_data = event_data.sort_values('round_rank')
    # #         athletes = event_data['name'].tolist()
    # #         ranks = event_data['round_rank'].tolist()
            
    # #         start_date = event_data['start_date'].iloc[0]

    # #         # Initialize new athletes with 1500 rating
    # #         for athlete in athletes:
    # #             if athlete not in elo_ratings:
    # #                 elo_ratings[athlete] = 1500
            
    # #         # Calculate ELO changes
    # #         n_athletes = len(athletes)
    # #         for i, athlete_a in enumerate(athletes):
    # #             rank_a = ranks[i]
    # #             rating_a = elo_ratings[athlete_a]
                
    # #             # Compare against all other athletes
    # #             total_change = 0
    # #             for j, athlete_b in enumerate(athletes):
    # #                 if i != j:
    # #                     rank_b = ranks[j]
    # #                     rating_b = elo_ratings[athlete_b]
                        
    # #                     # Expected score (better rank = higher expected score)
    # #                     expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                        
    # #                     # Actual score (better rank wins)
    # #                     actual_a = 1 if rank_a < rank_b else 0 if rank_a > rank_b else 0.5
                        
    # #                     # ELO change
    # #                     change = k_factor * (actual_a - expected_a) / (n_athletes - 1)
    # #                     total_change += change
                
    # #             # Update rating
    # #             elo_ratings[athlete_a] += total_change
                
    # #             # Record history
    # #             elo_history.append({
    # #                 'name': athlete_a,
    # #                 'event': event,
    # #                 'year': year,
    # #                 'date': start_date,
    # #                 'discipline': disc,
    # #                 'gender': gend,
    # #                 'round': round_type,
    # #                 'rank': rank_a,
    # #                 'elo_before': rating_a,
    # #                 'elo_after': elo_ratings[athlete_a],
    # #                 'elo_change': total_change
    # #             })
        
    # #     result = pd.DataFrame(elo_history)
    # #     self.elo_cache[cache_key] = result
    # #     return result
    
    # # def get_current_leaderboard(self, era_file: str, discipline: str = None, gender: str = None) -> pd.DataFrame:
    # #     """Get current ELO leaderboard for specified criteria."""
    # #     print('aaaaa')
    # #     elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
    # #     if elo_history.empty:
    # #         return pd.DataFrame()
        
    # #     # Get latest rating for each athlete
    # #     latest_ratings = elo_history.groupby('name').last()[['elo_after']].reset_index()
    # #     latest_ratings = latest_ratings.rename(columns={'elo_after': 'current_elo'})
        
    # #     # Add additional stats
    # #     athlete_stats = elo_history.groupby('name').agg({
    # #         'elo_change': ['count', 'mean', 'std'],
    # #         'rank': 'mean',
    # #         'elo_after': ['min', 'max']
    # #     }).round(2)
        
    # #     athlete_stats.columns = ['competitions', 'avg_elo_change', 'elo_volatility', 'avg_rank', 'min_elo', 'peak_elo']
    # #     athlete_stats = athlete_stats.reset_index()
        
    # #     # Merge with current ratings
    # #     leaderboard = latest_ratings.merge(athlete_stats, on='name')
    # #     leaderboard = leaderboard.sort_values('current_elo', ascending=False)
        
    # #     return leaderboard
    
    # # def create_historical_elo_chart(self, athletes: List[str], era_file: str, discipline: str = None, gender: str = None) -> go.Figure:
    #     """Create historical ELO progression chart for specified athletes."""
    #     elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
    #     if elo_history.empty:
    #         return go.Figure()
        
    #     # Ensure date column is datetime
    #     elo_history['date'] = pd.to_datetime(elo_history['date'], errors='coerce')
    #     elo_history = elo_history.dropna(subset=['date'])
        
    #     # Filter for specified athletes (case-insensitive)
    #     athlete_data = elo_history[elo_history['name'].str.lower().isin([a.lower() for a in athletes])]
        
    #     if athlete_data.empty:
    #         return go.Figure()
        
    #     fig = go.Figure()
        
    #     for i, athlete in enumerate(athletes):
    #         athlete_elo = athlete_data[athlete_data['name'].str.lower() == athlete.lower()].copy()
    #         if not athlete_elo.empty:
    #             athlete_elo = athlete_elo.sort_values(['date', 'event'])
                
    #             # Get actual name with correct case
    #             actual_name = athlete_elo['name'].iloc[0]
                
    #             # Hover info
    #             hover_text = []
    #             for _, row in athlete_elo.iterrows():
    #                 hover_text.append(
    #                     f"<b>{actual_name}</b><br>"
    #                     f"Event: {row['event']}<br>"
    #                     f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
    #                     f"Rank: {row['rank']}<br>"
    #                     f"ELO: {row['elo_after']:.0f}<br>"
    #                     f"Change: {row['elo_change']:+.1f}"
    #                 )
                
    #             fig.add_trace(go.Scatter(
    #                 x=athlete_elo['date'],  # ✅ use time on x-axis
    #                 y=athlete_elo['elo_after'],
    #                 mode='lines+markers',
    #                 name=actual_name,
    #                 line=dict(color=self.chart_colors[i % len(self.chart_colors)], width=2),
    #                 marker=dict(size=4, color=self.chart_colors[i % len(self.chart_colors)]),
    #                 hovertemplate='%{hovertext}<extra></extra>',
    #                 hovertext=hover_text
    #             ))
        
    #     # Title
    #     title_parts = ["Historical ELO Progression"]
    #     if discipline:
    #         title_parts.insert(0, discipline)
    #     if gender:
    #         title_parts.insert(-1, gender)
        
    #     fig.update_layout(
    #         title=" - ".join(title_parts),
    #         xaxis_title="Date",
    #         yaxis_title="ELO Rating",
    #         hovermode='closest',
    #         height=500,
    #         showlegend=True,
    #         legend=dict(
    #             yanchor="top",
    #             y=0.99,
    #             xanchor="left",
    #             x=0.01
    #         )
    #     )
        
    #     return fig

#     ### BY COMPETITION NUMBERS ####
#     def create_historical_elo_chart(self, athletes: List[str], era_file: str, discipline: str = None, gender: str = None) -> go.Figure:
#         """Create historical ELO progression chart for specified athletes."""
#         elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
#         if elo_history.empty:
#             return go.Figure()
        
#         # Filter for specified athletes (case-insensitive)
#         athlete_data = elo_history[elo_history['name'].str.lower().isin([a.lower() for a in athletes])]
        
#         if athlete_data.empty:
#             return go.Figure()
        
#         fig = go.Figure()
        
#         for i, athlete in enumerate(athletes):
#             # Case-insensitive matching
#             athlete_elo = athlete_data[athlete_data['name'].str.lower() == athlete.lower()].copy()
#             if not athlete_elo.empty:
#                 # Sort by year and create cumulative index
#                 athlete_elo = athlete_elo.sort_values(['year', 'event'])
#                 athlete_elo['competition_number'] = range(1, len(athlete_elo) + 1)
                
#                 # Get the actual athlete name (with correct case)
#                 actual_name = athlete_elo['name'].iloc[0]
                
#                 # Add event information to hover data
#                 hover_text = []
#                 for _, row in athlete_elo.iterrows():
#                     hover_text.append(
#                         f"<b>{actual_name}</b><br>"
#                         f"Competition: {row['competition_number']}<br>"
#                         f"Event: {row['event']}<br>"
#                         f"Year: {row['year']}<br>"
#                         f"Rank: {row['rank']}<br>"
#                         f"ELO: {row['elo_after']:.0f}<br>"
#                         f"Change: {row['elo_change']:+.1f}"
#                     )
                
#                 fig.add_trace(go.Scatter(
#                     x=athlete_elo['competition_number'],
#                     y=athlete_elo['elo_after'],
#                     mode='lines+markers',
#                     name=actual_name,
#                     line=dict(color=self.chart_colors[i % len(self.chart_colors)], width=2),
#                     marker=dict(size=4, color=self.chart_colors[i % len(self.chart_colors)]),
#                     hovertemplate='%{hovertext}<extra></extra>',
#                     hovertext=hover_text
#                 ))
        
#         # Create title based on filters
#         title_parts = ["Historical ELO Progression"]
#         if discipline:
#             title_parts.insert(0, discipline)
#         if gender:
#             title_parts.insert(-1, gender)
        
#         fig.update_layout(
#             title=" - ".join(title_parts),
#             xaxis_title="Competition Number",
#             yaxis_title="ELO Rating",
#             hovermode='closest',
#             height=500,
#             showlegend=True,
#             legend=dict(
#                 yanchor="top",
#                 y=0.99,
#                 xanchor="left",
#                 x=0.01
#             )
#         )
        
#         return fig
    
    
#     def create_top_5_scatter_chart(self, era_file: str, discipline: str = None, gender: str = None) -> go.Figure:
#         """Create scatter chart showing historical ELO for top 5 current athletes."""
#         leaderboard = self.get_current_leaderboard(era_file, discipline=discipline, gender=gender)
        
#         if leaderboard.empty:
#             return go.Figure()
        
#         top_5 = leaderboard.head(5)
#         elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
#         if elo_history.empty:
#             return go.Figure()
        
#         fig = go.Figure()
        
#         for i, (_, athlete_row) in enumerate(top_5.iterrows()):
#             athlete_name = athlete_row['name']
#             athlete_elo = elo_history[elo_history['name'].str.lower() == athlete_name.lower()].copy()
            
#             if not athlete_elo.empty:
#                 athlete_elo = athlete_elo.sort_values(['year', 'event'])
#                 athlete_elo['competition_number'] = range(1, len(athlete_elo) + 1)
                
#                 # Enhanced hover information
#                 hover_text = []
#                 for _, row in athlete_elo.iterrows():
#                     hover_text.append(
#                         f"<b>{athlete_name}</b><br>"
#                         f"Competition: {row['competition_number']}<br>"
#                         f"Event: {row['event']}<br>"
#                         f"Year: {row['year']}<br>"
#                         f"Rank: {row['rank']}<br>"
#                         f"ELO: {row['elo_after']:.0f}<br>"
#                         f"Change: {row['elo_change']:+.1f}<br>"
#                         f"Current Rank: #{i+1}"
#                     )
                
#                 fig.add_trace(go.Scatter(
#                     x=athlete_elo['competition_number'],
#                     y=athlete_elo['elo_after'],
#                     mode='markers+lines',
#                     name=f"{athlete_name} (ELO: {athlete_row['current_elo']:.0f})",
#                     line=dict(color=self.chart_colors[i % len(self.chart_colors)], width=3),
#                     marker=dict(size=6, color=self.chart_colors[i % len(self.chart_colors)]),
#                     hovertemplate='%{hovertext}<extra></extra>',
#                     hovertext=hover_text
#                 ))
        
#         title_parts = []
#         if discipline:
#             title_parts.append(discipline)
#         if gender:
#             title_parts.append(gender)
#         title_parts.append("Top 5 Athletes - Historical ELO")
        
#         fig.update_layout(
#             title=" - ".join(title_parts),
#             xaxis_title="Competition Number",
#             yaxis_title="ELO Rating",
#             hovermode='closest',
#             height=600,
#             showlegend=True,
#             legend=dict(
#                 yanchor="top",
#                 y=0.99,
#                 xanchor="left",
#                 x=0.01
#             )
#         )
        
#         return fig
    
#     def get_available_combinations(self) -> Dict[str, List[str]]:
#         """Get all available discipline/gender combinations across all era files."""
#         combinations = {}
        
#         for era_file, df in self.era_files.items():
#             if df.empty:
#                 continue
            
#             # Filter to only include the 3 main disciplines
#             valid_disciplines = ['Boulder', 'Lead', 'Speed']
#             df_filtered = df[df['discipline'].isin(valid_disciplines)]
            
#             if not df_filtered.empty:
#                 combinations[era_file] = {
#                     'disciplines': sorted(df_filtered['discipline'].unique().tolist()),
#                     'genders': sorted(df_filtered['gender'].unique().tolist())
#                 }
        
#         return combinations


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

class ELOCalculator:
    """Dedicated ELO calculation and analysis for climbing competitions with fixed initialization timing."""
    
    def __init__(self, era_files: Dict[str, pd.DataFrame]):
        self.era_files = era_files
        self.elo_cache = {}  # Cache calculated ELO histories
        
        # Get colors from environment variables - consistent with app.py
        self.discipline_colors = {
            'Boulder': os.getenv('BOULDER_COLOR', '#2777AC'),
            'Lead': os.getenv('LEAD_COLOR', '#C33727'), 
            'Speed': os.getenv('SPEED_COLOR', '#1C7C44')
        }
        
        self.chart_colors = [
            os.getenv('CHART_COLOR_1', '#D24130'),
            os.getenv('CHART_COLOR_2', '#2885C3'),
            os.getenv('CHART_COLOR_3', '#1C7C44'),
            os.getenv('CHART_COLOR_4', '#A156BE'),
            os.getenv('CHART_COLOR_5', '#D98B0F'),
            os.getenv('CHART_COLOR_6', '#1ABC9C'),
            os.getenv('CHART_COLOR_7', '#E67E22'),
            os.getenv('CHART_COLOR_8', '#95A5A6'),
            os.getenv('CHART_COLOR_9', '#34495E'),
            os.getenv('CHART_COLOR_10', '#8E44AD')
        ]

    ##ChatGPT
    def calculate_elo_ratings(self, era_file: str, k_factor: int = 32, discipline: str = None, gender: str = None) -> pd.DataFrame:
        """Calculate ELO ratings for athletes with FIXED initialization timing, processed strictly in chronological order."""
        cache_key = f"{era_file}_{discipline}_{gender}_{k_factor}"
        
        if cache_key in self.elo_cache:
            return self.elo_cache[cache_key]
        
        if era_file not in self.era_files:
            return pd.DataFrame()
        
        # Load and clean data
        df = pd.read_csv(f"./Data/aggregate_data/{era_file}.csv")
        df = df.dropna(subset=['round_rank'])
        
        # Filter by discipline and gender
        if discipline:
            df = df[df['discipline'].str.lower() == discipline.lower()]
        if gender:
            df = df[df['gender'].str.lower() == gender.lower()]
        
        if df.empty:
            return pd.DataFrame()
        
        # Ensure datetime and consistent round ordering
        df['start_date'] = pd.to_datetime(df['start_date'])
        round_priority = {
            'Qualification': 0, 'Qualifier': 0, 'Qualifiers': 0, 'Q': 0,
            'Semifinal': 1, 'Semi-final': 1, 'Semi Final': 1, 'Semi': 1,
            'Final': 2,
            'Superfinal': 3, 'Super Final': 3
        }
        df['round_order'] = df['round'].map(round_priority).fillna(99)

        # Strict chronological ordering (then round order)
        df = df.sort_values(['start_date', 'event_name', 'round_order', 'round_rank'], kind='mergesort')
        
        # Map first appearance BEFORE processing competitions
        athlete_first_appearance = df.groupby('name')['start_date'].min().to_dict()

        # Initialize ELO ratings and history
        elo_ratings = {}
        elo_history = []
        
        # Group by competition in already-sorted order
        for (event, date, disc, gend, round_type), event_data in df.groupby(
            ['event_name', 'start_date', 'discipline', 'gender', 'round'], sort=False
        ):
            event_data = event_data.sort_values(['round_rank'], kind='mergesort')
            athletes = event_data['name'].tolist()
            ranks = event_data['round_rank'].tolist()
            start_date = pd.to_datetime(date)

            # Initialize new athletes with 1500 rating AT THEIR FIRST APPEARANCE DATE
            for athlete in athletes:
                if athlete not in elo_ratings:
                    elo_ratings[athlete] = 1500
                    first_date = athlete_first_appearance[athlete]
                    elo_history.append({
                        'name': athlete,
                        'event': 'Initial Rating',
                        'year': first_date.year,
                        'date': first_date,
                        'discipline': disc,
                        'gender': gend,
                        'round': 'N/A',
                        'rank': None,
                        'elo_before': 1500,
                        'elo_after': 1500,
                        'elo_change': 0,
                        'competed': False
                    })
            
            # Calculate ELO changes for competing athletes
            n_athletes = len(athletes)
            for i, athlete_a in enumerate(athletes):
                rank_a = ranks[i]
                rating_a = elo_ratings[athlete_a]
                total_change = 0
                
                for j, athlete_b in enumerate(athletes):
                    if i != j:
                        rank_b = ranks[j]
                        rating_b = elo_ratings[athlete_b]
                        
                        # Expected score
                        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                        # Actual score
                        actual_a = 1 if rank_a < rank_b else 0 if rank_a > rank_b else 0.5
                        # ELO change (averaged over opponents)
                        change = k_factor * (actual_a - expected_a) / (n_athletes - 1)
                        total_change += change
                
                # Update rating
                elo_ratings[athlete_a] += total_change
                
                # Record history
                elo_history.append({
                    'name': athlete_a,
                    'event': event,
                    'year': start_date.year,
                    'date': start_date,
                    'discipline': disc,
                    'gender': gend,
                    'round': round_type,
                    'rank': rank_a,
                    'elo_before': rating_a,
                    'elo_after': elo_ratings[athlete_a],
                    'elo_change': total_change,
                    'competed': True
                })
        
        result = pd.DataFrame(elo_history)

        # Ensure chronological order in the output (so CSVs and visuals are consistent)
        if not result.empty:
            result['date'] = pd.to_datetime(result['date'])
            # Put non-competing "Initial Rating" entries before same-day competitions
            result = result.sort_values(['date', 'competed', 'event', 'round'], ascending=[True, True, True, True], kind='mergesort')

        self.elo_cache[cache_key] = result
        return result
    ##ChatGPT   
    def get_current_leaderboard(self, era_file: str, discipline: str = None, gender: str = None) -> pd.DataFrame:
        """Get current ELO leaderboard for specified criteria."""
        elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
        if elo_history.empty:
            return pd.DataFrame()
        
        # Only competitions
        competing_history = elo_history[elo_history['competed'] == True]
        if competing_history.empty:
            return pd.DataFrame()
        
        # Latest rating by true chronology
        competing_history = competing_history.sort_values('date', kind='mergesort')
        latest_ratings = competing_history.groupby('name').last()[['elo_after']].reset_index()
        latest_ratings = latest_ratings.rename(columns={'elo_after': 'current_elo'})
        
        # Stats over competitions
        athlete_stats = competing_history.groupby('name').agg({
            'elo_change': ['count', 'mean', 'std'],
            'rank': 'mean',
            'elo_after': ['min', 'max']
        }).round(2)
        athlete_stats.columns = ['competitions', 'avg_elo_change', 'elo_volatility', 'avg_rank', 'min_elo', 'peak_elo']
        athlete_stats = athlete_stats.reset_index()
        
        leaderboard = latest_ratings.merge(athlete_stats, on='name')
        leaderboard = leaderboard.sort_values('current_elo', ascending=False, kind='mergesort')
        return leaderboard
    ##ChatGPT
    def create_historical_elo_chart(self, athletes: List[str], era_file: str, discipline: str = None, gender: str = None) -> go.Figure:
        """Create historical ELO progression chart for specified athletes (by date)."""
        elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        if elo_history.empty:
            return go.Figure()
        
        elo_history['date'] = pd.to_datetime(elo_history['date'], errors='coerce')
        elo_history = elo_history.dropna(subset=['date'])
        
        # Filter for specified athletes (case-insensitive)
        athlete_data = elo_history[elo_history['name'].str.lower().isin([a.lower() for a in athletes])]
        if athlete_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        for i, athlete in enumerate(athletes):
            athlete_elo = athlete_data[athlete_data['name'].str.lower() == athlete.lower()].copy()
            if not athlete_elo.empty:
                athlete_elo = athlete_elo.sort_values(['date', 'event'], kind='mergesort')
                actual_name = athlete_elo['name'].iloc[0]
                
                competing_data = athlete_elo[athlete_elo['competed'] == True]
                if not competing_data.empty:
                    color = self.chart_colors[i % len(self.chart_colors)]
                    hover_text = []
                    for _, row in competing_data.iterrows():
                        hover_text.append(
                            f"<b>{actual_name}</b><br>"
                            f"Event: {row['event']}<br>"
                            f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                            f"Rank: {row['rank']}<br>"
                            f"ELO: {row['elo_after']:.0f}<br>"
                            f"Change: {row['elo_change']:+.1f}"
                        )
                    
                    fig.add_trace(go.Scatter(
                        x=competing_data['date'],
                        y=competing_data['elo_after'],
                        mode='lines+markers',
                        name=actual_name,
                        line=dict(color=color, width=2),
                        marker=dict(size=6, color=color),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_text
                    ))
        
        fig.update_layout(
            # remove titile = remove spacing
            xaxis_title="Date",
            yaxis_title="ELO Rating",
            hovermode='closest',
            height=500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig


    def create_top_5_scatter_chart(self, era_file: str, discipline: str = None, gender: str = None) -> go.Figure:
        """Create scatter chart showing historical ELO for top 5 current athletes."""
        leaderboard = self.get_current_leaderboard(era_file, discipline=discipline, gender=gender)
        if leaderboard.empty:
            return go.Figure()
        
        top_5 = leaderboard.head(5)
        elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        if elo_history.empty:
            return go.Figure()
        
        fig = go.Figure()
        for i, (_, athlete_row) in enumerate(top_5.iterrows()):
            athlete_name = athlete_row['name']
            athlete_elo = elo_history[elo_history['name'].str.lower() == athlete_name.lower()].copy()
            if not athlete_elo.empty:
                # Chronological order, then derive competition_number
                athlete_elo = athlete_elo.sort_values(['date', 'event'], kind='mergesort')
                athlete_elo['competition_number'] = range(1, len(athlete_elo) + 1)
                
                hover_text = []
                for _, row in athlete_elo.iterrows():
                    hover_text.append(
                        f"<b>{athlete_name}</b><br>"
                        f"Competition: {row['competition_number']}<br>"
                        f"Event: {row['event']}<br>"
                        f"Date: {pd.to_datetime(row['date']).strftime('%Y-%m-%d')}<br>"
                        f"Rank: {row['rank']}<br>"
                        f"ELO: {row['elo_after']:.0f}<br>"
                        f"Change: {row['elo_change']:+.1f}<br>"
                        f"Current Rank: #{i+1}"
                    )
                
                fig.add_trace(go.Scatter(
                    x=athlete_elo['competition_number'],
                    y=athlete_elo['elo_after'],
                    mode='markers+lines',
                    name=f"{athlete_name} (ELO: {athlete_row['current_elo']:.0f})",
                    line=dict(color=self.chart_colors[i % len(self.chart_colors)], width=3),
                    marker=dict(size=6, color=self.chart_colors[i % len(self.chart_colors)]),
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_text
                ))
        
        title_parts = []
        if discipline:
            title_parts.append(discipline)
        if gender:
            title_parts.append(gender)
        title_parts.append("Top 5 Athletes - Historical ELO")
        
        fig.update_layout(
            title=" - ".join(title_parts),
            xaxis_title="Competition Number",
            yaxis_title="ELO Rating",
            hovermode='closest',
            height=600,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig



    def get_available_combinations(self) -> Dict[str, List[str]]:
        """Get all available discipline/gender combinations across all era files."""
        combinations = {}
        
        for era_file, df in self.era_files.items():
            if df.empty:
                continue
            
            # Filter to only include the 3 main disciplines
            valid_disciplines = ['Boulder', 'Lead', 'Speed']
            df_filtered = df[df['discipline'].isin(valid_disciplines)]
            
            if not df_filtered.empty:
                combinations[era_file] = {
                    'disciplines': sorted(df_filtered['discipline'].unique().tolist()),
                    'genders': sorted(df_filtered['gender'].unique().tolist())
                }
        
        return combinations


# OPTIMIZED EXPORT FUNCTIONS

def create_time_based_elo_history(elo_history, era_key):
    """
    Optimized version: Only create time series records at competition dates + key milestones
    This is much faster than creating monthly records for every athlete
    """
    if elo_history.empty:
        return pd.DataFrame()
    
    print(f"  Optimizing time-based history for {len(elo_history)} records...")
    
    # Ensure we have date information
    if 'date' not in elo_history.columns:
        elo_history['date'] = pd.to_datetime(elo_history['year'].astype(str) + '-01-01')
        print(f"  Warning: No date column found for {era_key}, using year approximation")
    else:
        elo_history['date'] = pd.to_datetime(elo_history['date'])
    
    # Sort by athlete and date
    elo_history = elo_history.sort_values(['name', 'date'])
    
    time_based_records = []
    
    # Get all unique competition dates
    competition_dates = sorted(elo_history['date'].unique())
    
    # Add year boundaries for continuity
    min_year = elo_history['date'].dt.year.min()
    max_year = elo_history['date'].dt.year.max()
    year_boundaries = [pd.Timestamp(f'{year}-01-01') for year in range(min_year, max_year + 1)]
    year_boundaries = [d for d in year_boundaries if d not in competition_dates]
    
    # Combine and sort all key dates
    key_dates = sorted(list(competition_dates) + year_boundaries)
    
    print(f"  Processing {len(key_dates)} key dates instead of full monthly series...")
    
    # Process each athlete
    for athlete in elo_history['name'].unique():
        athlete_data = elo_history[elo_history['name'] == athlete].copy()
        athlete_data = athlete_data.sort_values('date')
        
        current_elo = 1500  # Default starting ELO
        competitions_count = 0
        last_competition_date = None
        
        for date in key_dates:
            # Find competitions on or before this date
            competitions_before = athlete_data[athlete_data['date'] <= date]
            
            if not competitions_before.empty:
                # Use the most recent ELO rating
                latest_competition = competitions_before.iloc[-1]
                current_elo = latest_competition['elo_after']
                competitions_count = len(competitions_before)
                last_competition_date = latest_competition['date']
            
            # Only add record if this athlete has competed at some point
            if competitions_count > 0 or date in competition_dates:
                time_based_records.append({
                    'name': athlete,
                    'date': date,
                    'elo': current_elo,
                    'has_competed': competitions_count > 0,
                    'competitions_to_date': competitions_count,
                    'last_competition_date': last_competition_date,
                    'days_since_last_competition': (date - last_competition_date).days if last_competition_date else None,
                    'is_competition_date': date in competition_dates,
                    'is_year_boundary': date in year_boundaries
                })
    
    result_df = pd.DataFrame(time_based_records)
    print(f"  Created {len(result_df)} optimized time-based records")
    return result_df

def create_minimal_time_series(elo_history, era_key):
    """
    Even faster alternative: Just add forward-fill records at year boundaries
    This preserves ELO continuity without massive data expansion
    """
    if elo_history.empty:
        return pd.DataFrame()
    
    print(f"  Creating minimal time series for {era_key}...")
    
    # Ensure dates
    if 'date' not in elo_history.columns:
        elo_history['date'] = pd.to_datetime(elo_history['year'].astype(str) + '-01-01')
    else:
        elo_history['date'] = pd.to_datetime(elo_history['date'])
    
    # Sort by athlete and date
    elo_history = elo_history.sort_values(['name', 'date'])
    
    # Get the last ELO for each athlete in each year
    elo_history['year'] = elo_history['date'].dt.year
    
    # For each athlete, get their last ELO rating of each year
    yearly_elo = elo_history.groupby(['name', 'year']).last().reset_index()
    
    # Create year-end records
    yearly_elo['date'] = pd.to_datetime(yearly_elo['year'].astype(str) + '-12-31')
    yearly_elo['is_year_end'] = True
    
    # Combine original records with year-end records
    original_records = elo_history.copy()
    original_records['is_year_end'] = False
    
    # Select relevant columns for consistency
    columns = ['name', 'date', 'elo_after', 'year']
    if 'competed' in elo_history.columns:
        columns.append('competed')
    
    # Rename elo_after to elo for consistency
    yearly_records = yearly_elo[columns].copy()
    yearly_records = yearly_records.rename(columns={'elo_after': 'elo'})
    yearly_records['record_type'] = 'year_end'
    
    original_records = original_records[columns].copy()
    original_records = original_records.rename(columns={'elo_after': 'elo'})
    original_records['record_type'] = 'competition'
    
    # Combine and sort
    combined = pd.concat([original_records, yearly_records], ignore_index=True)
    combined = combined.sort_values(['name', 'date']).drop_duplicates(['name', 'date'])
    
    print(f"  Created {len(combined)} minimal time series records")
    return combined

def export_all_elo_data(analyzer, base_dir):
    """Optimized version of the ELO export with FIXED initialization timing"""
    
    # Get overview to understand available data
    overview = analyzer.get_data_overview()
    era_files = overview.get('era_files', [])
    
    if not era_files:
        print("No era files found")
        return
    
    # Define discipline eras for systematic export
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
            ("IFSC_Time", 2009, 2025),
        ]
    }
    
    disciplines = ['Boulder', 'Lead', 'Speed']
    genders = ['Men', 'Women']
    
    # Initialize ELO Calculator with era data
    era_data = {}
    for era_file in era_files:
        if hasattr(analyzer, 'get_era_data'):
            era_data[era_file] = analyzer.get_era_data(era_file)
        elif hasattr(analyzer, 'era_files'):
            era_data = analyzer.era_files
            break
    
    if not era_data:
        print("Could not access era data from analyzer")
        return
    
    # Use the FIXED ELO Calculator
    
    elo_calculator = ELOCalculator(era_data)
    
    # Export summary
    export_summary = {
        "export_timestamp": datetime.now().isoformat(),
        "total_combinations": 0,
        "successful_exports": 0,
        "failed_exports": [],
        "exported_files": [],
        "optimization_used": "minimal_time_series",
        "fix_applied": "athlete_initialization_timing_corrected"
    }
    
    print("Starting FIXED ELO data export with corrected initialization timing...")
    
    for discipline in disciplines:
        for gender in genders:
            for era_name, start_year, end_year in DISCIPLINE_ERAS.get(discipline, []):
                
                # Create era key
                era_key = f"{discipline}_{era_name}_{start_year}-{end_year}_{gender}"
                export_summary["total_combinations"] += 1
                
                print(f"\nProcessing: {era_key}")
                
                try:
                    # Calculate ELO ratings with FIXED initialization
                    elo_history = elo_calculator.calculate_elo_ratings(
                        era_key, 
                        discipline=discipline, 
                        gender=gender
                    )
                    
                    if elo_history.empty:
                        print(f"  No data found for {era_key}")
                        export_summary["failed_exports"].append({
                            "era_key": era_key,
                            "reason": "No data found"
                        })
                        continue
                    
                    # Get current leaderboard
                    leaderboard = elo_calculator.get_current_leaderboard(
                        era_key,
                        discipline=discipline,
                        gender=gender
                    )
                    
                    # Create safe filename
                    safe_filename = era_key.replace(" ", "_").replace("/", "_")
                    
                    # Export all files as before, but now with FIXED timing
                    # Export 1: Current Leaderboard
                    leaderboard_file = base_dir / "leaderboards" / f"{safe_filename}_leaderboard.csv"
                    leaderboard.to_csv(leaderboard_file, index=False)
                    export_summary["exported_files"].append(str(leaderboard_file))
                    
                    # Export 2: Complete Historical Data (competition-based)
                    history_file = base_dir / "historical_data" / f"{safe_filename}_history.csv"
                    elo_history.to_csv(history_file, index=False)
                    export_summary["exported_files"].append(str(history_file))
                    
                    # Export 3: OPTIMIZED Time-based ELO History
                    print("  Creating optimized time-based ELO history...")
                    start_time = datetime.now()
                    
                    time_based_history = create_minimal_time_series(elo_history, era_key)
                    
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"  Time series creation took {elapsed:.2f} seconds")
                    
                    if not time_based_history.empty:
                        time_series_file = base_dir / "time_series" / f"{safe_filename}_timeseries.csv"
                        time_based_history.to_csv(time_series_file, index=False)
                        export_summary["exported_files"].append(str(time_series_file))
                        
                        # Export top 10 time series for visualization
                        if not leaderboard.empty:
                            top_10_names = leaderboard.head(10)['name'].tolist()
                            top_10_time_series = time_based_history[
                                time_based_history['name'].isin(top_10_names)
                            ]
                            
                            charts_time_file = base_dir / "charts_data" / f"{safe_filename}_top10_timeseries.csv"
                            top_10_time_series.to_csv(charts_time_file, index=False)
                            export_summary["exported_files"].append(str(charts_time_file))
                    
                    # Export 4: Statistics Summary (enhanced)
                    stats = {
                        "era_key": era_key,
                        "discipline": discipline,
                        "gender": gender,
                        "era_name": era_name,
                        "start_year": start_year,
                        "end_year": end_year,
                        "total_athletes": len(leaderboard),
                        "total_competitions": elo_history['event'].nunique() if 'event' in elo_history.columns else 0,
                        "total_elo_records": len(elo_history),
                        "avg_elo": float(leaderboard['current_elo'].mean()) if not leaderboard.empty else 0,
                        "median_elo": float(leaderboard['current_elo'].median()) if not leaderboard.empty else 0,
                        "max_elo": float(leaderboard['current_elo'].max()) if not leaderboard.empty else 0,
                        "min_elo": float(leaderboard['current_elo'].min()) if not leaderboard.empty else 0,
                        "most_active_athlete": leaderboard.loc[leaderboard['competitions'].idxmax(), 'name'] if not leaderboard.empty else "",
                        "most_active_competitions": int(leaderboard['competitions'].max()) if not leaderboard.empty else 0,
                        "highest_elo_athlete": leaderboard.loc[leaderboard['current_elo'].idxmax(), 'name'] if not leaderboard.empty else "",
                        "years_covered": sorted(elo_history['year'].unique().tolist()) if 'year' in elo_history.columns else [],
                        "date_range": {
                            "start": elo_history['date'].min().isoformat() if 'date' in elo_history.columns else None,
                            "end": elo_history['date'].max().isoformat() if 'date' in elo_history.columns else None
                        } if 'date' in elo_history.columns else None,
                        "time_series_records": len(time_based_history) if not time_based_history.empty else 0,
                        "time_series_generation_seconds": elapsed,
                        "initialization_fix_applied": True
                    }
                    
                    stats_file = base_dir / "statistics" / f"{safe_filename}_stats.json"
                    with open(stats_file, 'w') as f:
                        json.dump(stats, f, indent=2, default=str)
                    export_summary["exported_files"].append(str(stats_file))
                    
                    # Export remaining files as before...
                    if not leaderboard.empty:
                        top_10 = leaderboard.head(10)
                        top_10_history = elo_history[elo_history['name'].isin(top_10['name'])]
                        
                        charts_file = base_dir / "charts_data" / f"{safe_filename}_top10_history.csv"
                        top_10_history.to_csv(charts_file, index=False)
                        export_summary["exported_files"].append(str(charts_file))
                    
                    raw_file = base_dir / "raw_calculations" / f"{safe_filename}_raw.csv"
                    elo_history.to_csv(raw_file, index=False)
                    export_summary["exported_files"].append(str(raw_file))
                    
                    print(f"  ✓ FIXED export: {len(leaderboard)} athletes, {len(elo_history)} ELO records")
                    if not time_based_history.empty:
                        print(f"    Time series: {len(time_based_history)} records ({elapsed:.2f}s)")
                    export_summary["successful_exports"] += 1
                    
                except Exception as e:
                    print(f"  ✗ Error processing {era_key}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    export_summary["failed_exports"].append({
                        "era_key": era_key,
                        "reason": str(e)
                    })
    
    # Export master summary
    summary_file = base_dir / "export_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(export_summary, f, indent=2, default=str)
    
    # Create simplified master files
    print("\nCreating master combined files...")
    
    try:
        # Only combine leaderboards and essential files to avoid memory issues
        all_leaderboards = []
        
        for file_path in export_summary["exported_files"]:
            if "leaderboard.csv" in file_path:
                df = pd.read_csv(file_path)
                era_info = file_path.split("/")[-1].replace("_leaderboard.csv", "").split("_")
                if len(era_info) >= 4:
                    df['discipline'] = era_info[0]
                    df['era_name'] = era_info[1]
                    df['gender'] = era_info[-1]
                all_leaderboards.append(df)
        
        if all_leaderboards:
            master_leaderboard = pd.concat(all_leaderboards, ignore_index=True)
            master_leaderboard.to_csv(base_dir / "master_leaderboard.csv", index=False)
            print(f"  ✓ Master leaderboard: {len(master_leaderboard)} records")
            
    except Exception as e:
        print(f"  ✗ Error creating master files: {e}")
    
    return export_summary

def main():
    """Main function to run the FIXED ELO Data Export Script"""
    print("FIXED ELO Data Export Script - Corrected Initialization Timing")
    print("=" * 60)
    
    # Load analyzer (assuming this is already defined in your script)
    try:
        from analysis import ClimbingAnalyzer
        analyzer = ClimbingAnalyzer()
    except ImportError:
        print("Error: Could not import ClimbingAnalyzer from analysis module")
        print("Please ensure the analysis.py file is in the same directory")
        return
    
    # Create export structure
    from pathlib import Path
    base_dir = Path(".") / "ELO_data"
    subdirs = ["leaderboards", "historical_data", "statistics", "charts_data", "raw_calculations", "time_series"]
    
    # Create directories
    for subdir in [base_dir] + [base_dir / sub for sub in subdirs]:
        subdir.mkdir(exist_ok=True)
        print(f"Created directory: {subdir}")
    
    # Export all ELO data with FIXED initialization timing
    export_summary = export_all_elo_data(analyzer, base_dir)
    
    print("\n" + "=" * 60)
    print("FIXED ELO Export Complete!")
    print(f"Successful exports: {export_summary['successful_exports']}")
    print(f"Failed exports: {len(export_summary['failed_exports'])}")
    print(f"Total files created: {len(export_summary['exported_files'])}")
    print(f"Export location: {base_dir.absolute()}")
    
    if export_summary['failed_exports']:
        print("\nFailed exports:")
        for failure in export_summary['failed_exports']:
            print(f"  - {failure['era_key']}: {failure['reason']}")

if __name__ == "__main__":
    main()