import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ELOCalculator:
    """Dedicated ELO calculation and analysis for climbing competitions."""
    
    def __init__(self, era_files: Dict[str, pd.DataFrame]):
        self.era_files = era_files
        self.elo_cache = {}  # Cache calculated ELO histories
    
    def calculate_elo_ratings(self, era_file: str, k_factor: int = 32, discipline: str = None, gender: str = None) -> pd.DataFrame:
        """Calculate ELO ratings for athletes in a specific era with optional filters."""
        cache_key = f"{era_file}_{discipline}_{gender}_{k_factor}"
        
        if cache_key in self.elo_cache:
            return self.elo_cache[cache_key]
        
        if era_file not in self.era_files:
            return pd.DataFrame()
        
        df = self.era_files[era_file].copy()
        df = df.dropna(subset=['round_rank'])
        
        # Filter by discipline and gender
        if discipline:
            df = df[df['discipline'].str.lower() == discipline.lower()]
        if gender:
            df = df[df['gender'].str.lower() == gender.lower()]
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.sort_values(['year', 'start_date', 'event_name'])
        
        # Initialize ELO ratings
        elo_ratings = {}
        elo_history = []
        
        # Group by competition
        for (event, year, disc, gend, round_type), event_data in df.groupby(['event_name', 'year', 'discipline', 'gender', 'round']):
            event_data = event_data.sort_values('round_rank')
            athletes = event_data['name'].tolist()
            ranks = event_data['round_rank'].tolist()
            
            # Initialize new athletes with 1200 rating
            for athlete in athletes:
                if athlete not in elo_ratings:
                    elo_ratings[athlete] = 1500
            
            # Calculate ELO changes
            n_athletes = len(athletes)
            for i, athlete_a in enumerate(athletes):
                rank_a = ranks[i]
                rating_a = elo_ratings[athlete_a]
                
                # Compare against all other athletes
                total_change = 0
                for j, athlete_b in enumerate(athletes):
                    if i != j:
                        rank_b = ranks[j]
                        rating_b = elo_ratings[athlete_b]
                        
                        # Expected score (better rank = higher expected score)
                        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                        
                        # Actual score (better rank wins)
                        actual_a = 1 if rank_a < rank_b else 0 if rank_a > rank_b else 0.5
                        
                        # ELO change
                        change = k_factor * (actual_a - expected_a) / (n_athletes - 1)
                        total_change += change
                
                # Update rating
                elo_ratings[athlete_a] += total_change
                
                # Record history
                elo_history.append({
                    'name': athlete_a,
                    'event': event,
                    'year': year,
                    'discipline': disc,
                    'gender': gend,
                    'round': round_type,
                    'rank': rank_a,
                    'elo_before': rating_a,
                    'elo_after': elo_ratings[athlete_a],
                    'elo_change': total_change
                })
        
        result = pd.DataFrame(elo_history)
        self.elo_cache[cache_key] = result
        return result
    
    def get_current_leaderboard(self, era_file: str, discipline: str = None, gender: str = None) -> pd.DataFrame:
        """Get current ELO leaderboard for specified criteria."""
        elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
        if elo_history.empty:
            return pd.DataFrame()
        
        # Get latest rating for each athlete
        latest_ratings = elo_history.groupby('name').last()[['elo_after']].reset_index()
        latest_ratings = latest_ratings.rename(columns={'elo_after': 'current_elo'})
        
        # Add additional stats
        athlete_stats = elo_history.groupby('name').agg({
            'elo_change': ['count', 'mean', 'std'],
            'rank': 'mean',
            'elo_after': ['min', 'max']
        }).round(2)
        
        athlete_stats.columns = ['competitions', 'avg_elo_change', 'elo_volatility', 'avg_rank', 'min_elo', 'peak_elo']
        athlete_stats = athlete_stats.reset_index()
        
        # Merge with current ratings
        leaderboard = latest_ratings.merge(athlete_stats, on='name')
        leaderboard = leaderboard.sort_values('current_elo', ascending=False)
        
        return leaderboard
    
    def create_historical_elo_chart(self, athletes: List[str], era_file: str, discipline: str = None, gender: str = None) -> go.Figure:
        """Create historical ELO progression chart for specified athletes."""
        elo_history = self.calculate_elo_ratings(era_file, discipline=discipline, gender=gender)
        
        if elo_history.empty:
            return go.Figure()
        
        # Filter for specified athletes (case-insensitive)
        athlete_data = elo_history[elo_history['name'].str.lower().isin([a.lower() for a in athletes])]
        
        if athlete_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Consistent colors
        colors = ['#3498DB', '#FF8C00', '#27AE60', '#E74C3C', '#9B59B6', '#1ABC9C', '#E67E22', '#95A5A6', '#34495E', '#8E44AD']
        
        for i, athlete in enumerate(athletes):
            # Case-insensitive matching
            athlete_elo = athlete_data[athlete_data['name'].str.lower() == athlete.lower()].copy()
            if not athlete_elo.empty:
                # Sort by year and create cumulative index
                athlete_elo = athlete_elo.sort_values(['year', 'event'])
                athlete_elo['competition_number'] = range(1, len(athlete_elo) + 1)
                
                # Get the actual athlete name (with correct case)
                actual_name = athlete_elo['name'].iloc[0]
                
                fig.add_trace(go.Scatter(
                    x=athlete_elo['competition_number'],
                    y=athlete_elo['elo_after'],
                    mode='lines+markers',
                    name=actual_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Competition: %{x}<br>' +
                                'ELO: %{y:.0f}<br>' +
                                '<extra></extra>'
                ))
        
        fig.update_layout(
            title="Historical ELO Progression",
            xaxis_title="Competition Number",
            yaxis_title="ELO Rating",
            hovermode='closest',
            height=500,
            showlegend=True
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
        
        # Color mapping based on discipline
        discipline_colors = {'Boulder': '#3498DB', 'Lead': '#FF8C00', 'Speed': '#27AE60'}
        colors = ['#3498DB', '#FF8C00', '#27AE60', '#E74C3C', '#9B59B6']
        
        for i, (_, athlete_row) in enumerate(top_5.iterrows()):
            athlete_name = athlete_row['name']
            athlete_elo = elo_history[elo_history['name'].str.lower() == athlete_name.lower()].copy()
            
            if not athlete_elo.empty:
                athlete_elo = athlete_elo.sort_values(['year', 'event'])
                athlete_elo['competition_number'] = range(1, len(athlete_elo) + 1)
                
                fig.add_trace(go.Scatter(
                    x=athlete_elo['competition_number'],
                    y=athlete_elo['elo_after'],
                    mode='markers+lines',
                    name=f"{athlete_name} (ELO: {athlete_row['current_elo']:.0f})",
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6, color=colors[i % len(colors)]),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Competition: %{x}<br>' +
                                'ELO: %{y:.0f}<br>' +
                                '<extra></extra>'
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
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
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