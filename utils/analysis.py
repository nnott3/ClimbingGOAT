


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class ClimbingAnalyzer:
    """Advanced analysis for climbing competition data with ELO calculations."""
    
    def __init__(self, data_dir: str = "./Data"):
        self.data_dir = Path(data_dir)
        self.aggregated_df = None
        self.era_files = {}
        self.metadata_df = None
        self.elo_ratings = {}
        
        # Get color scheme from environment variables
        self.discipline_colors = {
            'Boulder': os.getenv('BOULDER_COLOR', '#3498DB'),
            'Lead': os.getenv('LEAD_COLOR', '#E74C3C'),
            'Speed': os.getenv('SPEED_COLOR', '#27AE60'),
            'Combined': os.getenv('COMBINED_COLOR', '#9B59B6')
        }
        
        # Load all data
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all CSV files from the data directory."""
        print(f"Looking for data in: {self.data_dir}")
        
        # Check if data directory exists
        if not self.data_dir.exists():
            print(f"Data directory not found: {self.data_dir}")
            return
        
        # Load aggregated results
        agg_file = self.data_dir / "raw_data" / "aggregated_results.csv"
        print(f"Looking for aggregated file: {agg_file}")
        
        if agg_file.exists():
            try:
                self.aggregated_df = pd.read_csv(agg_file)
                print(f"Loaded aggregated data: {len(self.aggregated_df)} records")
                self._clean_aggregated_data()
            except Exception as e:
                print(f"Error loading aggregated data: {e}")
        else:
            print("Aggregated results file not found")
        
        # Load era-specific files
        raw_data_dir = self.data_dir / "raw_data"
        if raw_data_dir.exists():
            for csv_file in raw_data_dir.glob("*.csv"):
                if csv_file.name != "aggregated_results.csv":
                    try:
                        era_name = csv_file.stem
                        self.era_files[era_name] = pd.read_csv(csv_file)
                        print(f"Loaded era file: {era_name} ({len(self.era_files[era_name])} records)")
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")
        
        # Load metadata
        metadata_file = self.data_dir / "data_summary" / "event_metadata.csv"
        if metadata_file.exists():
            try:
                self.metadata_df = pd.read_csv(metadata_file)
                print(f"Loaded metadata: {len(self.metadata_df)} records")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        else:
            print("Metadata file not found")
    
    def _clean_aggregated_data(self):
        """Clean and prepare aggregated data for analysis."""
        if self.aggregated_df is None:
            return
        
        # Convert dates
        date_cols = ['start_date', 'comp_date']
        for col in date_cols:
            if col in self.aggregated_df.columns:
                self.aggregated_df[col] = pd.to_datetime(self.aggregated_df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['year', 'round_rank']
        for col in numeric_cols:
            if col in self.aggregated_df.columns:
                self.aggregated_df[col] = pd.to_numeric(self.aggregated_df[col], errors='coerce')
        
        # Clean athlete names and countries
        if 'name' in self.aggregated_df.columns:
            self.aggregated_df['name'] = self.aggregated_df['name'].str.strip()
        if 'country' in self.aggregated_df.columns:
            self.aggregated_df['country'] = self.aggregated_df['country'].str.strip().str.upper()
    
    def get_data_overview(self) -> Dict:
        """Get comprehensive data overview statistics."""
        if self.aggregated_df is None:
            
            return {}
        
        # Filter to only include the 3 main disciplines
        valid_disciplines = ['Boulder', 'Lead', 'Speed']
        filtered_df = self.aggregated_df[self.aggregated_df['discipline'].isin(valid_disciplines)]
        
        overview = {
            'total_records': len(filtered_df),
            'unique_athletes': filtered_df['name'].nunique(),
            'unique_countries': filtered_df['country'].nunique(),
            'year_range': (int(filtered_df['year'].min()), int(filtered_df['year'].max())),
            'disciplines': filtered_df['discipline'].value_counts().to_dict(),
            'genders': filtered_df['gender'].value_counts().to_dict(),
            'rounds': filtered_df['round'].value_counts().to_dict(),
            'era_files': list(self.era_files.keys())
        }
        
        return overview
    
    def filter_data(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply comprehensive filters to dataframe."""
        filtered_df = df.copy()
        
        # Year range filter
        if 'year_range' in filters and filters['year_range']:
            start_year, end_year = filters['year_range']
            filtered_df = filtered_df[
                (filtered_df['year'] >= start_year) & 
                (filtered_df['year'] <= end_year)
            ]
        
        # Discipline filter
        if 'disciplines' in filters and filters['disciplines']:
            filtered_df = filtered_df[filtered_df['discipline'].isin(filters['disciplines'])]
        
        # Gender filter
        if 'genders' in filters and filters['genders']:
            filtered_df = filtered_df[filtered_df['gender'].isin(filters['genders'])]
        
        # Country filter
        if 'countries' in filters and filters['countries']:
            filtered_df = filtered_df[filtered_df['country'].isin(filters['countries'])]
        
        # Athlete filter
        if 'athletes' in filters and filters['athletes']:
            filtered_df = filtered_df[filtered_df['name'].isin(filters['athletes'])]
        
        # Round filter
        if 'rounds' in filters and filters['rounds']:
            filtered_df = filtered_df[filtered_df['round'].isin(filters['rounds'])]
        
        # Location filter
        if 'locations' in filters and filters['locations']:
            filtered_df = filtered_df[filtered_df['location'].isin(filters['locations'])]
        
        return filtered_df
    
    def get_athlete_stats(self, filters: Dict = None) -> pd.DataFrame:
        """Get comprehensive athlete statistics."""
        if self.aggregated_df is None:
            return pd.DataFrame()
        
        df = self.aggregated_df.copy()
        if filters:
            df = self.filter_data(df, filters)
        
        athlete_stats = df.groupby(['name', 'country']).agg({
            'round_rank': ['count', 'mean', 'median', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum(), lambda x: (x <= 8).sum()],
            'year': ['min', 'max', 'nunique'],
            'discipline': ['nunique', lambda x: list(x.unique())],
            'event_name': 'nunique',
            'location': 'nunique'
        }).round(2)
        
        # Flatten column names
        athlete_stats.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in athlete_stats.columns]
        athlete_stats = athlete_stats.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'count_round_rank': 'total_competitions',
            'mean_round_rank': 'avg_rank',
            'median_round_rank': 'median_rank',
            '<lambda_0>_round_rank': 'wins',
            '<lambda_1>_round_rank': 'podiums',
            '<lambda_2>_round_rank': 'top8_finishes',
            'min_year': 'career_start',
            'max_year': 'career_end',
            'nunique_year': 'active_years',
            'nunique_discipline': 'disciplines_competed',
            '<lambda>_discipline': 'disciplines_list',
            'nunique_event_name': 'unique_events',
            'nunique_location': 'countries_competed'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in athlete_stats.columns:
                athlete_stats = athlete_stats.rename(columns={old_name: new_name})
        
        # Calculate career length
        athlete_stats['career_length'] = athlete_stats['career_end'] - athlete_stats['career_start'] + 1
        
        # Calculate win rate and podium rate
        athlete_stats['win_rate'] = (athlete_stats['wins'] / athlete_stats['total_competitions'] * 100).round(2)
        athlete_stats['podium_rate'] = (athlete_stats['podiums'] / athlete_stats['total_competitions'] * 100).round(2)
        
        return athlete_stats.sort_values('total_competitions', ascending=False)
    
    def get_country_stats(self, filters: Dict = None) -> pd.DataFrame:
        """Get comprehensive country statistics."""
        if self.aggregated_df is None:
            return pd.DataFrame()
        
        df = self.aggregated_df.copy()
        if filters:
            df = self.filter_data(df, filters)
        
        country_stats = df.groupby('country').agg({
            'name': 'nunique',
            'round_rank': ['count', 'mean', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum(), lambda x: (x <= 8).sum()],
            'year': ['min', 'max', 'nunique'],
            'discipline': 'nunique',
            'event_name': 'nunique',
            'location': 'nunique'
        }).round(2)
        
        # Flatten and rename columns
        country_stats.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in country_stats.columns]
        country_stats = country_stats.reset_index()
        
        column_mapping = {
            'nunique_name': 'total_athletes',
            'count_round_rank': 'total_participations',
            'mean_round_rank': 'avg_rank',
            '<lambda_0>_round_rank': 'total_wins',
            '<lambda_1>_round_rank': 'total_podiums',
            '<lambda_2>_round_rank': 'total_top8',
            'nunique_discipline': 'disciplines_active',
            'nunique_event_name': 'events_participated',
            'nunique_location': 'host_countries'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in country_stats.columns:
                country_stats = country_stats.rename(columns={old_name: new_name})
        
        # Calculate rates
        country_stats['wins_per_athlete'] = (country_stats['total_wins'] / country_stats['total_athletes']).round(2)
        country_stats['podiums_per_athlete'] = (country_stats['total_podiums'] / country_stats['total_athletes']).round(2)
        
        return country_stats.sort_values('total_athletes', ascending=False)
    
    def create_athlete_timeline(self, athlete_name: str) -> go.Figure:
        """Create timeline visualization for specific athlete."""
        if self.aggregated_df is None:
            return go.Figure()
        
        # Filter to only include the 3 main disciplines
        valid_disciplines = ['Boulder', 'Lead', 'Speed']
        filtered_df = self.aggregated_df[self.aggregated_df['discipline'].isin(valid_disciplines)]
        
        # Case-insensitive athlete name matching
        athlete_data = filtered_df[filtered_df['name'].str.lower() == athlete_name.lower()].copy()
        if athlete_data.empty:
            return go.Figure()
        
        # Sort by date
        athlete_data = athlete_data.sort_values(['year', 'start_date'])
        
        fig = go.Figure()
        
        for discipline in athlete_data['discipline'].unique():
            disc_data = athlete_data[athlete_data['discipline'] == discipline]
            
            fig.add_trace(go.Scatter(
                x=disc_data['year'],
                y=disc_data['round_rank'],
                mode='markers+lines',
                name=discipline,
                marker=dict(
                    color=self.discipline_colors.get(discipline, '#95A5A6'),
                    size=8,
                    symbol='circle'
                ),
                text=disc_data['event_name'] + '<br>' + disc_data['location'],
                hovertemplate='<b>%{text}</b><br>Rank: %{y}<br>Year: %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Competition Timeline - {athlete_data['name'].iloc[0]}",
            xaxis_title="Year",
            yaxis_title="Rank",
            yaxis=dict(autorange='reversed'),
            hovermode='closest',
            height=500
        )
        
        return fig
    
    def create_country_discipline_heatmap(self, filters: Dict = None, metric: str = 'wins') -> go.Figure:
        """Create heatmap of country performance by discipline."""
        if self.aggregated_df is None:
            return go.Figure()
        
        df = self.aggregated_df.copy()
        if filters:
            df = self.filter_data(df, filters)
        
        # Calculate metrics by country and discipline
        if metric == 'wins':
            heatmap_data = df.groupby(['country', 'discipline']).apply(
                lambda x: (x['round_rank'] == 1).sum()
            ).unstack(fill_value=0)
            title = "Total Wins by Country and Discipline"
            colorscale = 'Reds'
        elif metric == 'podiums':
            heatmap_data = df.groupby(['country', 'discipline']).apply(
                lambda x: (x['round_rank'] <= 3).sum()
            ).unstack(fill_value=0)
            title = "Total Podiums by Country and Discipline"
            colorscale = 'Blues'
        else:
            heatmap_data = df.groupby(['country', 'discipline'])['round_rank'].mean().round(2).unstack(fill_value=np.nan)
            title = "Average Rank by Country and Discipline"
            colorscale = 'RdYlBu_r'
        
        # Get top countries for better visualization
        if metric in ['wins', 'podiums']:
            top_countries = heatmap_data.sum(axis=1).nlargest(15).index
            heatmap_data = heatmap_data.loc[top_countries]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=colorscale,
            text=heatmap_data.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y} - %{x}</b><br>" + metric.capitalize() + ": %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Discipline",
            yaxis_title="Country",
            height=max(400, len(heatmap_data.index) * 25)
        )
        
        return fig
    
    def create_participation_trends(self, filters: Dict = None) -> go.Figure:
        """Create participation trends over time."""
        if self.aggregated_df is None:
            return go.Figure()
        
        # Filter to only include the 3 main disciplines
        valid_disciplines = ['Boulder', 'Lead', 'Speed']
        df = self.aggregated_df[self.aggregated_df['discipline'].isin(valid_disciplines)].copy()
        
        if filters:
            df = self.filter_data(df, filters)
        
        yearly_participation = df.groupby(['year', 'discipline']).agg({
            'name': 'nunique',
            'country': 'nunique'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Unique Athletes per Year', 'Countries Represented per Year'),
            vertical_spacing=0.1
        )
        
        disciplines = yearly_participation['discipline'].unique()
        
        for discipline in disciplines:
            disc_data = yearly_participation[yearly_participation['discipline'] == discipline]
            color = self.discipline_colors.get(discipline, '#95A5A6')
            
            # Athletes plot
            fig.add_trace(
                go.Scatter(
                    x=disc_data['year'],
                    y=disc_data['name'],
                    mode='lines+markers',
                    name=f"{discipline} - Athletes",
                    line=dict(color=color),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Countries plot
            fig.add_trace(
                go.Scatter(
                    x=disc_data['year'],
                    y=disc_data['country'],
                    mode='lines+markers',
                    name=f"{discipline} - Countries",
                    line=dict(color=color, dash='dash'),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            title_text="Competition Participation Trends Over Time"
        )
        
        return fig
    
    def calculate_basic_elo(self, era_file: str, k_factor: int = 32) -> pd.DataFrame:
        """Calculate basic ELO ratings for athletes in a specific era."""
        if era_file not in self.era_files:
            return pd.DataFrame()
        
        df = self.era_files[era_file].copy()
        df = df.dropna(subset=['round_rank'])
        df = df.sort_values(['year', 'start_date', 'event_name'])
        
        # Initialize ELO ratings
        elo_ratings = {}
        elo_history = []
        
        # Group by competition
        for (event, year, discipline, gender, round_type), event_data in df.groupby(['event_name', 'year', 'discipline', 'gender', 'round']):
            event_data = event_data.sort_values('round_rank')
            athletes = event_data['name'].tolist()
            ranks = event_data['round_rank'].tolist()
            
            # Initialize new athletes with 1500 rating
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
                    'discipline': discipline,
                    'gender': gender,
                    'round': round_type,
                    'rank': rank_a,
                    'elo_before': rating_a,
                    'elo_after': elo_ratings[athlete_a],
                    'elo_change': total_change
                })
        
        return pd.DataFrame(elo_history)
    
    def get_elo_leaderboard(self, era_file: str, discipline: str = None, gender: str = None) -> pd.DataFrame:
        """Get current ELO leaderboard for an era, optionally filtered by discipline and gender."""
        elo_history = self.calculate_basic_elo(era_file)
        
        if elo_history.empty:
            return pd.DataFrame()
        
        # Filter by discipline and gender if specified
        if discipline:
            elo_history = elo_history[elo_history['discipline'] == discipline]
        if gender:
            elo_history = elo_history[elo_history['gender'] == gender]
        
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
    
    def create_historical_elo_chart(self, athletes: List[str], era_file: str) -> go.Figure:
        """Create historical ELO progression chart for specified athletes."""
        if era_file not in self.era_files:
            return go.Figure()
        
        elo_history = self.calculate_basic_elo(era_file)
        
        if elo_history.empty:
            return go.Figure()
        
        # Filter for specified athletes
        athlete_data = elo_history[elo_history['name'].isin(athletes)]
        
        if athlete_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Use consistent colors
        colors = [
            os.getenv('CHART_COLOR_1', '#3498DB'),
            os.getenv('CHART_COLOR_2', '#E74C3C'),
            os.getenv('CHART_COLOR_3', '#27AE60'),
            os.getenv('CHART_COLOR_4', '#9B59B6'),
            os.getenv('CHART_COLOR_5', '#F39C12'),
            os.getenv('CHART_COLOR_6', '#1ABC9C'),
            os.getenv('CHART_COLOR_7', '#E67E22'),
            os.getenv('CHART_COLOR_8', '#95A5A6'),
            os.getenv('CHART_COLOR_9', '#34495E'),
            os.getenv('CHART_COLOR_10', '#8E44AD')
        ]
        
        for i, athlete in enumerate(athletes):
            athlete_elo = athlete_data[athlete_data['name'] == athlete].copy()
            if not athlete_elo.empty:
                # Sort by year and create cumulative index
                athlete_elo = athlete_elo.sort_values(['year', 'event'])
                athlete_elo['competition_number'] = range(1, len(athlete_elo) + 1)
                
                fig.add_trace(go.Scatter(
                    x=athlete_elo['competition_number'],
                    y=athlete_elo['elo_after'],
                    mode='lines+markers',
                    name=athlete,
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