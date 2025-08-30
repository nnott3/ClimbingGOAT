import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class IFSCDataAggregator:
    """Aggregates and processes IFSC climbing competition data."""
    
    def __init__(self, data_dir: str = "./Results_Expanded_API", output_dir: str = "./ifsc_analysis_output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.results_df = None
        self.metadata_df = None
        
        # Create output directories
        self._create_output_directories()
        
    def _create_output_directories(self):
        """Create necessary output directories."""
        # Main output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        (self.output_dir / "visuals").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        print(f"Created output directories:")
        print(f"  Main: {self.output_dir}")
        print(f"  Visuals: {self.output_dir / 'visuals'}")
        print(f"  Data: {self.output_dir / 'data'}")
        print(f"  Reports: {self.output_dir / 'reports'}")
        
    def load_all_results(self) -> pd.DataFrame:
        """Load and combine all result CSV files."""
        print("Loading all result files...")
        
        all_results = []
        total_files = 0
        
        # Walk through all year directories
        for year_dir in sorted(self.data_dir.glob("*")):
            if not year_dir.is_dir():
                continue
                
            print(f"Processing {year_dir.name}...")
            year_files = 0
            
            for csv_file in year_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        # Add file metadata
                        df['source_file'] = csv_file.name
                        df['file_year'] = year_dir.name
                        df['file_path'] = str(csv_file)
                        all_results.append(df)
                        year_files += 1
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
            
            total_files += year_files
            print(f"  Loaded {year_files} files")
        
        if not all_results:
            raise ValueError("No result files found!")
        
        # Combine all results
        self.results_df = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal files loaded: {total_files}")
        print(f"Total records: {len(self.results_df)}")
        
        # Basic data cleaning
        self._clean_basic_data()
        
        # Save aggregated data
        self._save_aggregated_data()
        
        return self.results_df
    
    def load_metadata(self) -> pd.DataFrame:
        """Load event metadata files."""
        print("Loading event metadata...")
        
        metadata_files = list(Path("./Event_metadata_API").glob("*.csv"))
        if not metadata_files:
            print("No metadata files found, creating basic metadata from results")
            return self._create_basic_metadata()
        
        all_metadata = []
        for file in metadata_files:
            try:
                df = pd.read_csv(file)
                all_metadata.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if all_metadata:
            self.metadata_df = pd.concat(all_metadata, ignore_index=True)
            print(f"Loaded metadata for {len(self.metadata_df)} events")
            
            # Save metadata
            self._save_metadata()
        
        return self.metadata_df
    
    def _clean_basic_data(self):
        """Basic data cleaning operations."""
        print("Cleaning data...")
        
        # Standardize column names
        self.results_df.columns = [col.lower().strip() for col in self.results_df.columns]
        
        # Clean athlete names
        if 'name' in self.results_df.columns:
            self.results_df['name'] = self.results_df['name'].str.strip()
            
        # Clean countries
        if 'country' in self.results_df.columns:
            self.results_df['country'] = self.results_df['country'].str.strip().str.upper()
            
        # Parse dates
        date_columns = ['start_date']
        for col in date_columns:
            if col in self.results_df.columns:
                self.results_df[col] = pd.to_datetime(self.results_df[col], errors='coerce')
        
        # Extract competition info from filename
        self._extract_competition_info()
        
        # Add processing timestamp
        self.results_df['processed_at'] = datetime.now()
        
        print("Data cleaning completed")
    
    def _extract_competition_info(self):
        """Extract competition information from source filenames."""
        if 'source_file' not in self.results_df.columns:
            return
            
        # Parse filename format: date_location_discipline_gender_round.csv
        file_parts = self.results_df['source_file'].str.replace('.csv', '').str.split('_')
        
        try:
            self.results_df['comp_date'] = file_parts.str[0]
            self.results_df['comp_location'] = file_parts.str[1]  
            self.results_df['comp_discipline'] = file_parts.str[2]
            self.results_df['comp_gender'] = file_parts.str[3]
            self.results_df['comp_round'] = file_parts.str[4]
        except:
            print("Could not parse all filename components")
    
    def _create_basic_metadata(self) -> pd.DataFrame:
        """Create basic metadata from results if metadata files don't exist."""
        if self.results_df is None:
            return pd.DataFrame()
            
        # Group by competition identifiers
        metadata = self.results_df.groupby([
            'comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round'
        ]).agg({
            'name': 'count',  # Number of athletes
            'file_year': 'first'
        }).rename(columns={'name': 'athlete_count'}).reset_index()
        
        self.metadata_df = metadata
        return metadata
    
    def _save_aggregated_data(self):
        """Save aggregated results to CSV."""
        if self.results_df is not None:
            output_file = self.output_dir / "data" / "aggregated_results.csv"
            self.results_df.to_csv(output_file, index=False)
            print(f"Saved aggregated results to: {output_file}")
            
            # Save summary statistics
            summary_file = self.output_dir / "data" / "data_summary.csv"
            summary_stats = {
                'metric': ['total_records', 'unique_athletes', 'unique_competitions', 
                          'unique_countries', 'date_range_start', 'date_range_end',
                          'disciplines', 'genders', 'processing_date'],
                'value': [
                    len(self.results_df),
                    self.results_df['name'].nunique() if 'name' in self.results_df.columns else 'N/A',
                    self.results_df['source_file'].nunique(),
                    self.results_df['country'].nunique() if 'country' in self.results_df.columns else 'N/A',
                    self.results_df['file_year'].min() if 'file_year' in self.results_df.columns else 'N/A',
                    self.results_df['file_year'].max() if 'file_year' in self.results_df.columns else 'N/A',
                    self.results_df['comp_discipline'].nunique() if 'comp_discipline' in self.results_df.columns else 'N/A',
                    self.results_df['comp_gender'].nunique() if 'comp_gender' in self.results_df.columns else 'N/A',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            pd.DataFrame(summary_stats).to_csv(summary_file, index=False)
            print(f"Saved data summary to: {summary_file}")
    
    def _save_metadata(self):
        """Save metadata to CSV."""
        if self.metadata_df is not None:
            output_file = self.output_dir / "data" / "event_metadata.csv"
            self.metadata_df.to_csv(output_file, index=False)
            print(f"Saved metadata to: {output_file}")

class IFSCStatsAnalyzer:
    """Analyzes IFSC climbing statistics and generates insights."""
    
    def __init__(self, results_df: pd.DataFrame, metadata_df: pd.DataFrame = None, output_dir: Path = None):
        self.results_df = results_df
        self.metadata_df = metadata_df
        self.output_dir = output_dir or Path("./ifsc_analysis_output")
        
    def competition_overview(self):
        """Generate overview statistics of competitions."""
        print("=== COMPETITION OVERVIEW ===")
        
        overview_stats = []
        
        if 'file_year' in self.results_df.columns:
            year_range = self.results_df['file_year'].astype(str)
            print(f"Data spans: {year_range.min()} - {year_range.max()}")
            overview_stats.append(("Data range", f"{year_range.min()} - {year_range.max()}"))
            
        total_comps = self.results_df['source_file'].nunique()
        total_athletes = self.results_df['name'].nunique()
        total_results = len(self.results_df)
        
        print(f"Total competitions: {total_comps}")
        print(f"Total athletes: {total_athletes}")
        print(f"Total results: {total_results}")
        
        overview_stats.extend([
            ("Total competitions", total_comps),
            ("Total athletes", total_athletes),
            ("Total results", total_results)
        ])
        
        # Discipline breakdown
        if 'comp_discipline' in self.results_df.columns:
            print("\nDiscipline distribution:")
            discipline_counts = self.results_df['comp_discipline'].value_counts()
            for discipline, count in discipline_counts.head(10).items():
                print(f"  {discipline}: {count}")
        
        # Gender breakdown
        if 'comp_gender' in self.results_df.columns:
            print("\nGender distribution:")
            gender_counts = self.results_df['comp_gender'].value_counts()
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count}")
        
        # Country participation
        if 'country' in self.results_df.columns:
            unique_countries = self.results_df['country'].nunique()
            print(f"\nCountries represented: {unique_countries}")
            print("Top countries by participation:")
            country_counts = self.results_df['country'].value_counts()
            for country, count in country_counts.head(10).items():
                print(f"  {country}: {count}")
            overview_stats.append(("Countries represented", unique_countries))
        
        # Save overview stats
        overview_df = pd.DataFrame(overview_stats, columns=['Metric', 'Value'])
        overview_file = self.output_dir / "reports" / "competition_overview.csv"
        overview_df.to_csv(overview_file, index=False)
        print(f"\nSaved overview statistics to: {overview_file}")
    
    def athlete_statistics(self):
        """Analyze athlete participation and performance."""
        print("\n=== ATHLETE STATISTICS ===")
        
        # Most active athletes
        athlete_comps = self.results_df.groupby('name').agg({
            'source_file': 'nunique',
            'country': 'first',
            'round_rank': ['count', 'mean', 'median', 'min']
        }).round(2)
        
        athlete_comps.columns = ['competitions', 'country', 'total_results', 'avg_rank', 'median_rank', 'best_rank']
        athlete_comps = athlete_comps.sort_values('competitions', ascending=False)
        
        print("Most active athletes (by competitions):")
        print(athlete_comps.head(10)[['competitions', 'country', 'avg_rank']])
        
        # Save athlete statistics
        athlete_stats_file = self.output_dir / "data" / "athlete_statistics.csv"
        athlete_comps.to_csv(athlete_stats_file)
        print(f"Saved athlete statistics to: {athlete_stats_file}")
        
        # Best performers by average rank
        experienced_athletes = athlete_comps[athlete_comps['competitions'] >= 10]
        if not experienced_athletes.empty:
            best_performers = experienced_athletes.sort_values('avg_rank')
            print("\nBest performers (avg rank, min 10 competitions):")
            print(best_performers.head(10)[['competitions', 'country', 'avg_rank']])
            
            # Save best performers
            best_performers_file = self.output_dir / "data" / "best_performers.csv"
            best_performers.to_csv(best_performers_file)
            print(f"Saved best performers to: {best_performers_file}")
        
        return athlete_comps
    
    def create_visualizations(self):
        """Create various visualizations of the data."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Competitions per year
        if 'file_year' in self.results_df.columns:
            yearly_comps = self.results_df.groupby('file_year')['source_file'].nunique()
            axes[0, 0].plot(yearly_comps.index.astype(int), yearly_comps.values, marker='o')
            axes[0, 0].set_title('Competitions per Year')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Number of Competitions')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Discipline distribution
        if 'comp_discipline' in self.results_df.columns:
            discipline_counts = self.results_df['comp_discipline'].value_counts()
            axes[0, 1].pie(discipline_counts.values, labels=discipline_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Distribution by Discipline')
        
        # 3. Top countries by athlete count
        if 'country' in self.results_df.columns:
            top_countries = self.results_df['country'].value_counts().head(15)
            axes[1, 0].barh(range(len(top_countries)), top_countries.values)
            axes[1, 0].set_yticks(range(len(top_countries)))
            axes[1, 0].set_yticklabels(top_countries.index)
            axes[1, 0].set_title('Top Countries by Participation')
            axes[1, 0].set_xlabel('Number of Results')
        
        # 4. Rank distribution
        if 'round_rank' in self.results_df.columns:
            valid_ranks = self.results_df['round_rank'].dropna()
            if not valid_ranks.empty:
                axes[1, 1].hist(valid_ranks, bins=50, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Distribution of Ranks')
                axes[1, 1].set_xlabel('Rank')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / "visuals" / "ifsc_overview.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"Saved overview visualization to: {viz_file}")
        plt.show()

class EloRatingSystem:
    """Implements ELO rating system for climbing competitions."""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}
        self.rating_history = []
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_rating(self, player: str, opponent_ratings: List[float], 
                     actual_score: float) -> float:
        """Update player rating based on performance against opponents."""
        if player not in self.ratings:
            self.ratings[player] = self.initial_rating
        
        current_rating = self.ratings[player]
        
        # Calculate expected score against all opponents
        expected_total = sum(self.expected_score(current_rating, opp_rating) 
                           for opp_rating in opponent_ratings)
        
        # Update rating
        if len(opponent_ratings) > 0:
            expected_avg = expected_total / len(opponent_ratings)
            new_rating = current_rating + self.k_factor * (actual_score - expected_avg)
            self.ratings[player] = new_rating
        
        return self.ratings[player]
    
    def process_competition(self, competition_df: pd.DataFrame, 
                          competition_info: Dict):
        """Process a single competition and update all player ratings."""
        if 'round_rank' not in competition_df.columns or 'name' not in competition_df.columns:
            return
        
        # Filter valid ranks and sort by rank
        valid_results = competition_df.dropna(subset=['round_rank', 'name'])
        if len(valid_results) < 2:  # Need at least 2 competitors
            return
            
        valid_results = valid_results.sort_values('round_rank')
        players = valid_results['name'].tolist()
        ranks = valid_results['round_rank'].tolist()
        
        # Get current ratings for all players
        current_ratings = [self.ratings.get(player, self.initial_rating) 
                          for player in players]
        
        # Update ratings based on head-to-head comparisons
        for i, player in enumerate(players):
            # Calculate score based on rank (lower rank = higher score)
            player_score = 1 - (ranks[i] - 1) / (len(players) - 1) if len(players) > 1 else 0.5
            
            # Get opponent ratings (all other players in competition)
            opponent_ratings = current_ratings[:i] + current_ratings[i+1:]
            
            # Update rating
            new_rating = self.update_rating(player, opponent_ratings, player_score)
            
            # Record history
            self.rating_history.append({
                'player': player,
                'competition': competition_info.get('comp_id', 'unknown'),
                'date': competition_info.get('date', None),
                'discipline': competition_info.get('discipline', 'unknown'),
                'old_rating': current_ratings[i],
                'new_rating': new_rating,
                'rank': ranks[i],
                'competitors': len(players)
            })

class IFSCEloAnalyzer:
    """Analyzes IFSC data using ELO rating system."""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Path = None):
        self.results_df = results_df
        self.elo_systems = {}  # Separate ELO for each discipline/gender combo
        self.output_dir = output_dir or Path("./ifsc_analysis_output")
        
    def calculate_elo_ratings(self):
        """Calculate ELO ratings for all athletes across all competitions."""
        print("Calculating ELO ratings...")
        
        # Group by discipline and gender for separate rating systems
        if 'comp_discipline' in self.results_df.columns and 'comp_gender' in self.results_df.columns:
            categories = self.results_df.groupby(['comp_discipline', 'comp_gender'])
        else:
            categories = [('overall', self.results_df)]
        
        for (discipline, gender), group_df in categories:
            if isinstance(categories, pd.core.groupby.DataFrameGroupBy):
                category_key = f"{discipline}_{gender}"
            else:
                category_key = discipline
                group_df = gender  # In case of single category
            
            print(f"Processing {category_key}...")
            
            self.elo_systems[category_key] = EloRatingSystem()
            
            # Sort by date to process competitions chronologically
            if 'comp_date' in group_df.columns:
                competitions = group_df.groupby(['comp_date', 'comp_location', 'comp_round'])
            else:
                competitions = group_df.groupby(['source_file'])
            
            for comp_id, comp_df in competitions:
                comp_info = {
                    'comp_id': str(comp_id),
                    'discipline': discipline if isinstance(comp_id, tuple) else 'unknown',
                    'date': comp_id[0] if isinstance(comp_id, tuple) and len(comp_id) > 0 else None
                }
                
                self.elo_systems[category_key].process_competition(comp_df, comp_info)
        
        # Save ELO history for all categories
        self._save_elo_history()
        
        print("ELO calculation completed!")
    
    def _save_elo_history(self):
        """Save ELO rating history to CSV files."""
        for category, elo_system in self.elo_systems.items():
            if elo_system.rating_history:
                history_df = pd.DataFrame(elo_system.rating_history)
                history_file = self.output_dir / "data" / f"elo_history_{category}.csv"
                history_df.to_csv(history_file, index=False)
                print(f"Saved ELO history for {category} to: {history_file}")
    
    def get_top_athletes(self, category: str = None, top_n: int = 50) -> pd.DataFrame:
        """Get top athletes by ELO rating."""
        if category and category in self.elo_systems:
            ratings = self.elo_systems[category].ratings
            category_name = category
        else:
            # Combine all ratings (taking max rating across categories)
            all_ratings = {}
            for cat, elo_system in self.elo_systems.items():
                for player, rating in elo_system.ratings.items():
                    if player not in all_ratings or rating > all_ratings[player]:
                        all_ratings[player] = rating
            ratings = all_ratings
            category_name = "overall"
        
        if not ratings:
            return pd.DataFrame()
        
        # Create DataFrame
        top_athletes = pd.DataFrame([
            {'name': player, 'elo_rating': rating}
            for player, rating in sorted(ratings.items(), 
                                       key=lambda x: x[1], reverse=True)[:top_n]
        ])
        
        # Add additional stats from original data
        athlete_stats = self.results_df.groupby('name').agg({
            'round_rank': ['count', 'mean', 'median', 'min'],
            'country': 'first',
            'source_file': 'nunique'
        }).round(2)
        
        athlete_stats.columns = ['total_comps', 'avg_rank', 'median_rank', 'best_rank', 'country', 'unique_events']
        
        # Merge with ELO ratings
        result = top_athletes.merge(athlete_stats, left_on='name', right_index=True, how='left')
        
        # Save results
        output_file = self.output_dir / "data" / f"top_athletes_elo_{category_name}.csv"
        result.to_csv(output_file, index=False)
        print(f"Saved top athletes ({category_name}) to: {output_file}")
        
        print(f"\nTop {top_n} athletes by ELO rating ({category_name}):")
        print("=" * 80)
        print(f"{'Rank':<4} {'Name':<25} {'Country':<5} {'ELO':<6} {'Comps':<6} {'Avg Rank':<8} {'Best':<5}")
        print("-" * 80)
        
        for i, row in result.head(top_n).iterrows():
            print(f"{i+1:<4} {row['name'][:24]:<25} {str(row['country'])[:4]:<5} "
                  f"{row['elo_rating']:<6.0f} {row['total_comps']:<6.0f} "
                  f"{row['avg_rank']:<8.1f} {row['best_rank']:<5.0f}")
        
        return result
    
    def plot_rating_evolution(self, athletes: List[str], category: str = None):
        """Plot rating evolution over time for specific athletes."""
        if category and category in self.elo_systems:
            history = self.elo_systems[category].rating_history
            plot_title = f'ELO Rating Evolution - {category}'
            filename = f'elo_evolution_{category}.png'
        else:
            # Combine all histories
            history = []
            for elo_system in self.elo_systems.values():
                history.extend(elo_system.rating_history)
            plot_title = 'ELO Rating Evolution - Overall'
            filename = 'elo_evolution_overall.png'
        
        # Filter for specified athletes
        athlete_history = [h for h in history if h['player'] in athletes]
        
        if not athlete_history:
            print("No rating history found for specified athletes")
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for athlete in athletes:
            athlete_data = [h for h in athlete_history if h['player'] == athlete]
            if athlete_data:
                # Sort by date if available
                athlete_data.sort(key=lambda x: x.get('date', ''))
                
                ratings = [h['new_rating'] for h in athlete_data]
                competitions = range(len(ratings))
                
                plt.plot(competitions, ratings, marker='o', label=athlete, linewidth=2)
        
        plt.title(plot_title)
        plt.xlabel('Competition Number')
        plt.ylabel('ELO Rating')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "visuals" / filename
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved ELO evolution plot to: {plot_file}")
        plt.show()

def main():
    """Main analysis pipeline."""
    print("IFSC Climbing Data Analysis Pipeline")
    print("=" * 50)
    
    # 1. Load and aggregate data
    aggregator = IFSCDataAggregator()
    results_df = aggregator.load_all_results()
    metadata_df = aggregator.load_metadata()
    
    # 2. Basic statistics and overview
    stats_analyzer = IFSCStatsAnalyzer(results_df, metadata_df, aggregator.output_dir)
    stats_analyzer.competition_overview()
    athlete_stats = stats_analyzer.athlete_statistics()
    stats_analyzer.create_visualizations()
    
    # 3. ELO analysis
    elo_analyzer = IFSCEloAnalyzer(results_df, aggregator.output_dir)
    elo_analyzer.calculate_elo_ratings()
    
    # 4. Show top athletes overall
    top_athletes = elo_analyzer.get_top_athletes(top_n=20)
    
    # 5. Show top athletes by category (if data allows)
    print("\n" + "=" * 50)
    print("TOP ATHLETES BY CATEGORY")
    print("=" * 50)
    
    category_results = {}
    for category in elo_analyzer.elo_systems.keys():
        if len(elo_analyzer.elo_systems[category].ratings) > 0:
            print(f"\n{category.upper()}:")
            category_top = elo_analyzer.get_top_athletes(category, top_n=10)
            category_results[category] = category_top
    
    # 6. Plot evolution for top athletes
    if not top_athletes.empty:
        top_names = top_athletes.head(5)['name'].tolist()
        elo_analyzer.plot_rating_evolution(top_names)
        
        # Plot category-specific evolutions
        for category, cat_results in category_results.items():
            if not cat_results.empty:
                cat_top_names = cat_results.head(3)['name'].tolist()
                elo_analyzer.plot_rating_evolution(cat_top_names, category)
    
    # 7. Create final summary report
    summary_file = aggregator.output_dir / "reports" / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("IFSC Climbing Data Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total records processed: {len(results_df)}\n")
        f.write(f"Unique athletes: {results_df['name'].nunique()}\n")
        f.write(f"Unique competitions: {results_df['source_file'].nunique()}\n\n")
        f.write("Files generated:\n")
        f.write("- data/aggregated_results.csv\n")
        f.write("- data/data_summary.csv\n")
        f.write("- data/athlete_statistics.csv\n")
        f.write("- data/top_athletes_elo_*.csv\n")
        f.write("- data/elo_history_*.csv\n")
        f.write("- visuals/ifsc_overview.png\n")
        f.write("- visuals/elo_evolution_*.png\n")
        f.write("- reports/competition_overview.csv\n")
        f.write("- reports/analysis_summary.txt\n")
    
    print(f"\nAnalysis summary saved to: {summary_file}")
    print(f"\nAll outputs saved to: {aggregator.output_dir}")
    print("\nDirectory structure:")
    print(f"  {aggregator.output_dir}/")
    print(f"    ├── data/           (CSV files with processed data)")
    print(f"    ├── visuals/        (PNG charts and graphs)")
    print(f"    └── reports/        (Summary reports)")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()