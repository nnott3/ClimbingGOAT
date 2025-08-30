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


class IFSCStatsAnalyzer:
    """Analyzes IFSC climbing statistics and generates insights."""

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