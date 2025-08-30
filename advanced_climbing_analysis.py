import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import percentileofscore
import networkx as nx
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AlternativeRatingSystem:
    """Implements alternative rating systems beyond ELO."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        
    def glicko_rating(self, initial_rating=1500, initial_rd=350, c=15.8):
        """
        Simplified Glicko rating system implementation.
        More sophisticated than ELO as it considers rating reliability.
        """
        ratings = {}
        rd_values = {}  # Rating deviation
        
        # Group competitions chronologically
        competitions = self._get_chronological_competitions()
        
        for comp_info, comp_df in competitions:
            valid_results = comp_df.dropna(subset=['round_rank', 'name'])
            if len(valid_results) < 2:
                continue
                
            players = valid_results['name'].tolist()
            ranks = valid_results['round_rank'].tolist()
            
            # Initialize new players
            for player in players:
                if player not in ratings:
                    ratings[player] = initial_rating
                    rd_values[player] = initial_rd
            
            # Update RD values (increase with time)
            for player in players:
                rd_values[player] = min(initial_rd, 
                    np.sqrt(rd_values[player]**2 + c**2))
            
            # Calculate new ratings
            new_ratings = {}
            new_rds = {}
            
            for i, player in enumerate(players):
                current_rating = ratings[player]
                current_rd = rd_values[player]
                
                # Calculate performance against each opponent
                opponents = players[:i] + players[i+1:]
                
                d_squared_sum = 0
                score_sum = 0
                
                for j, opponent in enumerate(opponents):
                    opp_rating = ratings[opponent]
                    opp_rd = rd_values[opponent]
                    
                    # Expected score
                    expected = 1 / (1 + 10**((opp_rating - current_rating)/400))
                    
                    # Actual score (based on relative ranking)
                    actual = 1 if ranks[i] < ranks[players.index(opponent)] else 0
                    
                    # Glicko calculations
                    g_rd = 1 / np.sqrt(1 + 3 * (opp_rd/400)**2 / np.pi**2)
                    d_squared_sum += g_rd**2 * expected * (1 - expected)
                    score_sum += g_rd * (actual - expected)
                
                if d_squared_sum > 0:
                    d_squared = 1 / d_squared_sum
                    
                    # New rating
                    new_ratings[player] = current_rating + (
                        (15.8**2 / (current_rd**2 + d_squared)) * score_sum
                    )
                    
                    # New RD
                    new_rds[player] = np.sqrt(1 / (1/current_rd**2 + 1/d_squared))
                else:
                    new_ratings[player] = current_rating
                    new_rds[player] = current_rd
            
            # Update ratings
            ratings.update(new_ratings)
            rd_values.update(new_rds)
        
        return ratings, rd_values
    
    def trueskill_inspired_rating(self):
        """
        TrueSkill-inspired rating that considers uncertainty.
        Simplified version focusing on multi-player competitions.
        """
        mu = defaultdict(lambda: 25)  # Skill mean
        sigma = defaultdict(lambda: 8.33)  # Skill variance
        
        competitions = self._get_chronological_competitions()
        
        for comp_info, comp_df in competitions:
            valid_results = comp_df.dropna(subset=['round_rank', 'name'])
            if len(valid_results) < 2:
                continue
                
            # Sort by rank
            valid_results = valid_results.sort_values('round_rank')
            players = valid_results['name'].tolist()
            
            # Simple update: winners gain rating, losers lose rating
            # Weight by uncertainty (higher sigma = more change)
            for i, player in enumerate(players):
                performance = 1 - (i / (len(players) - 1)) if len(players) > 1 else 0.5
                
                # Update based on performance and uncertainty
                learning_rate = sigma[player] / 10
                mu[player] += learning_rate * (performance - 0.5) * 2
                
                # Decrease uncertainty with more games
                sigma[player] = max(1, sigma[player] * 0.99)
        
        # Conservative rating: mu - 3*sigma (99.7% confidence)
        conservative_ratings = {
            player: mu[player] - 3 * sigma[player] 
            for player in mu.keys()
        }
        
        return dict(mu), dict(sigma), conservative_ratings
    
    def percentile_based_rating(self):
        """
        Rating based on percentile performance across all competitions.
        Simple but effective for identifying consistent performers.
        """
        athlete_performances = defaultdict(list)
        
        # Collect all rank percentiles for each athlete
        for _, comp_df in self._get_chronological_competitions():
            valid_results = comp_df.dropna(subset=['round_rank', 'name'])
            if len(valid_results) < 2:
                continue
                
            for _, row in valid_results.iterrows():
                # Calculate percentile (lower rank = higher percentile)
                total_competitors = len(valid_results)
                rank = row['round_rank']
                percentile = (total_competitors - rank + 1) / total_competitors * 100
                
                athlete_performances[row['name']].append(percentile)
        
        # Calculate various statistics
        ratings = {}
        for athlete, performances in athlete_performances.items():
            ratings[athlete] = {
                'mean_percentile': np.mean(performances),
                'median_percentile': np.median(performances),
                'weighted_percentile': np.mean(performances[-10:]) if len(performances) >= 10 else np.mean(performances),  # Recent form
                'consistency': 100 - np.std(performances),  # Lower std = more consistent
                'competitions': len(performances),
                'peak_percentile': np.max(performances)
            }
        
        return ratings
    
    def _get_chronological_competitions(self):
        """Get competitions in chronological order."""
        if 'comp_date' in self.results_df.columns:
            # Sort by date
            groups = self.results_df.groupby(['comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round'])
            sorted_groups = sorted(groups, key=lambda x: x[0][0] if pd.notna(x[0][0]) else '9999')
        else:
            # Sort by filename as proxy for chronological order
            groups = self.results_df.groupby('source_file')
            sorted_groups = sorted(groups, key=lambda x: x[0])
        
        return sorted_groups



class AdvancedClimbingAnalytics:
    """Advanced analytics for climbing performance."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        
    def peak_performance_analysis(self):
        """Analyze when athletes reach their peak performance."""
        athlete_careers = {}
        
        # Group by athlete and analyze their competition timeline
        for athlete, athlete_df in self.results_df.groupby('name'):
            if len(athlete_df) < 5:  # Need at least 5 competitions
                continue
                
            # Sort by competition date/file
            if 'comp_date' in athlete_df.columns:
                athlete_df = athlete_df.sort_values('comp_date')
            else:
                athlete_df = athlete_df.sort_values('source_file')
            
            ranks = athlete_df['round_rank'].dropna()
            if len(ranks) < 5:
                continue
            
            # Find peak performance period (best average rank over 5 competitions)
            best_avg_rank = float('inf')
            peak_period = 0
            
            for i in range(len(ranks) - 4):
                avg_rank = ranks.iloc[i:i+5].mean()
                if avg_rank < best_avg_rank:
                    best_avg_rank = avg_rank
                    peak_period = i + 2  # Middle of the 5-competition window
            
            athlete_careers[athlete] = {
                'total_competitions': len(athlete_df),
                'career_best_rank': ranks.min(),
                'career_avg_rank': ranks.mean(),
                'peak_period_competition': peak_period,
                'peak_avg_rank': best_avg_rank,
                'country': athlete_df['country'].iloc[0] if 'country' in athlete_df else '',
                'early_career_avg': ranks.head(5).mean(),
                'late_career_avg': ranks.tail(5).mean(),
                'improvement': ranks.head(5).mean() - ranks.tail(5).mean()  # Positive = improved
            }
        
        return pd.DataFrame(athlete_careers).T
    
    def discipline_crossover_analysis(self):
        """Analyze athletes who compete in multiple disciplines."""
        if 'comp_discipline' not in self.results_df.columns:
            print("Discipline information not available")
            return pd.DataFrame()
        
        # Find athletes who compete in multiple disciplines
        multi_discipline_athletes = []
        
        for athlete, athlete_df in self.results_df.groupby('name'):
            disciplines = athlete_df['comp_discipline'].unique()
            if len(disciplines) > 1:
                discipline_performance = {}
                
                for discipline in disciplines:
                    discipline_results = athlete_df[athlete_df['comp_discipline'] == discipline]
                    ranks = discipline_results['round_rank'].dropna()
                    
                    if len(ranks) >= 3:  # At least 3 competitions in this discipline
                        discipline_performance[discipline] = {
                            'competitions': len(ranks),
                            'avg_rank': ranks.mean(),
                            'best_rank': ranks.min(),
                            'median_rank': ranks.median()
                        }
                
                if len(discipline_performance) > 1:
                    multi_discipline_athletes.append({
                        'name': athlete,
                        'country': athlete_df['country'].iloc[0] if 'country' in athlete_df else '',
                        'disciplines': list(discipline_performance.keys()),
                        'num_disciplines': len(discipline_performance),
                        'performance': discipline_performance
                    })
        
        return pd.DataFrame(multi_discipline_athletes)
    
    def dominance_periods(self, min_competitions: int = 10):
        """Identify periods of dominance by specific athletes."""
        dominance_analysis = []
        
        # Analyze by year if possible
        if 'file_year' in self.results_df.columns:
            time_groups = self.results_df.groupby('file_year')
        else:
            # Group by competition sets
            competitions = sorted(self.results_df['source_file'].unique())
            time_groups = []
            for i in range(0, len(competitions), 20):  # Group every 20 competitions
                period_comps = competitions[i:i+20]
                period_df = self.results_df[self.results_df['source_file'].isin(period_comps)]
                time_groups.append((f"Period_{i//20 + 1}", period_df))
        
        for period, period_df in time_groups:
            # Calculate dominance metrics for this period
            athlete_stats = period_df.groupby('name').agg({
                'round_rank': ['count', 'mean', 'median', 'std'],
                'source_file': 'nunique'
            }).round(2)
            
            athlete_stats.columns = ['competitions', 'avg_rank', 'median_rank', 'rank_std', 'unique_events']
            
            # Filter athletes with enough competitions
            qualified_athletes = athlete_stats[athlete_stats['competitions'] >= min_competitions]
            
            if not qualified_athletes.empty:
                # Calculate dominance score (lower avg rank + consistency)
                qualified_athletes['dominance_score'] = (
                    1 / qualified_athletes['avg_rank'] * 100 +  # Performance component
                    1 / (qualified_athletes['rank_std'] + 1) * 50  # Consistency component
                )
                
                top_athlete = qualified_athletes['dominance_score'].idxmax()
                
                dominance_analysis.append({
                    'period': period,
                    'dominant_athlete': top_athlete,
                    'avg_rank': qualified_athletes.loc[top_athlete, 'avg_rank'],
                    'competitions': qualified_athletes.loc[top_athlete, 'competitions'],
                    'dominance_score': qualified_athletes.loc[top_athlete, 'dominance_score'],
                    'total_athletes': len(qualified_athletes)
                })
        
        return pd.DataFrame(dominance_analysis)
    
    def performance_clustering(self):
        """Cluster athletes based on performance characteristics."""
        # Create feature matrix for each athlete
        athlete_features = []
        athlete_names = []
        
        for athlete, athlete_df in self.results_df.groupby('name'):
            if len(athlete_df) < 10:  # Need sufficient data
                continue
                
            ranks = athlete_df['round_rank'].dropna()
            if len(ranks) < 10:
                continue
            
            # Calculate performance features
            features = [
                ranks.mean(),  # Average performance
                ranks.median(),  # Median performance
                ranks.std(),  # Consistency (lower std = more consistent)
                ranks.min(),  # Best performance
                ranks.quantile(0.75),  # 75th percentile performance
                len(ranks),  # Total competitions
                np.sum(ranks <= 3),  # Number of podium finishes
                np.sum(ranks <= 10),  # Number of top 10 finishes
            ]
            
            athlete_features.append(features)
            athlete_names.append(athlete)
        
        if len(athlete_features) < 10:
            print("Not enough data for clustering analysis")
            return None
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(athlete_features)
        
        # Perform clustering
        n_clusters = min(8, len(athlete_features) // 10)  # Reasonable number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Create results DataFrame
        clustering_results = pd.DataFrame({
            'athlete': athlete_names,
            'cluster': cluster_labels,
            'avg_rank': [f[0] for f in athlete_features],
            'consistency': [f[2] for f in athlete_features],
            'best_rank': [f[3] for f in athlete_features],
            'total_competitions': [f[5] for f in athlete_features],
            'podium_finishes': [f[6] for f in athlete_features]
        })
        
        # Analyze cluster characteristics
        cluster_summary = clustering_results.groupby('cluster').agg({
            'avg_rank': 'mean',
            'consistency': 'mean',
            'best_rank': 'mean',
            'total_competitions': 'mean',
            'podium_finishes': 'mean',
            'athlete': 'count'
        }).round(2)
        
        cluster_summary.columns = ['avg_avg_rank', 'avg_consistency', 'avg_best_rank', 
                                 'avg_competitions', 'avg_podiums', 'cluster_size']
        
        return clustering_results, cluster_summary, features_scaled, scaler

def comprehensive_goat_analysis(results_df: pd.DataFrame):
    """
    Comprehensive GOAT (Greatest of All Time) analysis using multiple rating systems.
    """
    print("🏔️ COMPREHENSIVE GOAT ANALYSIS 🏔️")
    print("=" * 60)
    
    # Initialize rating systems
    alt_rating = AlternativeRatingSystem(results_df)
    advanced_analytics = AdvancedClimbingAnalytics(results_df)
    
    # 1. ELO Ratings (from previous implementation)
    print("1. Calculating ELO ratings...")
    from ifsc_analysis_pipeline import IFSCEloAnalyzer
    elo_analyzer = IFSCEloAnalyzer(results_df)
    elo_analyzer.calculate_elo_ratings()
    
    # 2. Alternative Rating Systems
    print("2. Calculating Glicko ratings...")
    glicko_ratings, glicko_rds = alt_rating.glicko_rating()
    
    print("3. Calculating TrueSkill-inspired ratings...")
    trueskill_mu, trueskill_sigma, trueskill_conservative = alt_rating.trueskill_inspired_rating()
    
    print("4. Calculating percentile-based ratings...")
    percentile_ratings = alt_rating.percentile_based_rating()
    

    
    # 6. Peak Performance Analysis
    print("6. Analyzing peak performances...")
    peak_analysis = advanced_analytics.peak_performance_analysis()
    
    # 7. Combine all ratings into a unified GOAT score
    print("7. Computing unified GOAT scores...")
    
    # Get all athletes
    all_athletes = set()
    for system in [glicko_ratings, trueskill_mu, percentile_ratings]:
        if isinstance(system, dict):
            all_athletes.update(system.keys())
        else:
            all_athletes.update(system.index)
    
    # Combine ratings
    goat_scores = []
    
    for athlete in all_athletes:
        scores = {}
        
        # ELO score (normalized)
        elo_score = 0
        for category, elo_system in elo_analyzer.elo_systems.items():
            if athlete in elo_system.ratings:
                elo_score = max(elo_score, elo_system.ratings[athlete])
        scores['elo'] = (elo_score - 1500) / 200 if elo_score > 0 else 0
        
        # Glicko score (normalized)
        glicko_score = glicko_ratings.get(athlete, 1500)
        scores['glicko'] = (glicko_score - 1500) / 200
        
        # TrueSkill conservative score (normalized)
        trueskill_score = trueskill_conservative.get(athlete, 25)
        scores['trueskill'] = (trueskill_score - 25) / 10
        
        # Percentile score
        if athlete in percentile_ratings:
            percentile_data = percentile_ratings[athlete]
            scores['percentile'] = percentile_data['weighted_percentile'] / 10
            scores['consistency'] = percentile_data['consistency'] / 10
        else:
            scores['percentile'] = 5
            scores['consistency'] = 5
    
        # Peak performance bonus
        if athlete in peak_analysis.index:
            peak_data = peak_analysis.loc[athlete]
            # Bonus for low peak average rank
            scores['peak_performance'] = max(0, (20 - peak_data['peak_avg_rank']) / 2)
        else:
            scores['peak_performance'] = 0
        
        # Calculate weighted GOAT score
        weights = {
            'elo': 0.25,
            'glicko': 0.25,
            'trueskill': 0.20,
            'percentile': 0.15,
            'consistency': 0.05,
            'peak_performance': 0.05
        }
        
        goat_score = sum(scores[metric] * weights[metric] for metric in weights.keys())
        
        goat_scores.append({
            'athlete': athlete,
            'goat_score': goat_score,
            **scores
        })
    
    # Create final GOAT DataFrame
    goat_df = pd.DataFrame(goat_scores)
    goat_df = goat_df.sort_values('goat_score', ascending=False)
    
    # Add athlete metadata
    athlete_metadata = results_df.groupby('name').agg({
        'country': 'first',
        'round_rank': ['count', 'mean', 'min'],
        'source_file': 'nunique'
    }).round(2)
    
    athlete_metadata.columns = ['country', 'total_results', 'avg_rank', 'best_rank', 'competitions']
    
    goat_final = goat_df.merge(athlete_metadata, left_on='athlete', right_index=True, how='left')
    
    # Display top GOATs
    print("\n🏆 TOP 20 GREATEST OF ALL TIME 🏆")
    print("=" * 80)
    print(f"{'Rank':<4} {'Name':<25} {'Country':<5} {'GOAT Score':<10} {'Comps':<6} {'Avg Rank':<8} {'Best':<5}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(goat_final.head(20).iterrows()):
        print(f"{i+1:<4} {str(row['athlete'])[:24]:<25} {str(row['country'])[:4]:<5} "
              f"{row['goat_score']:<10.2f} {row['competitions']:<6.0f} "
              f"{row['avg_rank']:<8.1f} {row['best_rank']:<5.0f}")
    
    # Save results
    goat_final.to_csv('goat_analysis.csv', index=False)
    print(f"\nDetailed results saved to 'goat_analysis.csv'")
    
    return goat_final

def create_goat_visualizations(goat_df: pd.DataFrame, results_df: pd.DataFrame):
    """Create comprehensive visualizations for GOAT analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Top 15 GOAT scores
    top_15 = goat_df.head(15)
    axes[0, 0].barh(range(len(top_15)), top_15['goat_score'])
    axes[0, 0].set_yticks(range(len(top_15)))
    axes[0, 0].set_yticklabels([name[:20] for name in top_15['athlete']])
    axes[0, 0].set_xlabel('GOAT Score')
    axes[0, 0].set_title('Top 15 Greatest of All Time')
    axes[0, 0].invert_yaxis()
    
    # 2. Score components for top 10
    top_10 = goat_df.head(10)
    components = ['elo', 'glicko', 'trueskill', 'percentile', 'consistency']
    
    for i, component in enumerate(components):
        if i == 0:
            bottom = np.zeros(len(top_10))
        else:
            bottom = np.sum([top_10[comp] for comp in components[:i]], axis=0)
        
        axes[0, 1].bar(range(len(top_10)), top_10[component], bottom=bottom, 
                      label=component, alpha=0.8)
    
    axes[0, 1].set_xticks(range(len(top_10)))
    axes[0, 1].set_xticklabels([name[:10] for name in top_10['athlete']], rotation=45)
    axes[0, 1].set_ylabel('Score Component')
    axes[0, 1].set_title('GOAT Score Components (Top 10)')
    axes[0, 1].legend()
    
    # 3. GOAT Score vs Average Rank
    axes[0, 2].scatter(goat_df['avg_rank'], goat_df['goat_score'], alpha=0.6)
    axes[0, 2].set_xlabel('Average Rank')
    axes[0, 2].set_ylabel('GOAT Score')
    axes[0, 2].set_title('GOAT Score vs Average Performance')
    axes[0, 2].invert_xaxis()  # Lower rank (better performance) on the right
    
    # 4. Country representation in top 50
    top_50_countries = goat_df.head(50)['country'].value_counts()
    axes[1, 0].pie(top_50_countries.values, labels=top_50_countries.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Country Representation in Top 50 GOATs')
    
    # 5. Competitions vs GOAT Score
    axes[1, 1].scatter(goat_df['competitions'], goat_df['goat_score'], alpha=0.6)
    axes[1, 1].set_xlabel('Number of Competitions')
    axes[1, 1].set_ylabel('GOAT Score')
    axes[1, 1].set_title('Competition Experience vs GOAT Score')
    
    # 6. Distribution of GOAT scores
    axes[1, 2].hist(goat_df['goat_score'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('GOAT Score')
    axes[1, 2].set_ylabel('Number of Athletes')
    axes[1, 2].set_title('Distribution of GOAT Scores')
    
    plt.tight_layout()
    plt.savefig('goat_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run comprehensive analysis."""
    print("Loading data...")
    
    # Load your results data
    from ifsc_analysis_pipeline import IFSCDataAggregator
    aggregator = IFSCDataAggregator()
    results_df = aggregator.load_all_results()
    
    # Run comprehensive GOAT analysis
    goat_df = comprehensive_goat_analysis(results_df)
    
    # Create visualizations
    create_goat_visualizations(goat_df, results_df)
    
    # Additional analyses
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSES")
    print("=" * 60)
    
    advanced_analytics = AdvancedClimbingAnalytics(results_df)
    
    # Peak performance analysis
    peak_analysis = advanced_analytics.peak_performance_analysis()
    if not peak_analysis.empty:
        print("\nAthletes who improved most over their careers:")
        improvers = peak_analysis.sort_values('improvement', ascending=False).head(10)
        for athlete, data in improvers.iterrows():
            print(f"{athlete}: {data['improvement']:.1f} rank improvement")
    
    # Dominance periods
    dominance = advanced_analytics.dominance_periods()
    if not dominance.empty:
        print("\nPeriods of dominance:")
        for _, period in dominance.iterrows():
            print(f"{period['period']}: {period['dominant_athlete']} "
                  f"(avg rank {period['avg_rank']:.1f})")
    
    print("\nAnalysis complete! Check the generated CSV files and visualizations.")

if __name__ == "__main__":
    main()