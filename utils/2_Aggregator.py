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
    """Aggregates and processes IFSC climbing competition data with era-based grouping."""
    
    def __init__(self, data_dir: str = "./Results_Expanded_API", output_dir: str = "./ifsc_analysis_output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.results_df = None
        self.metadata_df = None
        
        # Define scoring system eras with proper naming
        self.scoring_systems = {
            'Lead': {
                'IFSC_2007+': {'start': 2007, 'end': 2025, 'description': 'IFSC system with qualis, semis, finals'},
                'UIAA_pre2007': {'start': 1900, 'end': 2006, 'description': 'UIAA system, ranking only'}
            },
            'Boulder': {
                'IFSC_AddedPoints_2025+': {'start': 2025, 'end': 2025, 'description': 'Combined point system (25 top, 10 zone, -0.1 per attempt)'},
                'IFSC_ZoneTopAttempts_2007-2024': {'start': 2007, 'end': 2024, 'description': 'Separate top/zone counting system'},
                'UIAA': {'start': 1900, 'end': 2006, 'description': 'UIAA system, ranking only'}
            },
            'Speed': {
                'IFSC_Time_2009+': {'start': 2009, 'end': 2025, 'description': 'Timed system with quali (2 laps) and finals (tournament)'},
                'IFSC_Score_2007-2008': {'start': 2007, 'end': 2008, 'description': 'IFSC scoring system, no times'},
                'UIAA': {'start': 1900, 'end': 2006, 'description': 'UIAA system, ranking only'}
            }
        }
        
        # Create output directories
        self._create_output_directories()
        
    def _create_output_directories(self):
        """Create necessary output directories."""
        # Main output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        (self.output_dir / "visuals").mkdir(exist_ok=True)
        (self.output_dir / "raw_data").mkdir(exist_ok=True)
        (self.output_dir / "data_summary").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        print(f"Created output directories:")
        print(f"  Main: {self.output_dir}")
        print(f"  Visuals: {self.output_dir / 'visuals'}")
        print(f"  Raw Data: {self.output_dir / 'raw_data'}")
        print(f"  Data Summary: {self.output_dir / 'data_summary'}")
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
        
        # Add scoring system classification
        self._classify_scoring_systems()
        
        # Save aggregated data (main file and era-based files)
        self._save_all_data()
        
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
        
        # Convert year to numeric for classification
        if 'year' in self.results_df.columns:
            self.results_df['year'] = pd.to_numeric(self.results_df['year'], errors='coerce')
        elif 'file_year' in self.results_df.columns:
            self.results_df['year'] = pd.to_numeric(self.results_df['file_year'], errors='coerce')
        
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
    
    def _classify_scoring_systems(self):
        """Classify each record by its scoring system era."""
        print("Classifying scoring systems by era...")
        
        def get_scoring_era(row):
            discipline = row.get('discipline', row.get('comp_discipline', ''))
            year = row.get('year', 0)
            
            if pd.isna(year) or year == 0:
                return 'Unknown'
            
            year = int(year)
            
            # Normalize discipline names
            discipline_map = {
                'Lead': 'Lead',
                'L': 'Lead',
                'Boulder': 'Boulder', 
                'B': 'Boulder',
                'Speed': 'Speed',
                'S': 'Speed',
                'Combined': 'Combined'  # Handle combined events separately
            }
            
            discipline = discipline_map.get(discipline, discipline)
            
            if discipline not in self.scoring_systems:
                return f'{discipline}_Unknown'
            
            # Find the appropriate era
            for era_name, era_info in self.scoring_systems[discipline].items():
                if era_info['start'] <= year <= era_info['end']:
                    return f'{discipline}_{era_name}'
            
            return f'{discipline}_Unknown'
        
        self.results_df['scoring_era'] = self.results_df.apply(get_scoring_era, axis=1)
        
        # Print era distribution
        era_counts = self.results_df['scoring_era'].value_counts()
        print("\nScoring Era Distribution:")
        for era, count in era_counts.items():
            print(f"  {era}: {count:,} records")
    
    def _save_all_data(self):
        """Save aggregated results - both main file and era-based splits."""
        if self.results_df is None:
            return
        
        # Save main aggregated file
        main_file = self.output_dir / "raw_data" / "Aggregated_results.csv"
        self.results_df.to_csv(main_file, index=False)
        print(f"Saved main aggregated results to: {main_file}")
        
        # Save era-based splits
        self._save_era_based_files()
        
        # Save summary statistics
        self._save_summary_statistics()
    
    def _save_era_based_files(self):
        """Save separate files for each scoring era and gender combination."""
        print("Saving era-based files...")
        
        raw_data_dir = self.output_dir / "raw_data"
        
        # Get unique combinations of era, gender, and discipline
        groupby_cols = ['scoring_era']
        if 'gender' in self.results_df.columns:
            groupby_cols.append('gender')
        elif 'comp_gender' in self.results_df.columns:
            groupby_cols.append('comp_gender')
        
        grouped = self.results_df.groupby(groupby_cols)
        
        file_summary = []
        
        # Define files to skip
        skip_patterns = ['Unknown', 'Combined_']
        
        for group_keys, group_df in grouped:
            if isinstance(group_keys, str):
                group_keys = (group_keys,)
            
            scoring_era = group_keys[0] if len(group_keys) > 0 else 'Unknown'
            gender = group_keys[1] if len(group_keys) > 1 else 'All'
            
            # Skip files matching skip patterns
            should_skip = any(pattern in scoring_era for pattern in skip_patterns)
            if should_skip:
                print(f"  Skipping {scoring_era}_{gender} (matches skip pattern)")
                continue
            
            # Create filename with proper format
            filename = f"{scoring_era}_{gender}.csv"
            filepath = raw_data_dir / filename
            
            # Save the group
            group_df.to_csv(filepath, index=False)
            
            file_summary.append({
                'filename': filename,
                'scoring_era': scoring_era,
                'gender': gender,
                'record_count': len(group_df),
                'unique_athletes': group_df['name'].nunique() if 'name' in group_df.columns else 0,
                'year_range': f"{group_df['year'].min():.0f}-{group_df['year'].max():.0f}" if 'year' in group_df.columns else 'Unknown'
            })
            
            print(f"  Saved {filename}: {len(group_df):,} records")
        
        # Save file summary in data_summary folder
        summary_df = pd.DataFrame(file_summary)
        summary_file = self.output_dir / "data_summary" / "era_files_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved era files summary to: {summary_file}")
    
    def _save_summary_statistics(self):
        """Save comprehensive summary statistics."""
        summary_dir = self.output_dir / "data_summary"
        
        # Basic statistics
        summary_file = summary_dir / "data_summary.csv"
        basic_stats = {
            'metric': [
                'total_records', 'unique_athletes', 'unique_competitions', 
                'unique_countries', 'date_range_start', 'date_range_end',
                'disciplines', 'genders', 'rounds', 'processing_date'
            ],
            'value': [
                len(self.results_df),
                self.results_df['name'].nunique() if 'name' in self.results_df.columns else 'N/A',
                self.results_df['source_file'].nunique(),
                self.results_df['country'].nunique() if 'country' in self.results_df.columns else 'N/A',
                self.results_df['year'].min() if 'year' in self.results_df.columns else 'N/A',
                self.results_df['year'].max() if 'year' in self.results_df.columns else 'N/A',
                self.results_df['discipline'].nunique() if 'discipline' in self.results_df.columns else 
                    self.results_df['comp_discipline'].nunique() if 'comp_discipline' in self.results_df.columns else 'N/A',
                self.results_df['gender'].nunique() if 'gender' in self.results_df.columns else
                    self.results_df['comp_gender'].nunique() if 'comp_gender' in self.results_df.columns else 'N/A',
                self.results_df['round'].nunique() if 'round' in self.results_df.columns else 'N/A',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        pd.DataFrame(basic_stats).to_csv(summary_file, index=False)
        print(f"Saved data summary to: {summary_file}")
        
        # Save detailed era statistics
        era_stats_file = summary_dir / "era_statistics.csv"
        if 'scoring_era' in self.results_df.columns:
            era_stats = self.results_df.groupby(['scoring_era']).agg({
                'name': 'nunique',
                'country': 'nunique',
                'year': ['min', 'max', 'nunique'],
                'source_file': 'nunique'
            }).round(2)
            
            # Flatten column names
            era_stats.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in era_stats.columns]
            era_stats = era_stats.reset_index()
            era_stats.to_csv(era_stats_file, index=False)
            print(f"Saved era statistics to: {era_stats_file}")
    
    def _create_basic_metadata(self) -> pd.DataFrame:
        """Create basic metadata from results if metadata files don't exist."""
        if self.results_df is None:
            return pd.DataFrame()
            
        # Group by competition identifiers
        metadata = self.results_df.groupby([
            'comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round'
        ]).agg({
            'name': 'count',  # Number of athletes
            'file_year': 'first',
            'scoring_era': 'first'
        }).rename(columns={'name': 'athlete_count'}).reset_index()
        
        self.metadata_df = metadata
        return metadata
    
    def _save_metadata(self):
        """Save metadata to CSV."""
        if self.metadata_df is not None:
            output_file = self.output_dir / "data_summary" / "event_metadata.csv"
            self.metadata_df.to_csv(output_file, index=False)
            print(f"Saved metadata to: {output_file}")
    
    def print_scoring_system_guide(self):
        """Print a guide to the scoring systems used."""
        print("\nScoring System Classification Guide:")
        print("=" * 50)
        
        for discipline, systems in self.scoring_systems.items():
            print(f"\n{discipline}:")
            for era_name, era_info in systems.items():
                print(f"  {era_name} ({era_info['start']}-{era_info['end']}): {era_info['description']}")
        
        if self.results_df is not None and 'scoring_era' in self.results_df.columns:
            print(f"\nCurrent Data Distribution:")
            era_counts = self.results_df['scoring_era'].value_counts()
            for era, count in era_counts.items():
                print(f"  {era}: {count:,} records")


def main():
    """Main analysis pipeline."""
    print("Enhanced IFSC Climbing Data Analysis Pipeline")
    print("=" * 60)
    
    # 1. Load and aggregate data
    aggregator = IFSCDataAggregator()
    
    # Print scoring system guide
    aggregator.print_scoring_system_guide()
    
    # Load data
    results_df = aggregator.load_all_results()
    metadata_df = aggregator.load_metadata()
    
    print(f"\nData processing completed!")
    print(f"Main results shape: {results_df.shape}")
    print(f"Era-specific files saved to 'raw_data' folder")
    print(f"Summary files saved to 'data_summary' folder")


if __name__ == "__main__":
    main()