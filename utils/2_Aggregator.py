# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import re
# from typing import Dict, List, Tuple, Optional
# from datetime import datetime
# import warnings
# import os
# warnings.filterwarnings('ignore')

# class IFSCDataAggregator:
#     """Aggregates and processes IFSC climbing competition data with era-based grouping."""
    
#     def __init__(self, data_dir: str = "./API_Results_Expanded", output_dir: str = "./Data"):
#         self.data_dir = Path(data_dir)
#         self.output_dir = Path(output_dir)
#         self.results_df = None
#         self.metadata_df = None
        
#         # Define scoring system eras with proper naming
#         self.scoring_systems = {
#             'Lead': {
#                 'IFSC_Modern_2007-2025': {'start': 2007, 'end': 2025, 'description': 'IFSC system with qualis, semis, finals'},
#                 'UIAA_Legacy_1991-2006': {'start': 1991, 'end': 2006, 'description': 'UIAA system, ranking only'}
#             },
#             'Boulder': {
#                 'IFSC_AddedPoints_2025-2025': {'start': 2025, 'end': 2025, 'description': 'Combined point system (25 top, 10 zone, -0.1 per attempt)'},
#                 'IFSC_ZoneTop_2007-2024': {'start': 2007, 'end': 2024, 'description': 'Separate top/zone counting system'},
#                 'UIAA_Legacy_1991-2006': {'start': 1991, 'end': 2006, 'description': 'UIAA system, ranking only'}
#             },
#             'Speed': {
#                 'IFSC_Time_2009-2025': {'start': 2009, 'end': 2025, 'description': 'Timed system with quali (2 laps) and finals (tournament)'},
#                 'IFSC_Score_2007-2008': {'start': 2007, 'end': 2008, 'description': 'IFSC scoring system, no times'},
#                 'UIAA_Legacy_1991-2006': {'start': 1991, 'end': 2006, 'description': 'UIAA system, ranking only'}
#             }
#         }
        
#         # Create output directories
#         self._create_output_directories()
        
#     def _create_output_directories(self):
#         """Create necessary output directories."""
#         try:
#             # Main output directory
#             self.output_dir.mkdir(exist_ok=True)
            
#             # Subdirectories
#             (self.output_dir / "visuals").mkdir(exist_ok=True)
#             (self.output_dir / "aggregate_data").mkdir(exist_ok=True)
#             (self.output_dir / "data_summary").mkdir(exist_ok=True)
#             (self.output_dir / "reports").mkdir(exist_ok=True)
            
#             print(f"Created output directories:")
#             print(f"  Main: {self.output_dir}")
#             print(f"  Visuals: {self.output_dir / 'visuals'}")
#             print(f"  Raw Data: {self.output_dir / 'aggregate_data'}")
#             print(f"  Data Summary: {self.output_dir / 'data_summary'}")
#             print(f"  Reports: {self.output_dir / 'reports'}")
#         except Exception as e:
#             print(f"Error creating directories: {e}")
#             raise
        
#     def load_all_results(self) -> pd.DataFrame:
#         """Load and combine all result CSV files."""
#         print("Loading all result files...")
        
#         # Check if data directory exists
#         if not self.data_dir.exists():
#             raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
#         all_results = []
#         total_files = 0
#         failed_files = []
        
#         # Walk through all year directories
#         year_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
#         if not year_dirs:
#             print(f"No year directories found in {self.data_dir}")
#             # Try loading CSV files directly from the main directory
#             csv_files = list(self.data_dir.glob("*.csv"))
#             if csv_files:
#                 print(f"Found {len(csv_files)} CSV files in main directory")
#                 for csv_file in csv_files:
#                     try:
#                         df = pd.read_csv(csv_file, encoding='utf-8')
#                         if not df.empty:
#                             # Add file metadata
#                             df['source_file'] = csv_file.name
#                             df['file_year'] = 'Unknown'
#                             df['file_path'] = str(csv_file)
#                             all_results.append(df)
#                             total_files += 1
#                         else:
#                             print(f"  Empty file: {csv_file.name}")
#                     except Exception as e:
#                         print(f"  Error loading {csv_file}: {e}")
#                         failed_files.append(str(csv_file))
#             else:
#                 raise ValueError("No CSV files or year directories found!")
#         else:
#             # Process year directories
#             for year_dir in sorted(year_dirs):
#                 print(f"Processing {year_dir.name}...")
#                 year_files = 0
                
#                 csv_files = list(year_dir.glob("*.csv"))
#                 if not csv_files:
#                     print(f"  No CSV files found in {year_dir}")
#                     continue
                
#                 for csv_file in csv_files:
#                     try:
#                         # Try different encodings
#                         df = None
#                         for encoding in ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252']:
#                             try:
#                                 df = pd.read_csv(csv_file, encoding=encoding)
#                                 break
#                             except UnicodeDecodeError:
#                                 continue
                        
#                         if df is None:
#                             print(f"  Could not decode {csv_file.name}")
#                             failed_files.append(str(csv_file))
#                             continue
                            
#                         if not df.empty:
#                             # Add file metadata
#                             df['source_file'] = csv_file.name
#                             df['file_year'] = year_dir.name
#                             df['file_path'] = str(csv_file)
#                             all_results.append(df)
#                             year_files += 1
#                         else:
#                             print(f"  Empty file: {csv_file.name}")
#                     except Exception as e:
#                         print(f"  Error loading {csv_file}: {e}")
#                         failed_files.append(str(csv_file))
                
#                 total_files += year_files
#                 print(f"  Loaded {year_files} files")
        
#         if not all_results:
#             print("Failed files:")
#             for file in failed_files[:10]:  # Show first 10 failed files
#                 print(f"  {file}")
#             raise ValueError("No result files could be loaded!")
        
#         # Combine all results
#         print("Combining results...")
#         try:
#             self.results_df = pd.concat(all_results, ignore_index=True, sort=False)
#         except Exception as e:
#             print(f"Error combining DataFrames: {e}")
#             print("Trying to diagnose column issues...")
            
#             # Show column info for first few dataframes
#             for i, df in enumerate(all_results[:5]):
#                 print(f"DataFrame {i} columns: {list(df.columns)}")
#                 print(f"DataFrame {i} shape: {df.shape}")
            
#             # Try combining with outer join
#             self.results_df = pd.concat(all_results, ignore_index=True, sort=False, join='outer')
        
#         print(f"\nTotal files loaded: {total_files}")
#         print(f"Failed files: {len(failed_files)}")
#         print(f"Total records: {len(self.results_df)}")
#         print(f"Columns found: {list(self.results_df.columns)}")
        
#         # Basic data cleaning
#         self._clean_basic_data()
        
#         # Add scoring system classification
#         self._classify_scoring_systems()
        
#         # Save aggregated data (main file and era-based files)
#         self._save_all_data()
        
#         return self.results_df
    
#     def load_metadata(self) -> pd.DataFrame:
#         """Load event metadata files."""
#         print("Loading event metadata...")
        
#         metadata_dir = Path("./API_Event_metadata")
#         if not metadata_dir.exists():
#             print(f"Metadata directory not found: {metadata_dir}")
#             print("Creating basic metadata from results")
#             return self._create_basic_metadata()
        
#         metadata_files = list(metadata_dir.glob("*.csv"))
#         if not metadata_files:
#             print("No metadata files found, creating basic metadata from results")
#             return self._create_basic_metadata()
        
#         all_metadata = []
#         for file in metadata_files:
#             try:
#                 df = pd.read_csv(file, encoding='utf-8')
#                 if not df.empty:
#                     all_metadata.append(df)
#             except Exception as e:
#                 print(f"Error loading {file}: {e}")
        
#         if all_metadata:
#             self.metadata_df = pd.concat(all_metadata, ignore_index=True)
#             print(f"Loaded metadata for {len(self.metadata_df)} events")
            
#             # Save metadata
#             self._save_metadata()
#         else:
#             print("No metadata could be loaded, creating basic metadata from results")
#             return self._create_basic_metadata()
        
#         return self.metadata_df
    
#     def _clean_basic_data(self):
#         """Basic data cleaning operations."""
#         print("Cleaning data...")
        
#         # Check if DataFrame is empty
#         if self.results_df is None or self.results_df.empty:
#             print("No data to clean")
#             return
        
#         # Standardize column names
#         self.results_df.columns = [col.lower().strip().replace(' ', '_') for col in self.results_df.columns]
        
#         # Clean athlete names
#         name_columns = ['name', 'athlete_name', 'athlete', 'climber']
#         for col in name_columns:
#             if col in self.results_df.columns:
#                 self.results_df['name'] = self.results_df[col].astype(str).str.strip()
#                 break
        
#         # Clean countries
#         country_columns = ['country', 'nation', 'nationality']
#         for col in country_columns:
#             if col in self.results_df.columns:
#                 self.results_df['country'] = self.results_df[col].astype(str).str.strip().str.upper()
#                 break
            
#         # Parse dates
#         date_columns = ['start_date', 'date', 'event_date', 'competition_date']
#         for col in date_columns:
#             if col in self.results_df.columns:
#                 self.results_df[col] = pd.to_datetime(self.results_df[col], errors='coerce')
        
#         # Extract competition info from filename
#         self._extract_competition_info()
        
#         # Convert year to numeric for classification
#         year_columns = ['year', 'file_year']
#         for col in year_columns:
#             if col in self.results_df.columns:
#                 self.results_df['year'] = pd.to_numeric(self.results_df[col], errors='coerce')
#                 break
        
#         # If no year found, try to extract from date or filename
#         if 'year' not in self.results_df.columns or self.results_df['year'].isna().all():
#             self._extract_year_from_data()
        
#         # Add processing timestamp
#         self.results_df['processed_at'] = datetime.now()
        
#         print(f"Data cleaning completed. Shape: {self.results_df.shape}")
#         print(f"Columns after cleaning: {list(self.results_df.columns)}")
    
#     def _extract_year_from_data(self):
#         """Extract year from various sources."""
#         print("Extracting year from data...")
        
#         # Try to extract year from date columns
#         date_columns = [col for col in self.results_df.columns if 'date' in col.lower()]
#         for col in date_columns:
#             try:
#                 dates = pd.to_datetime(self.results_df[col], errors='coerce')
#                 years = dates.dt.year
#                 if not years.isna().all():
#                     self.results_df['year'] = years
#                     print(f"Extracted year from {col}")
#                     return
#             except:
#                 continue
        
#         # Try to extract from filename or file_year
#         if 'file_year' in self.results_df.columns:
#             try:
#                 self.results_df['year'] = pd.to_numeric(self.results_df['file_year'], errors='coerce')
#                 if not self.results_df['year'].isna().all():
#                     print("Used file_year as year")
#                     return
#             except:
#                 pass
        
#         # Try to extract from source_file
#         if 'source_file' in self.results_df.columns:
#             # Look for 4-digit years in filename
#             def extract_year_from_filename(filename):
#                 if pd.isna(filename):
#                     return None
#                 matches = re.findall(r'(\d{4})', str(filename))
#                 for match in matches:
#                     year = int(match)
#                     if 1990 <= year <= 2025:  # Reasonable year range
#                         return year
#                 return None
            
#             self.results_df['year'] = self.results_df['source_file'].apply(extract_year_from_filename)
#             if not self.results_df['year'].isna().all():
#                 print("Extracted year from filename")
#                 return
        
#         print("Could not extract year from data")
    
#     def _extract_competition_info(self):
#         """Extract competition information from source filenames."""
#         if 'source_file' not in self.results_df.columns:
#             return
        
#         print("Extracting competition info from filenames...")
        
#         try:
#             # Parse filename format: date_location_discipline_gender_round.csv
#             file_parts = self.results_df['source_file'].str.replace('.csv', '').str.split('_')
            
#             # Only proceed if we have enough parts
#             valid_parts = file_parts.str.len() >= 5
            
#             self.results_df.loc[valid_parts, 'comp_date'] = file_parts[valid_parts].str[0]
#             self.results_df.loc[valid_parts, 'comp_location'] = file_parts[valid_parts].str[1]  
#             self.results_df.loc[valid_parts, 'comp_discipline'] = file_parts[valid_parts].str[2]
#             self.results_df.loc[valid_parts, 'comp_gender'] = file_parts[valid_parts].str[3]
#             self.results_df.loc[valid_parts, 'comp_round'] = file_parts[valid_parts].str[4]
            
#             # Fill NaN values for files that don't match expected format
#             for col in ['comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round']:
#                 if col in self.results_df.columns:
#                     self.results_df[col] = self.results_df[col].fillna('Unknown')
            
#             print(f"Extracted competition info for {valid_parts.sum()} records")
#         except Exception as e:
#             print(f"Could not parse filename components: {e}")
#             # Create default columns
#             for col in ['comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round']:
#                 self.results_df[col] = 'Unknown'
    
#     def _classify_scoring_systems(self):
#         """Classify each record by its scoring system era."""
#         print("Classifying scoring systems by era...")
        
#         if self.results_df is None or self.results_df.empty:
#             print("No data to classify")
#             return
        
#         def get_scoring_era(row):
#             # Try multiple discipline column names
#             discipline = None
#             for col in ['discipline', 'comp_discipline', 'event_discipline']:
#                 if col in row and pd.notna(row[col]):
#                     discipline = row[col]
#                     break
            
#             if not discipline:
#                 discipline = 'Unknown'
            
#             # Try multiple year column names
#             year = None
#             for col in ['year', 'event_year', 'comp_year']:
#                 if col in row and pd.notna(row[col]):
#                     year = row[col]
#                     break
            
#             if pd.isna(year) or year == 0:
#                 return f'{discipline}_Unknown'
            
#             try:
#                 year = int(year)
#             except (ValueError, TypeError):
#                 return f'{discipline}_Unknown'
            
#             # Normalize discipline names
#             discipline_map = {
#                 'Lead': 'Lead', 'L': 'Lead', 'lead': 'Lead',
#                 'Boulder': 'Boulder', 'B': 'Boulder', 'boulder': 'Boulder',
#                 'Speed': 'Speed', 'S': 'Speed', 'speed': 'Speed',
#                 'Combined': 'Combined', 'combined': 'Combined'
#             }
            
#             discipline = discipline_map.get(discipline, discipline)
            
#             if discipline not in self.scoring_systems:
#                 return f'{discipline}_Unknown'
            
#             # Find the appropriate era
#             for era_name, era_info in self.scoring_systems[discipline].items():
#                 if era_info['start'] <= year <= era_info['end']:
#                     return f'{discipline}_{era_name}'
            
#             return f'{discipline}_Unknown'
        
#         self.results_df['scoring_era'] = self.results_df.apply(get_scoring_era, axis=1)
        
#         # Print era distribution
#         if 'scoring_era' in self.results_df.columns:
#             era_counts = self.results_df['scoring_era'].value_counts()
#             print("\nScoring Era Distribution:")
#             for era, count in era_counts.items():
#                 print(f"  {era}: {count:,} records")
    
#     def _save_all_data(self):
#         """Save aggregated results - both main file and era-based splits."""
#         if self.results_df is None or self.results_df.empty:
#             print("No data to save")
#             return
        
#         try:
#             # Save main aggregated file
#             main_file = self.output_dir / "aggregate_data" / "Aggregated_results.csv"
#             self.results_df.to_csv(main_file, index=False, encoding='utf-8')
#             print(f"Saved main aggregated results to: {main_file}")
            
#             # Save era-based splits
#             self._save_era_based_files()
            
#             # Save summary statistics
#             self._save_summary_statistics()
#         except Exception as e:
#             print(f"Error saving data: {e}")
    
#     def _save_era_based_files(self):
#         """Save separate files for each scoring era and gender combination."""
#         print("Saving era-based files...")
        
#         try:
#             raw_data_dir = self.output_dir / "aggregate_data"
            
#             # Get unique combinations of era, gender, and discipline
#             groupby_cols = ['scoring_era']
            
#             # Find gender column
#             gender_col = None
#             for col in ['gender', 'comp_gender', 'sex']:
#                 if col in self.results_df.columns:
#                     gender_col = col
#                     groupby_cols.append(col)
#                     break
            
#             if len(groupby_cols) == 1:
#                 print("No gender column found, grouping by era only")
            
#             grouped = self.results_df.groupby(groupby_cols)
            
#             file_summary = []
            
#             # Define files to skip
#             skip_patterns = ['Unknown', 'Combined_']
            
#             for group_keys, group_df in grouped:
#                 if isinstance(group_keys, str):
#                     group_keys = (group_keys,)
                
#                 scoring_era = group_keys[0] if len(group_keys) > 0 else 'Unknown'
#                 gender = group_keys[1] if len(group_keys) > 1 else 'All'
                
#                 # Skip files matching skip patterns
#                 should_skip = any(pattern in scoring_era for pattern in skip_patterns)
#                 if should_skip:
#                     print(f"  Skipping {scoring_era}_{gender} (matches skip pattern)")
#                     continue
                
#                 # Create filename with proper format
                
#                 filename = f"{scoring_era}_{gender}.csv"
#                 filepath = raw_data_dir / filename
                
#                 # Save the group
#                 group_df.to_csv(filepath, index=False, encoding='utf-8')
                
#                 # Calculate statistics safely
#                 unique_athletes = 0
#                 if 'name' in group_df.columns:
#                     unique_athletes = group_df['name'].nunique()
                
#                 year_range = 'Unknown'
#                 if 'year' in group_df.columns and not group_df['year'].isna().all():
#                     year_min = group_df['year'].min()
#                     year_max = group_df['year'].max()
#                     year_range = f"{year_min:.0f}-{year_max:.0f}"
                
#                 file_summary.append({
#                     'filename': filename,
#                     'scoring_era': scoring_era,
#                     'gender': gender,
#                     'record_count': len(group_df),
#                     'unique_athletes': unique_athletes,
#                     'year_range': year_range
#                 })
                
#                 print(f"  Saved {filename}: {len(group_df):,} records")
            
#             # Save file summary in data_summary folder
#             if file_summary:
#                 summary_df = pd.DataFrame(file_summary)
#                 summary_file = self.output_dir / "data_summary" / "era_files_summary.csv"
#                 summary_df.to_csv(summary_file, index=False, encoding='utf-8')
#                 print(f"Saved era files summary to: {summary_file}")
#         except Exception as e:
#             print(f"Error saving era-based files: {e}")
    
#     def _save_summary_statistics(self):
#         """Save comprehensive summary statistics."""
#         try:
#             summary_dir = self.output_dir / "data_summary"
            
#             # Helper function to safely get unique count
#             def safe_nunique(column_name):
#                 if column_name in self.results_df.columns:
#                     return self.results_df[column_name].nunique()
#                 return 'N/A'
            
#             # Helper function to safely get min/max
#             def safe_min_max(column_name):
#                 if column_name in self.results_df.columns and not self.results_df[column_name].isna().all():
#                     return self.results_df[column_name].min(), self.results_df[column_name].max()
#                 return 'N/A', 'N/A'
            
#             year_min, year_max = safe_min_max('year')
            
#             # Basic statistics
#             summary_file = summary_dir / "data_summary.csv"
#             basic_stats = {
#                 'metric': [
#                     'total_records', 'unique_athletes', 'unique_competitions', 
#                     'unique_countries', 'date_range_start', 'date_range_end',
#                     'disciplines', 'genders', 'rounds', 'processing_date'
#                 ],
#                 'value': [
#                     len(self.results_df),
#                     safe_nunique('name'),
#                     safe_nunique('source_file'),
#                     safe_nunique('country'),
#                     year_min,
#                     year_max,
#                     safe_nunique('discipline') if safe_nunique('discipline') != 'N/A' else safe_nunique('comp_discipline'),
#                     safe_nunique('gender') if safe_nunique('gender') != 'N/A' else safe_nunique('comp_gender'),
#                     safe_nunique('round') if safe_nunique('round') != 'N/A' else safe_nunique('comp_round'),
#                     datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                 ]
#             }
            
#             pd.DataFrame(basic_stats).to_csv(summary_file, index=False, encoding='utf-8')
#             print(f"Saved data summary to: {summary_file}")
            
#             # Save detailed era statistics
#             era_stats_file = summary_dir / "era_statistics.csv"
#             if 'scoring_era' in self.results_df.columns:
#                 # Build aggregation dictionary based on available columns
#                 agg_dict = {}
                
#                 if 'name' in self.results_df.columns:
#                     agg_dict['name'] = 'nunique'
#                 if 'country' in self.results_df.columns:
#                     agg_dict['country'] = 'nunique'
#                 if 'year' in self.results_df.columns:
#                     agg_dict['year'] = ['min', 'max', 'nunique']
#                 if 'source_file' in self.results_df.columns:
#                     agg_dict['source_file'] = 'nunique'
                
#                 if agg_dict:
#                     era_stats = self.results_df.groupby(['scoring_era']).agg(agg_dict).round(2)
                    
#                     # Flatten column names
#                     if isinstance(era_stats.columns, pd.MultiIndex):
#                         era_stats.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in era_stats.columns]
                    
#                     era_stats = era_stats.reset_index()
#                     era_stats.to_csv(era_stats_file, index=False, encoding='utf-8')
#                     print(f"Saved era statistics to: {era_stats_file}")
#         except Exception as e:
#             print(f"Error saving summary statistics: {e}")
    
#     def _create_basic_metadata(self) -> pd.DataFrame:
#         """Create basic metadata from results if metadata files don't exist."""
#         if self.results_df is None or self.results_df.empty:
#             return pd.DataFrame()
        
#         try:
#             # Group by competition identifiers
#             group_cols = []
#             for col in ['comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round']:
#                 if col in self.results_df.columns:
#                     group_cols.append(col)
            
#             if not group_cols:
#                 print("No competition identifier columns found for metadata creation")
#                 return pd.DataFrame()
            
#             metadata = self.results_df.groupby(group_cols).agg({
#                 'name': 'count' if 'name' in self.results_df.columns else 'size',
#                 'file_year': 'first' if 'file_year' in self.results_df.columns else lambda x: 'Unknown',
#                 'scoring_era': 'first' if 'scoring_era' in self.results_df.columns else lambda x: 'Unknown'
#             }).rename(columns={'name': 'athlete_count'}).reset_index()
            
#             self.metadata_df = metadata
#             return metadata
#         except Exception as e:
#             print(f"Error creating basic metadata: {e}")
#             return pd.DataFrame()
    
#     def _save_metadata(self):
#         """Save metadata to CSV."""
#         if self.metadata_df is not None and not self.metadata_df.empty:
#             try:
#                 output_file = self.output_dir / "data_summary" / "event_metadata.csv"
#                 self.metadata_df.to_csv(output_file, index=False, encoding='utf-8')
#                 print(f"Saved metadata to: {output_file}")
#             except Exception as e:
#                 print(f"Error saving metadata: {e}")
    
#     def print_scoring_system_guide(self):
#         """Print a guide to the scoring systems used."""
#         print("\nScoring System Classification Guide:")
#         print("=" * 50)
        
#         for discipline, systems in self.scoring_systems.items():
#             print(f"\n{discipline}:")
#             for era_name, era_info in systems.items():
#                 print(f"  {era_name} ({era_info['start']}-{era_info['end']}): {era_info['description']}")
        
#         if self.results_df is not None and 'scoring_era' in self.results_df.columns:
#             print(f"\nCurrent Data Distribution:")
#             era_counts = self.results_df['scoring_era'].value_counts()
#             for era, count in era_counts.items():
#                 print(f"  {era}: {count:,} records")


# def main():
#     """Main analysis pipeline."""
#     print("Enhanced IFSC Climbing Data Analysis Pipeline")
#     print("=" * 60)
    
#     try:
#         # 1. Load and aggregate data
#         aggregator = IFSCDataAggregator()
        
#         # Print scoring system guide
#         aggregator.print_scoring_system_guide()
        
#         # Load data
#         results_df = aggregator.load_all_results()
#         metadata_df = aggregator.load_metadata()
        
#         print(f"\nData processing completed successfully!")
#         print(f"Main results shape: {results_df.shape}")
#         print(f"Era-specific files saved to 'aggregate_data' folder")
#         print(f"Summary files saved to 'data_summary' folder")
        
#     except Exception as e:
#         print(f"\nError in main pipeline: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()

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
    
    def __init__(self, data_dir: str = "./API_Results_Expanded", output_dir: str = "./Data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.results_df = None
        self.metadata_df = None
        
        # Define scoring system eras with proper naming
        self.scoring_systems = {
            'Lead': {
                'IFSC_Modern_2007-2025': {'start': 2007, 'end': 2025, 'description': 'IFSC system with qualis, semis, finals'},
                'UIAA_Legacy_1991-2006': {'start': 1991, 'end': 2006, 'description': 'UIAA system, ranking only'}
            },
            'Boulder': {
                'IFSC_AddedPoints_2025-2025': {'start': 2025, 'end': 2025, 'description': 'Combined point system (25 top, 10 zone, -0.1 per attempt)'},
                'IFSC_ZoneTop_2007-2024': {'start': 2007, 'end': 2024, 'description': 'Separate top/zone counting system'},
                'UIAA_Legacy_1991-2006': {'start': 1991, 'end': 2006, 'description': 'UIAA system, ranking only'}
            },
            'Speed': {
                'IFSC_Time_2009-2025': {'start': 2009, 'end': 2025, 'description': 'Timed system with quali (2 laps) and finals (tournament)'},
                'IFSC_Score_2007-2008': {'start': 2007, 'end': 2008, 'description': 'IFSC scoring system, no times'},
                'UIAA_Legacy_1991-2006': {'start': 1991, 'end': 2006, 'description': 'UIAA system, ranking only'}
            }
        }
        
        # Create output directories
        self._create_output_directories()
        
    def _create_output_directories(self):
        """Create necessary output directories."""
        try:
            # Main output directory
            self.output_dir.mkdir(exist_ok=True)
            
            # Subdirectories
            (self.output_dir / "visuals").mkdir(exist_ok=True)
            (self.output_dir / "aggregate_data").mkdir(exist_ok=True)
            (self.output_dir / "data_summary").mkdir(exist_ok=True)
            (self.output_dir / "reports").mkdir(exist_ok=True)
            
            print(f"Created output directories:")
            print(f"  Main: {self.output_dir}")
            print(f"  Visuals: {self.output_dir / 'visuals'}")
            print(f"  Raw Data: {self.output_dir / 'aggregate_data'}")
            print(f"  Data Summary: {self.output_dir / 'data_summary'}")
            print(f"  Reports: {self.output_dir / 'reports'}")
        except Exception as e:
            print(f"Error creating directories: {e}")
            raise
        
    def load_all_results(self) -> pd.DataFrame:
        """Load and combine all result CSV files."""
        print("Loading all result files...")
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        all_results = []
        total_files = 0
        failed_files = []
        
        # Walk through all year directories
        year_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not year_dirs:
            print(f"No year directories found in {self.data_dir}")
            # Try loading CSV files directly from the main directory
            csv_files = list(self.data_dir.glob("*.csv"))
            if csv_files:
                print(f"Found {len(csv_files)} CSV files in main directory")
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file, encoding='utf-8')
                        if not df.empty:
                            # Add file metadata
                            df['source_file'] = csv_file.name
                            df['file_year'] = 'Unknown'
                            df['file_path'] = str(csv_file)
                            all_results.append(df)
                            total_files += 1
                        else:
                            print(f"  Empty file: {csv_file.name}")
                    except Exception as e:
                        print(f"  Error loading {csv_file}: {e}")
                        failed_files.append(str(csv_file))
            else:
                raise ValueError("No CSV files or year directories found!")
        else:
            # Process year directories
            for year_dir in sorted(year_dirs):
                print(f"Processing {year_dir.name}...")
                year_files = 0
                
                csv_files = list(year_dir.glob("*.csv"))
                if not csv_files:
                    print(f"  No CSV files found in {year_dir}")
                    continue
                
                for csv_file in csv_files:
                    try:
                        # Try different encodings
                        df = None
                        for encoding in ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252']:
                            try:
                                df = pd.read_csv(csv_file, encoding=encoding)
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if df is None:
                            print(f"  Could not decode {csv_file.name}")
                            failed_files.append(str(csv_file))
                            continue
                            
                        if not df.empty:
                            # Add file metadata
                            df['source_file'] = csv_file.name
                            df['file_year'] = year_dir.name
                            df['file_path'] = str(csv_file)
                            all_results.append(df)
                            year_files += 1
                        else:
                            print(f"  Empty file: {csv_file.name}")
                    except Exception as e:
                        print(f"  Error loading {csv_file}: {e}")
                        failed_files.append(str(csv_file))
                
                total_files += year_files
                print(f"  Loaded {year_files} files")
        
        if not all_results:
            print("Failed files:")
            for file in failed_files[:10]:  # Show first 10 failed files
                print(f"  {file}")
            raise ValueError("No result files could be loaded!")
        
        # Combine all results
        print("Combining results...")
        try:
            self.results_df = pd.concat(all_results, ignore_index=True, sort=False)
        except Exception as e:
            print(f"Error combining DataFrames: {e}")
            print("Trying to diagnose column issues...")
            
            # Show column info for first few dataframes
            for i, df in enumerate(all_results[:5]):
                print(f"DataFrame {i} columns: {list(df.columns)}")
                print(f"DataFrame {i} shape: {df.shape}")
            
            # Try combining with outer join
            self.results_df = pd.concat(all_results, ignore_index=True, sort=False, join='outer')
        
        print(f"\nTotal files loaded: {total_files}")
        print(f"Failed files: {len(failed_files)}")
        print(f"Total records: {len(self.results_df)}")
        print(f"Columns found: {list(self.results_df.columns)}")
        
        # Basic data cleaning
        self._clean_basic_data()
        
        # Add scoring system classification
        self._classify_scoring_systems()
        
        # Extract and parse start_date from filenames
        self._extract_start_date()
        
        # Save aggregated data (main file and era-based files)
        self._save_all_data()
        
        return self.results_df
    
    def _extract_start_date(self):
        """Extract and parse start_date from filename patterns."""
        print("Extracting start_date from filenames...")
        
        if 'source_file' not in self.results_df.columns:
            print("No source_file column found, cannot extract start_date")
            return
        
        def parse_date_from_filename(filename):
            """Parse date from various filename patterns."""
            if pd.isna(filename):
                return None
            
            filename = str(filename)
            
            # Pattern 1: YYYY-MM-DD_Location_Discipline_Gender_Round.csv
            date_match = re.match(r'^(\d{4}-\d{2}-\d{2})_', filename)
            if date_match:
                try:
                    return pd.to_datetime(date_match.group(1))
                except:
                    pass
            
            # Pattern 2: YYYY-MM-DD format anywhere in filename
            date_matches = re.findall(r'(\d{4}-\d{2}-\d{2})', filename)
            for date_str in date_matches:
                try:
                    return pd.to_datetime(date_str)
                except:
                    continue
            
            # Pattern 3: YYYYMMDD format
            date_matches = re.findall(r'(\d{8})', filename)
            for date_str in date_matches:
                try:
                    return pd.to_datetime(date_str, format='%Y%m%d')
                except:
                    continue
            
            # Pattern 4: Extract year and use January 1st as default
            year_matches = re.findall(r'(\d{4})', filename)
            for year_str in year_matches:
                year = int(year_str)
                if 1990 <= year <= 2025:  # Reasonable range
                    return pd.to_datetime(f'{year}-01-01')
            
            return None
        
        # Extract start_date from source_file
        self.results_df['start_date'] = self.results_df['source_file'].apply(parse_date_from_filename)
        
        # Fill missing dates using year information if available
        if 'year' in self.results_df.columns:
            mask_missing = self.results_df['start_date'].isna()
            mask_has_year = self.results_df['year'].notna()
            combined_mask = mask_missing & mask_has_year
            
            if combined_mask.any():
                # Use January 1st of the year as default date
                self.results_df.loc[combined_mask, 'start_date'] = pd.to_datetime(
                    self.results_df.loc[combined_mask, 'year'].astype(int).astype(str) + '-01-01'
                )
        
        # Count successful date extractions
        valid_dates = self.results_df['start_date'].notna().sum()
        total_records = len(self.results_df)
        
        print(f"  Successfully extracted start_date for {valid_dates:,} of {total_records:,} records ({valid_dates/total_records*100:.1f}%)")
        
        if valid_dates > 0:
            date_range = f"{self.results_df['start_date'].min().strftime('%Y-%m-%d')} to {self.results_df['start_date'].max().strftime('%Y-%m-%d')}"
            print(f"  Date range: {date_range}")
    
    def load_metadata(self) -> pd.DataFrame:
        """Load event metadata files."""
        print("Loading event metadata...")
        
        metadata_dir = Path("./API_Event_metadata")
        if not metadata_dir.exists():
            print(f"Metadata directory not found: {metadata_dir}")
            print("Creating basic metadata from results")
            return self._create_basic_metadata()
        
        metadata_files = list(metadata_dir.glob("*.csv"))
        if not metadata_files:
            print("No metadata files found, creating basic metadata from results")
            return self._create_basic_metadata()
        
        all_metadata = []
        for file in metadata_files:
            try:
                df = pd.read_csv(file, encoding='utf-8')
                if not df.empty:
                    all_metadata.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if all_metadata:
            self.metadata_df = pd.concat(all_metadata, ignore_index=True)
            print(f"Loaded metadata for {len(self.metadata_df)} events")
            
            # Save metadata
            self._save_metadata()
        else:
            print("No metadata could be loaded, creating basic metadata from results")
            return self._create_basic_metadata()
        
        return self.metadata_df
    
    def _clean_basic_data(self):
        """Basic data cleaning operations."""
        print("Cleaning data...")
        
        # Check if DataFrame is empty
        if self.results_df is None or self.results_df.empty:
            print("No data to clean")
            return
        
        # Standardize column names
        self.results_df.columns = [col.lower().strip().replace(' ', '_') for col in self.results_df.columns]
        
        # Clean athlete names
        name_columns = ['name', 'athlete_name', 'athlete', 'climber']
        for col in name_columns:
            if col in self.results_df.columns:
                self.results_df['name'] = self.results_df[col].astype(str).str.strip()
                break
        
        # Clean countries
        country_columns = ['country', 'nation', 'nationality']
        for col in country_columns:
            if col in self.results_df.columns:
                self.results_df['country'] = self.results_df[col].astype(str).str.strip().str.upper()
                break
            
        # Parse dates
        date_columns = ['start_date', 'date', 'event_date', 'competition_date']
        for col in date_columns:
            if col in self.results_df.columns:
                self.results_df[col] = pd.to_datetime(self.results_df[col], errors='coerce')
        
        # Extract competition info from filename
        self._extract_competition_info()
        
        # Convert year to numeric for classification
        year_columns = ['year', 'file_year']
        for col in year_columns:
            if col in self.results_df.columns:
                self.results_df['year'] = pd.to_numeric(self.results_df[col], errors='coerce')
                break
        
        # If no year found, try to extract from date or filename
        if 'year' not in self.results_df.columns or self.results_df['year'].isna().all():
            self._extract_year_from_data()
        
        # Add processing timestamp
        self.results_df['processed_at'] = datetime.now()
        
        print(f"Data cleaning completed. Shape: {self.results_df.shape}")
        print(f"Columns after cleaning: {list(self.results_df.columns)}")
    
    def _extract_year_from_data(self):
        """Extract year from various sources."""
        print("Extracting year from data...")
        
        # Try to extract year from date columns
        date_columns = [col for col in self.results_df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                dates = pd.to_datetime(self.results_df[col], errors='coerce')
                years = dates.dt.year
                if not years.isna().all():
                    self.results_df['year'] = years
                    print(f"Extracted year from {col}")
                    return
            except:
                continue
        
        # Try to extract from filename or file_year
        if 'file_year' in self.results_df.columns:
            try:
                self.results_df['year'] = pd.to_numeric(self.results_df['file_year'], errors='coerce')
                if not self.results_df['year'].isna().all():
                    print("Used file_year as year")
                    return
            except:
                pass
        
        # Try to extract from source_file
        if 'source_file' in self.results_df.columns:
            # Look for 4-digit years in filename
            def extract_year_from_filename(filename):
                if pd.isna(filename):
                    return None
                matches = re.findall(r'(\d{4})', str(filename))
                for match in matches:
                    year = int(match)
                    if 1990 <= year <= 2025:  # Reasonable year range
                        return year
                return None
            
            self.results_df['year'] = self.results_df['source_file'].apply(extract_year_from_filename)
            if not self.results_df['year'].isna().all():
                print("Extracted year from filename")
                return
        
        print("Could not extract year from data")
    
    def _extract_competition_info(self):
        """Extract competition information from source filenames."""
        if 'source_file' not in self.results_df.columns:
            return
        
        print("Extracting competition info from filenames...")
        
        try:
            # Parse filename format: date_location_discipline_gender_round.csv
            file_parts = self.results_df['source_file'].str.replace('.csv', '').str.split('_')
            
            # Only proceed if we have enough parts
            valid_parts = file_parts.str.len() >= 5
            
            self.results_df.loc[valid_parts, 'comp_date'] = file_parts[valid_parts].str[0]
            self.results_df.loc[valid_parts, 'comp_location'] = file_parts[valid_parts].str[1]  
            self.results_df.loc[valid_parts, 'comp_discipline'] = file_parts[valid_parts].str[2]
            self.results_df.loc[valid_parts, 'comp_gender'] = file_parts[valid_parts].str[3]
            self.results_df.loc[valid_parts, 'comp_round'] = file_parts[valid_parts].str[4]
            
            # Fill NaN values for files that don't match expected format
            for col in ['comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round']:
                if col in self.results_df.columns:
                    self.results_df[col] = self.results_df[col].fillna('Unknown')
            
            print(f"Extracted competition info for {valid_parts.sum()} records")
        except Exception as e:
            print(f"Could not parse filename components: {e}")
            # Create default columns
            for col in ['comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round']:
                self.results_df[col] = 'Unknown'
    
    def _classify_scoring_systems(self):
        """Classify each record by its scoring system era."""
        print("Classifying scoring systems by era...")
        
        if self.results_df is None or self.results_df.empty:
            print("No data to classify")
            return
        
        def get_scoring_era(row):
            # Try multiple discipline column names
            discipline = None
            for col in ['discipline', 'comp_discipline', 'event_discipline']:
                if col in row and pd.notna(row[col]):
                    discipline = row[col]
                    break
            
            if not discipline:
                discipline = 'Unknown'
            
            # Try multiple year column names
            year = None
            for col in ['year', 'event_year', 'comp_year']:
                if col in row and pd.notna(row[col]):
                    year = row[col]
                    break
            
            if pd.isna(year) or year == 0:
                return f'{discipline}_Unknown'
            
            try:
                year = int(year)
            except (ValueError, TypeError):
                return f'{discipline}_Unknown'
            
            # Normalize discipline names
            discipline_map = {
                'Lead': 'Lead', 'L': 'Lead', 'lead': 'Lead',
                'Boulder': 'Boulder', 'B': 'Boulder', 'boulder': 'Boulder',
                'Speed': 'Speed', 'S': 'Speed', 'speed': 'Speed',
                'Combined': 'Combined', 'combined': 'Combined'
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
        if 'scoring_era' in self.results_df.columns:
            era_counts = self.results_df['scoring_era'].value_counts()
            print("\nScoring Era Distribution:")
            for era, count in era_counts.items():
                print(f"  {era}: {count:,} records")
    
    def _save_all_data(self):
        """Save aggregated results - both main file and era-based splits."""
        if self.results_df is None or self.results_df.empty:
            print("No data to save")
            return
        
        try:
            # Save main aggregated file
            main_file = self.output_dir / "aggregate_data" / "aggregated_results.csv"
            self.results_df.to_csv(main_file, index=False, encoding='utf-8')
            print(f"Saved main aggregated results to: {main_file}")
            
            # Save era-based splits
            self._save_era_based_files()
            
            # Save summary statistics
            self._save_summary_statistics()
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def _save_era_based_files(self):
        """Save separate files for each scoring era and gender combination."""
        print("Saving era-based files...")
        
        try:
            raw_data_dir = self.output_dir / "aggregate_data"
            
            # Get unique combinations of era, gender, and discipline
            groupby_cols = ['scoring_era']
            
            # Find gender column
            gender_col = None
            for col in ['gender', 'comp_gender', 'sex']:
                if col in self.results_df.columns:
                    gender_col = col
                    groupby_cols.append(col)
                    break
            
            if len(groupby_cols) == 1:
                print("No gender column found, grouping by era only")
            
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
                group_df.to_csv(filepath, index=False, encoding='utf-8')
                
                # Calculate statistics safely
                unique_athletes = 0
                if 'name' in group_df.columns:
                    unique_athletes = group_df['name'].nunique()
                
                year_range = 'Unknown'
                if 'year' in group_df.columns and not group_df['year'].isna().all():
                    year_min = group_df['year'].min()
                    year_max = group_df['year'].max()
                    year_range = f"{year_min:.0f}-{year_max:.0f}"
                
                # Get date range if available
                date_range_start = None
                date_range_end = None
                if 'start_date' in group_df.columns and not group_df['start_date'].isna().all():
                    date_range_start = group_df['start_date'].min()
                    date_range_end = group_df['start_date'].max()
                
                file_summary.append({
                    'filename': filename,
                    'scoring_era': scoring_era,
                    'gender': gender,
                    'record_count': len(group_df),
                    'unique_athletes': unique_athletes,
                    'year_range': year_range,
                    'date_range_start': date_range_start,
                    'date_range_end': date_range_end
                })
                
                print(f"  Saved {filename}: {len(group_df):,} records")
            
            # Save file summary in data_summary folder
            if file_summary:
                summary_df = pd.DataFrame(file_summary)
                summary_file = self.output_dir / "data_summary" / "era_files_summary.csv"
                summary_df.to_csv(summary_file, index=False, encoding='utf-8')
                print(f"Saved era files summary to: {summary_file}")
        except Exception as e:
            print(f"Error saving era-based files: {e}")
    
    def _save_summary_statistics(self):
        """Save comprehensive summary statistics."""
        try:
            summary_dir = self.output_dir / "data_summary"
            
            # Helper function to safely get unique count
            def safe_nunique(column_name):
                if column_name in self.results_df.columns:
                    return self.results_df[column_name].nunique()
                return 'N/A'
            
            # Helper function to safely get min/max
            def safe_min_max(column_name):
                if column_name in self.results_df.columns and not self.results_df[column_name].isna().all():
                    return self.results_df[column_name].min(), self.results_df[column_name].max()
                return 'N/A', 'N/A'
            
            year_min, year_max = safe_min_max('year')
            
            # Get start_date range if available
            start_date_min, start_date_max = safe_min_max('start_date')
            
            # Basic statistics
            summary_file = summary_dir / "data_summary.csv"
            basic_stats = {
                'metric': [
                    'total_records', 'unique_athletes', 'unique_competitions', 
                    'unique_countries', 'date_range_start', 'date_range_end',
                    'start_date_min', 'start_date_max',
                    'disciplines', 'genders', 'rounds', 'processing_date'
                ],
                'value': [
                    len(self.results_df),
                    safe_nunique('name'),
                    safe_nunique('source_file'),
                    safe_nunique('country'),
                    year_min,
                    year_max,
                    start_date_min.strftime('%Y-%m-%d') if isinstance(start_date_min, pd.Timestamp) else start_date_min,
                    start_date_max.strftime('%Y-%m-%d') if isinstance(start_date_max, pd.Timestamp) else start_date_max,
                    safe_nunique('discipline') if safe_nunique('discipline') != 'N/A' else safe_nunique('comp_discipline'),
                    safe_nunique('gender') if safe_nunique('gender') != 'N/A' else safe_nunique('comp_gender'),
                    safe_nunique('round') if safe_nunique('round') != 'N/A' else safe_nunique('comp_round'),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            
            pd.DataFrame(basic_stats).to_csv(summary_file, index=False, encoding='utf-8')
            print(f"Saved data summary to: {summary_file}")
            
            # Save detailed era statistics
            era_stats_file = summary_dir / "era_statistics.csv"
            if 'scoring_era' in self.results_df.columns:
                # Build aggregation dictionary based on available columns
                agg_dict = {}
                
                if 'name' in self.results_df.columns:
                    agg_dict['name'] = 'nunique'
                if 'country' in self.results_df.columns:
                    agg_dict['country'] = 'nunique'
                if 'year' in self.results_df.columns:
                    agg_dict['year'] = ['min', 'max', 'nunique']
                if 'source_file' in self.results_df.columns:
                    agg_dict['source_file'] = 'nunique'
                if 'start_date' in self.results_df.columns:
                    agg_dict['start_date'] = ['min', 'max']
                
                if agg_dict:
                    era_stats = self.results_df.groupby(['scoring_era']).agg(agg_dict).round(2)
                    
                    # Flatten column names
                    if isinstance(era_stats.columns, pd.MultiIndex):
                        era_stats.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in era_stats.columns]
                    
                    era_stats = era_stats.reset_index()
                    era_stats.to_csv(era_stats_file, index=False, encoding='utf-8')
                    print(f"Saved era statistics to: {era_stats_file}")
        except Exception as e:
            print(f"Error saving summary statistics: {e}")
    
    def _create_basic_metadata(self) -> pd.DataFrame:
        """Create basic metadata from results if metadata files don't exist."""
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()
        
        try:
            # Group by competition identifiers
            group_cols = []
            for col in ['comp_date', 'comp_location', 'comp_discipline', 'comp_gender', 'comp_round']:
                if col in self.results_df.columns:
                    group_cols.append(col)
            
            if not group_cols:
                print("No competition identifier columns found for metadata creation")
                return pd.DataFrame()
            
            # Build aggregation dictionary
            agg_dict = {'name': 'count' if 'name' in self.results_df.columns else 'size'}
            
            # Add other columns if available
            if 'file_year' in self.results_df.columns:
                agg_dict['file_year'] = 'first'
            if 'scoring_era' in self.results_df.columns:
                agg_dict['scoring_era'] = 'first'
            if 'start_date' in self.results_df.columns:
                agg_dict['start_date'] = 'first'
            
            metadata = self.results_df.groupby(group_cols).agg(agg_dict)
            metadata = metadata.rename(columns={'name': 'athlete_count'}).reset_index()
            
            self.metadata_df = metadata
            return metadata
        except Exception as e:
            print(f"Error creating basic metadata: {e}")
            return pd.DataFrame()
    
    def _save_metadata(self):
        """Save metadata to CSV."""
        if self.metadata_df is not None and not self.metadata_df.empty:
            try:
                output_file = self.output_dir / "data_summary" / "event_metadata.csv"
                self.metadata_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"Saved metadata to: {output_file}")
            except Exception as e:
                print(f"Error saving metadata: {e}")
    
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
    
    try:
        # 1. Load and aggregate data
        aggregator = IFSCDataAggregator()
        
        # Print scoring system guide
        aggregator.print_scoring_system_guide()
        
        # Load data
        results_df = aggregator.load_all_results()
        metadata_df = aggregator.load_metadata()
        
        print(f"\nData processing completed successfully!")
        print(f"Main results shape: {results_df.shape}")
        print(f"Era-specific files saved to 'aggregate_data' folder")
        print(f"Summary files saved to 'data_summary' folder")
        
        # Show start_date statistics
        if 'start_date' in results_df.columns:
            valid_dates = results_df['start_date'].notna().sum()
            print(f"Start dates extracted: {valid_dates:,} records")
            if valid_dates > 0:
                date_range = f"{results_df['start_date'].min().strftime('%Y-%m-%d')} to {results_df['start_date'].max().strftime('%Y-%m-%d')}"
                print(f"Date range: {date_range}")
        
    except Exception as e:
        print(f"\nError in main pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()