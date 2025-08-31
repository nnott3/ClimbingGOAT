# import pandas as pd
# import numpy as np
# from datetime import datetime
# import json

# def create_time_based_elo_history(elo_history, era_key):
#     """
#     Optimized version: Only create time series records at competition dates + key milestones
#     This is much faster than creating monthly records for every athlete
#     """
#     if elo_history.empty:
#         return pd.DataFrame()
    
#     print(f"  Optimizing time-based history for {len(elo_history)} records...")
    
#     # Ensure we have date information
#     if 'date' not in elo_history.columns:
#         elo_history['date'] = pd.to_datetime(elo_history['year'].astype(str) + '-01-01')
#         print(f"  Warning: No date column found for {era_key}, using year approximation")
#     else:
#         elo_history['date'] = pd.to_datetime(elo_history['date'])
    
#     # Sort by athlete and date
#     elo_history = elo_history.sort_values(['name', 'date'])
    
#     # Instead of creating monthly records, we'll create records only at:
#     # 1. Competition dates (where ELO actually changes)
#     # 2. Year boundaries (for annual summaries)
#     # 3. Optional: quarterly milestones
    
#     time_based_records = []
    
#     # Get all unique competition dates
#     competition_dates = sorted(elo_history['date'].unique())
    
#     # Add year boundaries for continuity
#     min_year = elo_history['date'].dt.year.min()
#     max_year = elo_history['date'].dt.year.max()
#     year_boundaries = [pd.Timestamp(f'{year}-01-01') for year in range(min_year, max_year + 1)]
#     year_boundaries = [d for d in year_boundaries if d not in competition_dates]
    
#     # Combine and sort all key dates
#     key_dates = sorted(list(competition_dates) + year_boundaries)
    
#     print(f"  Processing {len(key_dates)} key dates instead of full monthly series...")
    
#     # Process each athlete
#     for athlete in elo_history['name'].unique():
#         athlete_data = elo_history[elo_history['name'] == athlete].copy()
#         athlete_data = athlete_data.sort_values('date')
        
#         current_elo = 1500  # Default starting ELO
#         competitions_count = 0
#         last_competition_date = None
        
#         for date in key_dates:
#             # Find competitions on or before this date
#             competitions_before = athlete_data[athlete_data['date'] <= date]
            
#             if not competitions_before.empty:
#                 # Use the most recent ELO rating
#                 latest_competition = competitions_before.iloc[-1]
#                 current_elo = latest_competition['elo_after']
#                 competitions_count = len(competitions_before)
#                 last_competition_date = latest_competition['date']
            
#             # Only add record if:
#             # 1. This athlete has competed at some point, OR
#             # 2. This is a competition date where someone competed
#             if competitions_count > 0 or date in competition_dates:
#                 time_based_records.append({
#                     'name': athlete,
#                     'date': date,
#                     'elo': current_elo,
#                     'has_competed': competitions_count > 0,
#                     'competitions_to_date': competitions_count,
#                     'last_competition_date': last_competition_date,
#                     'days_since_last_competition': (date - last_competition_date).days if last_competition_date else None,
#                     'is_competition_date': date in competition_dates,
#                     'is_year_boundary': date in year_boundaries
#                 })
    
#     result_df = pd.DataFrame(time_based_records)
#     print(f"  Created {len(result_df)} optimized time-based records")
#     return result_df

# def create_minimal_time_series(elo_history, era_key):
#     """
#     Even faster alternative: Just add forward-fill records at year boundaries
#     This preserves ELO continuity without massive data expansion
#     """
#     if elo_history.empty:
#         return pd.DataFrame()
    
#     print(f"  Creating minimal time series for {era_key}...")
    
#     # Ensure dates
#     if 'date' not in elo_history.columns:
#         elo_history['date'] = pd.to_datetime(elo_history['year'].astype(str) + '-01-01')
#     else:
#         elo_history['date'] = pd.to_datetime(elo_history['date'])
    
#     # Sort by athlete and date
#     elo_history = elo_history.sort_values(['name', 'date'])
    
#     # Get the last ELO for each athlete in each year
#     elo_history['year'] = elo_history['date'].dt.year
    
#     # For each athlete, get their last ELO rating of each year
#     yearly_elo = elo_history.groupby(['name', 'year']).last().reset_index()
    
#     # Create year-end records
#     yearly_elo['date'] = pd.to_datetime(yearly_elo['year'].astype(str) + '-12-31')
#     yearly_elo['is_year_end'] = True
    
#     # Combine original records with year-end records
#     original_records = elo_history.copy()
#     original_records['is_year_end'] = False
    
#     # Select relevant columns for consistency
#     columns = ['name', 'date', 'elo_after', 'year']
#     if 'competed' in elo_history.columns:
#         columns.append('competed')
    
#     # Rename elo_after to elo for consistency
#     yearly_records = yearly_elo[columns].copy()
#     yearly_records = yearly_records.rename(columns={'elo_after': 'elo'})
#     yearly_records['record_type'] = 'year_end'
    
#     original_records = original_records[columns].copy()
#     original_records = original_records.rename(columns={'elo_after': 'elo'})
#     original_records['record_type'] = 'competition'
    
#     # Combine and sort
#     combined = pd.concat([original_records, yearly_records], ignore_index=True)
#     combined = combined.sort_values(['name', 'date']).drop_duplicates(['name', 'date'])
    
#     print(f"  Created {len(combined)} minimal time series records")
#     return combined

# def export_all_elo_data(analyzer, base_dir):
#     """Optimized version of the ELO export with much faster time series generation"""
    
#     # Get overview to understand available data
#     overview = analyzer.get_data_overview()
#     era_files = overview.get('era_files', [])
    
#     if not era_files:
#         print("No era files found")
#         return
    
#     # Define discipline eras for systematic export
#     DISCIPLINE_ERAS = {
#         "Lead": [
#             ("UIAA_Legacy", 1991, 2006),
#             ("IFSC_Modern", 2007, 2025),
#         ],
#         "Boulder": [
#             ("UIAA_Legacy", 1991, 2006),
#             ("IFSC_ZoneTop", 2007, 2024),
#             ("IFSC_AddedPoints", 2025, 2025),
#         ],
#         "Speed": [
#             ("UIAA_Legacy", 1991, 2006),
#             ("IFSC_Score", 2007, 2008),
#             ("IFSC_Time", 2009, 2025),
#         ]
#     }
    
#     disciplines = ['Boulder', 'Lead', 'Speed']
#     genders = ['Men', 'Women']
    
#     # Initialize ELO Calculator
#     era_data = {}
#     for era_file in era_files:
#         if hasattr(analyzer, 'get_era_data'):
#             era_data[era_file] = analyzer.get_era_data(era_file)
#         elif hasattr(analyzer, 'era_files'):
#             era_data = analyzer.era_files
#             break
    
#     if not era_data:
#         print("Could not access era data from analyzer")
#         return
    
#     # Import here to avoid circular imports
#     from elo_scoring import ELOCalculator
#     elo_calculator = ELOCalculator(era_data)
    
#     # Export summary
#     export_summary = {
#         "export_timestamp": datetime.now().isoformat(),
#         "total_combinations": 0,
#         "successful_exports": 0,
#         "failed_exports": [],
#         "exported_files": [],
#         "optimization_used": "minimal_time_series"
#     }
    
#     print("Starting OPTIMIZED ELO data export...")
    
#     for discipline in disciplines:
#         for gender in genders:
#             for era_name, start_year, end_year in DISCIPLINE_ERAS.get(discipline, []):
                
#                 # Create era key
#                 era_key = f"{discipline}_{era_name}_{start_year}-{end_year}_{gender}"
#                 export_summary["total_combinations"] += 1
                
#                 print(f"\nProcessing: {era_key}")
                
#                 try:
#                     # Calculate ELO ratings
#                     elo_history = elo_calculator.calculate_elo_ratings(
#                         era_key, 
#                         discipline=discipline, 
#                         gender=gender
#                     )
                    
#                     if elo_history.empty:
#                         print(f"  No data found for {era_key}")
#                         export_summary["failed_exports"].append({
#                             "era_key": era_key,
#                             "reason": "No data found"
#                         })
#                         continue
                    
#                     # Get current leaderboard
#                     leaderboard = elo_calculator.get_current_leaderboard(
#                         era_key,
#                         discipline=discipline,
#                         gender=gender
#                     )
                    
#                     # Create safe filename
#                     safe_filename = era_key.replace(" ", "_").replace("/", "_")
                    
#                     # Export 1: Current Leaderboard
#                     leaderboard_file = base_dir / "leaderboards" / f"{safe_filename}_leaderboard.csv"
#                     leaderboard.to_csv(leaderboard_file, index=False)
#                     export_summary["exported_files"].append(str(leaderboard_file))
                    
#                     # Export 2: Complete Historical Data (competition-based)
#                     history_file = base_dir / "historical_data" / f"{safe_filename}_history.csv"
#                     elo_history.to_csv(history_file, index=False)
#                     export_summary["exported_files"].append(str(history_file))
                    
#                     # Export 3: OPTIMIZED Time-based ELO History
#                     print("  Creating optimized time-based ELO history...")
#                     start_time = datetime.now()
                    
#                     # Use the minimal time series approach for speed
#                     time_based_history = create_minimal_time_series(elo_history, era_key)
                    
#                     elapsed = (datetime.now() - start_time).total_seconds()
#                     print(f"  Time series creation took {elapsed:.2f} seconds")
                    
#                     if not time_based_history.empty:
#                         time_series_file = base_dir / "time_series" / f"{safe_filename}_timeseries.csv"
#                         time_based_history.to_csv(time_series_file, index=False)
#                         export_summary["exported_files"].append(str(time_series_file))
                        
#                         # Export top 10 time series for visualization
#                         if not leaderboard.empty:
#                             top_10_names = leaderboard.head(10)['name'].tolist()
#                             top_10_time_series = time_based_history[
#                                 time_based_history['name'].isin(top_10_names)
#                             ]
                            
#                             charts_time_file = base_dir / "charts_data" / f"{safe_filename}_top10_timeseries.csv"
#                             top_10_time_series.to_csv(charts_time_file, index=False)
#                             export_summary["exported_files"].append(str(charts_time_file))
                    
#                     # Export 4: Statistics Summary (enhanced)
#                     stats = {
#                         "era_key": era_key,
#                         "discipline": discipline,
#                         "gender": gender,
#                         "era_name": era_name,
#                         "start_year": start_year,
#                         "end_year": end_year,
#                         "total_athletes": len(leaderboard),
#                         "total_competitions": elo_history['event'].nunique() if 'event' in elo_history.columns else 0,
#                         "total_elo_records": len(elo_history),
#                         "avg_elo": float(leaderboard['current_elo'].mean()) if not leaderboard.empty else 0,
#                         "median_elo": float(leaderboard['current_elo'].median()) if not leaderboard.empty else 0,
#                         "max_elo": float(leaderboard['current_elo'].max()) if not leaderboard.empty else 0,
#                         "min_elo": float(leaderboard['current_elo'].min()) if not leaderboard.empty else 0,
#                         "most_active_athlete": leaderboard.loc[leaderboard['competitions'].idxmax(), 'name'] if not leaderboard.empty else "",
#                         "most_active_competitions": int(leaderboard['competitions'].max()) if not leaderboard.empty else 0,
#                         "highest_elo_athlete": leaderboard.loc[leaderboard['current_elo'].idxmax(), 'name'] if not leaderboard.empty else "",
#                         "years_covered": sorted(elo_history['year'].unique().tolist()) if 'year' in elo_history.columns else [],
#                         "date_range": {
#                             "start": elo_history['date'].min().isoformat() if 'date' in elo_history.columns else None,
#                             "end": elo_history['date'].max().isoformat() if 'date' in elo_history.columns else None
#                         } if 'date' in elo_history.columns else None,
#                         "time_series_records": len(time_based_history) if not time_based_history.empty else 0,
#                         "time_series_generation_seconds": elapsed
#                     }
                    
#                     stats_file = base_dir / "statistics" / f"{safe_filename}_stats.json"
#                     with open(stats_file, 'w') as f:
#                         json.dump(stats, f, indent=2, default=str)
#                     export_summary["exported_files"].append(str(stats_file))
                    
#                     # Export 5: Top 10 Historical Data (for charts)
#                     if not leaderboard.empty:
#                         top_10 = leaderboard.head(10)
#                         top_10_history = elo_history[elo_history['name'].isin(top_10['name'])]
                        
#                         charts_file = base_dir / "charts_data" / f"{safe_filename}_top10_history.csv"
#                         top_10_history.to_csv(charts_file, index=False)
#                         export_summary["exported_files"].append(str(charts_file))
                    
#                     # Export 6: Essential raw calculations (streamlined)
#                     raw_file = base_dir / "raw_calculations" / f"{safe_filename}_raw.csv"
#                     elo_history.to_csv(raw_file, index=False)
#                     export_summary["exported_files"].append(str(raw_file))
                    
#                     print(f"  ✓ Exported {len(leaderboard)} athletes, {len(elo_history)} ELO records")
#                     if not time_based_history.empty:
#                         print(f"    Time series: {len(time_based_history)} records ({elapsed:.2f}s)")
#                     export_summary["successful_exports"] += 1
                    
#                 except Exception as e:
#                     print(f"  ✗ Error processing {era_key}: {str(e)}")
#                     import traceback
#                     traceback.print_exc()
#                     export_summary["failed_exports"].append({
#                         "era_key": era_key,
#                         "reason": str(e)
#                     })
    
#     # Export master summary
#     summary_file = base_dir / "export_summary.json"
#     with open(summary_file, 'w') as f:
#         json.dump(export_summary, f, indent=2, default=str)
    
#     # Create simplified master files
#     print("\nCreating master combined files...")
    
#     try:
#         # Only combine leaderboards and essential files to avoid memory issues
#         all_leaderboards = []
        
#         for file_path in export_summary["exported_files"]:
#             if "leaderboard.csv" in file_path:
#                 df = pd.read_csv(file_path)
#                 era_info = file_path.split("/")[-1].replace("_leaderboard.csv", "").split("_")
#                 if len(era_info) >= 4:
#                     df['discipline'] = era_info[0]
#                     df['era_name'] = era_info[1]
#                     df['gender'] = era_info[-1]
#                 all_leaderboards.append(df)
        
#         if all_leaderboards:
#             master_leaderboard = pd.concat(all_leaderboards, ignore_index=True)
#             master_leaderboard.to_csv(base_dir / "master_leaderboard.csv", index=False)
#             print(f"  ✓ Master leaderboard: {len(master_leaderboard)} records")
            
#     except Exception as e:
#         print(f"  ✗ Error creating master files: {e}")
    
#     return export_summary

# # Replace the slow function in your existing script
# def main():
#     print("OPTIMIZED ELO Data Export Script")
#     print("=" * 50)
    
#     # Load analyzer (assuming this is already defined in your script)
#     from analysis import ClimbingAnalyzer
#     analyzer = ClimbingAnalyzer()
    
#     # Create export structure (assuming this function exists)
#     from pathlib import Path
#     base_dir = Path("..") / "ELO_data"
#     subdirs = ["leaderboards", "historical_data", "statistics", "charts_data", "raw_calculations", "time_series"]
    
#     for subdir in [base_dir] + [base_dir / sub for sub in subdirs]:
#         subdir.mkdir(exist_ok=True)
    
#     # Export all ELO data with optimizations
#     export_summary = export_all_elo_data(analyzer, base_dir)
    
#     print("\n" + "=" * 50)
#     print("OPTIMIZED ELO Export Complete!")
#     print(f"Successful exports: {export_summary['successful_exports']}")
#     print(f"Failed exports: {len(export_summary['failed_exports'])}")
#     print(f"Total files created: {len(export_summary['exported_files'])}")
#     print(f"Export location: {base_dir.absolute()}")
#     print("\nOptimizations applied:")
#     print("- Minimal time series generation (key dates only)")
#     print("- Reduced memory footprint")
#     print("- Faster processing with targeted data creation")

# if __name__ == "__main__":
#     main()