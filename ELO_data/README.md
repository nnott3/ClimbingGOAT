# ELO Data Export

Generated on: 2025-08-31T02:01:32.691223

## Summary
- Total combinations processed: 16
- Successful exports: 10
- Failed exports: 6
- Total files exported: 50

## Folder Structure

### leaderboards/
Current ELO rankings for each discipline/gender/era combination
- Format: `[Discipline]_[Era]_[Years]_[Gender]_leaderboard.csv`
- Contains: name, current_elo, peak_elo, competitions, avg_elo_change, elo_volatility, avg_rank, min_elo

### historical_data/
Complete ELO progression history for all athletes
- Format: `[Discipline]_[Era]_[Years]_[Gender]_history.csv`
- Contains: name, event, year, discipline, gender, round, rank, elo_before, elo_after, elo_change

### statistics/
Summary statistics for each combination
- Format: `[Discipline]_[Era]_[Years]_[Gender]_stats.json`
- Contains: athlete counts, ELO statistics, most active athletes, date ranges

### charts_data/
Top 10 athletes' historical data optimized for visualization
- Format: `[Discipline]_[Era]_[Years]_[Gender]_top10_history.csv`
- Contains: Historical progression for the top 10 current ELO-ranked athletes

### raw_calculations/
Enhanced ELO data with additional calculated metrics
- Format: `[Discipline]_[Era]_[Years]_[Gender]_raw.csv`
- Contains: All historical data plus rolling averages, rank improvements, competition sequences

## Master Files
- `master_leaderboard.csv`: All leaderboards combined
- `master_history.csv`: All historical data combined
- `export_summary.json`: This export session's metadata

## Era Definitions
- **Lead**: UIAA_Legacy (1991-2006), IFSC_Modern (2007-2025)
- **Boulder**: UIAA_Legacy (1991-2006), IFSC_ZoneTop (2007-2024), IFSC_AddedPoints (2025)
- **Speed**: UIAA_Legacy (1991-2006), IFSC_Score (2007-2008), IFSC_ShortestTime (2009-2025)

## Failed Exports
- Boulder_UIAA_Legacy_1991-2006_Men: No data found
- Boulder_UIAA_Legacy_1991-2006_Women: No data found
- Lead_UIAA_Legacy_1991-2006_Men: No data found
- Lead_UIAA_Legacy_1991-2006_Women: No data found
- Speed_UIAA_Legacy_1991-2006_Men: No data found
- Speed_UIAA_Legacy_1991-2006_Women: No data found
