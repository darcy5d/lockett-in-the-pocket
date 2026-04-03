# AFL Data Fetching

This directory contains scripts for fetching and loading AFL (Australian Football League) data from the [akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) repository.

## Rugby League (NRL) Data

- **`rebuild_nrl_from_rlp.py`**: Scrapes [Rugby League Project](https://www.rugbyleagueproject.org/) for match and lineup data. Supports NRL, NSW Cup, QLD Cup, UK Super League, UK Championship.
- **NSW Cup full lineage**: Run `python datafetch/rebuild_nrl_from_rlp.py --competition nsw-cup` to fetch reserve grade (1908–1983), NSWRL reserve (1984–1997), first division (1998–2007), and NSW Cup (2008–2025). See `nrl_competition_history.md` for lineage details.
- **Output**: `nrl_data/data/matches/matches_{slug}_{year_from}_{year_to}.csv` and `nrl_data/data/lineups/lineup_details_{slug}_*.csv`

## Files

- `fetch_afl_data.py`: Script to download the latest AFL data
- `load_afl_data.py`: Script to load and process the downloaded AFL data
- `requirements.txt`: List of Python packages required for data fetching

## Usage

### Fetching Data

To download the latest AFL data, run:

```bash
cd datafetch
python fetch_afl_data.py
```

This will download the data into the `afl_data` directory at the project root.

### Loading Data

After downloading the data, you can use the `AFLDataLoader` class to load and explore the data:

```python
from datafetch.load_afl_data import AFLDataLoader

# Initialize the loader
loader = AFLDataLoader()

# Load match data for 2023
matches_2023 = loader.load_matches(2023)

# Load player performance data for 'Bontempelli'
bontempelli_performance = loader.load_player_data('bontempelli', data_type='performance')

# Load player personal data for 'Bontempelli'
bontempelli_personal = loader.load_player_data('bontempelli', data_type='personal')

# Load odds data
odds_data = loader.load_odds_data()
``` 