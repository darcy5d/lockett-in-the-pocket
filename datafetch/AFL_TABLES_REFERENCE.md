# AFL Tables — reference for our database

[AFL Tables](https://afltables.com/afl/afl_index.html) is the canonical source for AFL match and player data. They state **"Complete to End of season, 2025"**. Our current pipeline uses the [akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) repo (which is derived from AFL Tables); this doc maps AFL Tables URLs and structure to our schema so an AFL Tables fetcher can stay in sync.

## Useful AFL Tables URLs

| What | URL pattern / example |
|------|------------------------|
| Index | https://afltables.com/afl/afl_index.html |
| Match scores 1897–2025 | https://afltables.com/afl/seas/season_idx.html |
| Season scores (one year) | https://afltables.com/afl/seas/YYYY.html (e.g. [2025](https://afltables.com/afl/seas/2025.html), [2024](https://afltables.com/afl/seas/2024.html)) |
| 2025 / 2024 player stats | https://afltables.com/afl/stats/2025.html, .../2024.html |
| Team index (e.g. Collingwood) | https://afltables.com/afl/teams/collingwood_idx.html |
| Single match stats (scores + lineups + player stats) | https://afltables.com/afl/stats/games/YYYY/... (e.g. [Sydney v Hawthorn R1 2025](https://afltables.com/afl/stats/games/2025/101620250307.html)) |

## How AFL Tables lines up with our schema

### Match data (`afl_data/data/matches/matches_YYYY.csv`)

- **Season page** (e.g. `seas/2025.html`): each round has matches with:
  - **Round** — maps to our `round_num` (1, 2, … or "Preliminary Final", "Grand Final"; we store as string).
  - **Team names** — e.g. Sydney, Hawthorn, Greater Western Sydney, Collingwood. These match our **internal** team names in `core/mappings.py` (no mapping needed).
  - **Quarter scores** — e.g. `4.3 6.4 11.5 11.10` → our `team_1_q1_goals`, `team_1_q1_behinds`, … `team_1_final_goals`, `team_1_final_behinds` (and same for team_2).
  - **Date/time** — e.g. `Fri 07-Mar-2025 7:40 PM` → we use `date` like `2025-03-07 19:40`.
  - **Venue** — e.g. S.C.G., M.C.G., Docklands, Sydney Showground, Adelaide Oval, Perth Stadium, Gabba, Kardinia Park, Carrara. Same as our **internal** venue names; no mapping needed for writing CSVs.
  - **Attendance** — maps to our `attendance` (number or time string; we can normalise if needed).

- **Match stats page** (e.g. `stats/games/2025/101620250307.html`): same match with:
  - Quarter scores in goals.behinds form (confirms/corrects season page).
  - **Player Details** and **Match Statistics** tables — both list players per team. Player links use relative hrefs like `../../players/T/Taylor_Adams.html` (not `stats/players/`). Names appear as "Surname, FirstName" → we use "FirstName LastName" in lineups. The scraper detects tables by first-row text ("Sydney Player Details", "Hawthorn Match Statistics", etc.) and extracts player names from links matching `players/`.

### Lineups (`afl_data/data/lineups/team_lineups_<team>.csv`)

- Our format: `year,date,round_num,team_name,player1;player2;...` with players as "FirstName LastName".
- AFL Tables match stats page lists every player in the match; team name is in the section header (Sydney / Hawthorn). Round and date are in the page title/header. So a scraper can build lineup rows from each match stats page (and optionally deduplicate by round if the season page is also scraped for match list).

### Player performance stats (`afl_data/data/players/*_performance_details.csv`)

- **Match Statistics** tables on match stats pages include full stat lines per player (KI=Kicks, MK=Marks, HB=Handballs, GL=Goals, BH=Behinds, HO=Hit outs, TK=Tackles, RB=Rebound 50s, IF=Inside 50s, CL=Clearances, CG=Clangers, FF/FA=Free kicks, etc.). Column order is in the "Abbreviations key" table. These align with `core/afl_data_store.STAT_FIELDS`. A future fetcher could parse these tables to produce `*_performance_details.csv` files for model training.

## Practical use

- **Full 2025 data**: AFL Tables has the full 2025 season (all rounds + finals). Our current DB only has 2025 rounds 1–4 (from akareen at the time of the last fetch). A scraper that writes `matches_YYYY.csv` and lineup CSVs from AFL Tables would bring us up to "end of 2025".
- **Ongoing updates**: Re-running an AFL Tables fetcher each season (or after each round) would keep matches and lineups aligned with the canonical source without depending on akareen’s update cycle.
- **Respectful scraping**: Use a reasonable delay between requests and a identifiable User-Agent; prefer scraping season/match list from `seas/YYYY.html` and then fetching only needed match stats pages to minimise load.

The `datafetch/afl_tables_scraper.py` fetches matches and lineups from AFL Tables into `afl_data_afltables/data/`. Match stats URLs from the season page use relative hrefs (e.g. `../stats/games/2025/xxx.html`); the scraper uses `urljoin` with the season page URL to resolve them correctly.
