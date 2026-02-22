# Core Module - Map-First Layer

This module provides canonical mappings so the rest of the pipeline uses consistent internal identifiers. **Map first, analyze next.**

## Mappings

### Team names
- **Internal** (match CSVs, lineup `team_name`): Sydney, Brisbane Lions, Greater Western Sydney, etc.
- **External** (fixturedownload): Sydney Swans, GWS GIANTS, Gold Coast SUNS, etc.
- Use `TeamNameMapper.to_internal(external)` and `TeamNameMapper.to_display(internal)`

### Venues
- **Internal** (match CSV `venue`): S.C.G., M.C.G., Docklands, Carrara, Sydney Showground, etc.
- **External** (fixturedownload): SCG, MCG, Marvel Stadium, People First Stadium, ENGIE Stadium
- Use `VenueMapper.to_internal(external)` and `VenueMapper.to_display(internal)`

### Players
- **Lineup format**: "FirstName LastName" (semicolon-separated in lineup `players` column)
- **Player index format**: "lastname_firstname_DDMMYYYY" (e.g. crisp_jack_02101993)
- Use `PlayerMapper.to_player_id("Jack Crisp")` and `PlayerMapper.to_display("crisp_jack_02101993")`
- Handles compound surnames: "Jordan de Goey" -> goey_jordan_15031996

## Validation

Run `python scripts/validate_mappings.py` from project root to verify all mappings.
