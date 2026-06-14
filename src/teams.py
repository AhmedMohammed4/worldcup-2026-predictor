"""
Team name normalization map. Every data source routes through normalize_team()
so that names are consistent across openfootball, international results CSV,
and betting APIs.
"""

# Map of variant names to canonical name.
# Canonical names generally follow the short common English name.
TEAM_ALIASES = {
    # Americas
    "United States": "USA",
    "United States of America": "USA",
    "US": "USA",

    # Korea
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Korea, Republic of": "South Korea",

    # UK
    "Northern Ireland": "Northern Ireland",

    # Others
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Antigua and Barbuda": "Antigua & Barbuda",
    "Saint Kitts and Nevis": "St. Kitts & Nevis",
    "Saint Vincent and the Grenadines": "St. Vincent & Grenadines",
    "São Tomé and Príncipe": "Sao Tome & Principe",
    "Sao Tome and Principe": "Sao Tome & Principe",
    "DR Congo": "Congo DR",
    "Congo Republic": "Congo",
    "Republic of Ireland": "Ireland",
    "Eswatini": "Swaziland",
    "Cabo Verde": "Cape Verde",
    "Timor-Leste": "East Timor",
    "Brunei Darussalam": "Brunei",
}


def normalize_team(name: str) -> str:
    """Return the canonical team name."""
    if name is None or (isinstance(name, float) and name != name):
        return None
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)
