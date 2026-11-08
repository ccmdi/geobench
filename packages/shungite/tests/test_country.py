import pytest

from shungite import same_country
from shungite.country import canonical_country


@pytest.mark.parametrize(
    ("truth", "guess"),
    [
        ("United States of America (the)", "United States"),
        ("United States of America (the)", "usa"),
        ("Korea (the Republic of)", "South Korea"),
        ("Türkiye", "Turkey"),
        ("Czechia", "Czech Republic"),
        ("Russian Federation (the)", "russia"),
        ("Netherlands (the)", "Holland"),
        ("United Kingdom of Great Britain and Northern Ireland (the)", "Scotland"),
        ("China", "Hong Kong"),
        ("Reunion", "Réunion"),
        ("Reunion", "France"),
        ("Reunion", "Réunion (France)"),
        ("Reunion", "France (Réunion)"),
        ("Reunion", "Réunion, France"),
        ("Netherlands (Kingdom of the)", "Netherlands"),
        ("Portugal", "Portugal (Azores)"),
        ("Puerto Rico", "Puerto Rico (United States)"),
        ("Spain", "Spain (Canary Islands)"),
        ("Palestine, State of", "palestine"),
        ("Tanzania, United Republic of", "Tanzania"),
        ("New Zealand", "New\xa0Zealand"),
        ("Somaliland", "  somaliland "),
        ("Atlantis", " atlantis "),
    ],
)
def test_same(truth: str, guess: str) -> None:
    assert same_country(truth, guess)
    assert same_country(guess, truth)


@pytest.mark.parametrize(
    ("truth", "guess"),
    [
        ("Reunion", "Mauritius"),
        ("France", "Belgium"),
        ("Korea (the Republic of)", "North Korea"),
        ("Laos", "Laos/Cambodia"),
        ("United States of America (the)", "Georgia (US)"),
        ("Georgia", "Georgia (US)"),
        ("Haiti", "Haiti, Dominican Republic"),
        ("France", ""),
        ("", ""),
        ("   ", "   "),
    ],
)
def test_different(truth: str, guess: str) -> None:
    assert not same_country(truth, guess)


def test_somaliland_counts_as_somalia_and_kosovo_stands_alone() -> None:
    assert same_country("Somaliland", "Somalia")
    assert same_country("Somalia (specifically Somaliland region)", "Somalia")
    assert same_country("Kosovo", "Republic of Kosovo")
    assert not same_country("Kosovo", "Serbia")


def test_a_name_it_does_not_know_keeps_its_casing() -> None:
    assert canonical_country(" Saint  Barthelemy ") == "Saint Barthelemy"
