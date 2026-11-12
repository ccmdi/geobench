from geobench.catalog import Catalog

MODELS = {
    "lab/model-1": {"id": "lab/model-1", "name": "Lab Model 1", "release_date": "2025-01-14"},
    "reseller/other": {"id": "reseller/other", "name": "Other", "release_date": "2025-02"},
}
PROVIDERS = {
    "lab": {"id": "lab", "name": "Lab", "models": {"model-1": {"id": "model-1", "cost": {"input": 2, "output": 8}}}},
    "reseller": {"id": "reseller", "name": "Reseller", "models": {"model-1": {"id": "model-1"}, "lab-model": {"id": "lab-model", "canonical_model_id": "lab/model-1"}}},
}


def test_a_plain_id_resolves_to_the_lab_that_makes_it_not_a_reseller_listing_it() -> None:
    assert Catalog.of(MODELS, PROVIDERS).canonical("model-1") == "lab/model-1"


def test_a_hosts_own_id_resolves_through_its_canonical_model() -> None:
    assert Catalog.of(MODELS, PROVIDERS)["lab-model"].name == "Lab Model 1"


def test_a_model_its_lab_no_longer_lists_is_deprecated_and_unpriced() -> None:
    model = Catalog.of(MODELS, PROVIDERS)["reseller/other"]
    assert (model.deprecated, model.price) == (True, None)
