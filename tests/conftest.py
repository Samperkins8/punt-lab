import pytest

from punt_lab.nba import model


@pytest.fixture(autouse=True)
def default_model_settings():
    """Tests check the model's logic against hand-worked numbers, so run them on the hand-set defaults,
    not whatever the weekly tune last saved."""
    model.reset_defaults()
    yield
    model.reset_defaults()
    model.load_tuned()
