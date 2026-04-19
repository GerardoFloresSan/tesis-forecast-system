import pytest

from app.utils.channel_rules import (
    canonicalize_channel,
    get_shift_label,
    get_slots_per_day,
)


def test_canonicalize_channel_choice_returns_choice():
    assert canonicalize_channel("choice") == "Choice"


def test_canonicalize_channel_espana_returns_espana():
    assert canonicalize_channel("espana") == "España"


def test_canonicalize_channel_mexico_raises_value_error():
    with pytest.raises(ValueError, match="Canal 'Mexico' no permitido para LSTM"):
        canonicalize_channel("Mexico")


def test_get_slots_per_day_returns_34():
    assert get_slots_per_day("Choice") == 34


def test_get_shift_label_classifies_morning_and_afternoon_correctly():
    assert get_shift_label("Choice", 11 * 60 + 30) == "morning"
    assert get_shift_label("Choice", 12 * 60) == "afternoon"
    assert get_shift_label("España", 16 * 60 + 30) == "afternoon"