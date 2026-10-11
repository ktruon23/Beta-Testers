import pytest
from pydantic import ValidationError

from shared.schemas import Climber


def test_reach_is_wingspan_times_factor():
    assert Climber(height_in=66, wingspan_in=68).reach_in == pytest.approx(57.8)
    assert Climber(height_in=66, wingspan_in=68, reach_factor=1.0).reach_in == 68


@pytest.mark.parametrize("field", ["height_in", "wingspan_in"])
def test_zero_or_negative_body_size_rejected(field):
    kwargs = {"height_in": 66, "wingspan_in": 68, field: 0}
    with pytest.raises(ValidationError):
        Climber(**kwargs)


@pytest.mark.parametrize("height, wingspan", [(170, 68), (66, 180)])
def test_centimeters_by_mistake_rejected(height, wingspan):
    # 180 cm typed into an inches field would give a 153 in reach and one giant "move".
    with pytest.raises(ValidationError):
        Climber(height_in=height, wingspan_in=wingspan)


def test_unknown_level_rejected():
    with pytest.raises(ValidationError):
        Climber(height_in=66, wingspan_in=68, level="pro")
