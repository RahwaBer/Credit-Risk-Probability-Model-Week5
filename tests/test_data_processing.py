import pytest
from utils import calculate_rfm_score

def calculate_rfm_score(recency, frequency, monetary):
    """Returns a weighted RFM score (out of 5)"""
    if any(val < 0 for val in [recency, frequency, monetary]):
        raise ValueError("R, F, and M must be non-negative")

    return round((0.2 * recency + 0.4 * frequency + 0.4 * monetary) / 10, 2)

def test_rfm_score_valid_input():
    # Arrange
    r, f, m = 10, 20, 30
    # Act
    result = calculate_rfm_score(r, f, m)
    # Assert
    expected = round((0.2*10 + 0.4*20 + 0.4*30)/10, 2)
    assert result == expected

def test_rfm_score_negative_input():
    # Arrange
    r, f, m = -5, 20, 30
    # Act & Assert
    with pytest.raises(ValueError, match="non-negative"):
        calculate_rfm_score(r, f, m)


