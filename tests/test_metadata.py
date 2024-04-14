from thundera.metadata import parse_range, Range
import pytest


@pytest.mark.parametrize(
    "range, expected",
    [
        ("[-5, 5]", (-5, 5, True, True)),
        ("(-5, 5]", (-5, 5, False, True)),
        ("[-5, 5)", (-5, 5, True, False)),
        ("(-5, 5)", (-5, 5, False, False)),
    ],
)
def test_parse_range(range: str, expected: tuple[float, float, bool, bool]):
    assert parse_range(range) == dict(
        zip(("start", "end", "include_start", "include_end"), expected)
    )


invalid_range_format_test_cases = [
    ("5, 10", ValueError, "Invalid range format"),
    ("]5, 10]", ValueError, "Invalid range format"),
    ("(5, 10[", ValueError, "Invalid range format"),
]


@pytest.mark.parametrize("range, expected, message", invalid_range_format_test_cases)
def test_fail_parse_range(range: str, expected: Exception, message: str):
    with pytest.raises(expected, match=message):
        parse_range(range)


invalid_range_specification_test_cases = [
    (
        "[10, 5]",
        ValueError,
        "Start value should be less than or equal to end. Got start = 10.0 and end = 5.0",
    )
]


@pytest.mark.parametrize(
    "range, expected, message",
    invalid_range_format_test_cases + invalid_range_specification_test_cases,
)
def test_fail_range(range: str, expected: Exception, message: str):
    with pytest.raises(expected, match=message):
        Range(**parse_range(range))
