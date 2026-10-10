import pytest

from reviewpipe import contracts
from tests.conftest import product, review


def test_a_valid_review_becomes_a_typed_row():
    row = contracts.review(review("U1", "P1", day=0, rating=4, votes=3, images=[{}, {}]))
    assert row["rating"] == 4 and row["helpful_vote"] == 3 and row["image_count"] == 2


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({"rating": 6}, "bad_rating"),
        ({"rating": 3.5}, "bad_rating"),
        ({"rating": True}, "bad_rating"),
        ({"timestamp": "yesterday"}, "bad_timestamp"),
        ({"timestamp": 1_600_000_000}, "timestamp_out_of_range"),  # seconds instead of milliseconds
        ({"helpful_vote": -1}, "bad_helpful_vote"),
        ({"helpful_vote": True}, "bad_helpful_vote"),
        ({"verified_purchase": "yes"}, "bad_verified_purchase"),
        ({"user_id": " "}, "missing_user_id"),
        ({"parent_asin": None}, "missing_parent_asin"),
        ({"text": 12}, "bad_text"),
    ],
)
def test_each_broken_field_has_its_own_reason(patch, reason):
    with pytest.raises(contracts.Rejected) as e:
        contracts.review({**review("U1", "P1", day=0), **patch})
    assert e.value.reason == reason


@pytest.mark.parametrize(
    ("raw", "dollars"),
    [
        (12.5, 12.5),
        ("$1,299.99", 1299.99),
        ("19", 19.0),
        ("$ 5.5", 5.5),
        (None, None),
        ("None", None),
        ("$5 - $9", None),
        ("from $3", None),
        (-1, None),
        (float("nan"), None),
        (True, None),
    ],
)
def test_prices_parse_or_are_unknown(raw, dollars):
    assert contracts.parse_price(raw) == dollars


def test_meta_rejects_impossible_ratings_and_counts_text():
    with pytest.raises(contracts.Rejected):
        contracts.meta(product("P1", avg=7))
    row = contracts.meta({**product("P1"), "description": ["ab", "cde", 5]})
    assert row["description_len"] == 5 and row["feature_count"] == 1
