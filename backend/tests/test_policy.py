import pytest

from safeframe.pipeline.policy import (
    Decision,
    DecisionPolicy,
    Override,
    aggregate,
    bucket_stats,
    minor_fraction,
    needs_age,
    parse_age_bucket,
)
from safeframe.pipeline.types import AgeSample, AgeStatus, BandHit, BandVia, Box, Track


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("0-2", (0, 2)),
        ("10-19", (10, 19)),
        ("(15, 20)", (15, 20)),
        ("more than 70", (70, 100)),
        ("70+", (70, 100)),
    ],
)
def test_parse_age_bucket(label, expected):
    assert parse_age_bucket(label) == expected


def test_parse_age_bucket_rejects_garbage():
    with pytest.raises(ValueError):
        parse_age_bucket("adult")


def test_minor_fraction_splits_straddling_bucket():
    assert minor_fraction(0, 2) == 1.0
    assert minor_fraction(10, 19) == pytest.approx(0.8)  # ages 10..17 of 10..19
    assert minor_fraction(15, 20) == pytest.approx(3 / 6)
    assert minor_fraction(18, 25) == 0.0
    assert minor_fraction(20, 29) == 0.0


def test_bucket_stats_for_fairface_labels():
    labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
    fractions, mids = bucket_stats(labels)
    assert fractions == pytest.approx([1, 1, 0.8, 0, 0, 0, 0, 0, 0])
    assert mids[0] == 1.0 and mids[2] == 14.5


def _track(samples):
    t = Track(track_id=1, boxes={0: Box(0, 0, 10, 10)}, samples=samples)
    t.p_minor, t.expected_age = aggregate(samples)
    return t


def _s(p, w=1.0, age=10.0):
    return AgeSample(frame=0, p_minor=p, expected_age=age, weight=w)


def test_aggregate_is_weighted_mean():
    p, age = aggregate([_s(1.0, w=3, age=10), _s(0.0, w=1, age=30)])
    assert p == pytest.approx(0.75)
    assert age == pytest.approx(15)
    assert aggregate([]) == (None, None)


def test_too_few_samples_is_uncertain():
    policy = DecisionPolicy(threshold=0.35, min_face_samples=3)
    track = _track([_s(0.0), _s(0.0)])  # two confident "adult" samples are not enough
    assert policy.age_status(track) is AgeStatus.UNCERTAIN


def test_threshold_splits_minor_and_adult():
    policy = DecisionPolicy(threshold=0.35, min_face_samples=2)
    assert policy.age_status(_track([_s(0.4), _s(0.4)])) is AgeStatus.MINOR
    assert policy.age_status(_track([_s(0.1), _s(0.1)])) is AgeStatus.ADULT


def _band(track, clean=1, ambiguous=0):
    for i in range(clean):
        track.add_band_hit(BandHit(i, BandVia.WRIST, False, 12.0))
    for i in range(ambiguous):
        track.add_band_hit(BandHit(100 + i, BandVia.BOX, True, 12.0))
    return track


CHILD = [_s(0.9)] * 3
ADULT = [_s(0.05)] * 3
NO_FACE: list = []


@pytest.mark.parametrize(
    ("band", "age", "guarded", "event", "expected"),
    [
        # Standard capture: the band alone decides, age is irrelevant.
        ("clean", ADULT, False, False, Decision.WEARER),
        ("clean", NO_FACE, False, False, Decision.WEARER),
        ("ambiguous", ADULT, False, False, Decision.POSSIBLE),
        (None, CHILD, False, False, Decision.VISIBLE),
        (None, NO_FACE, False, False, Decision.VISIBLE),
        # Guarded zone: only positive evidence of a child keeps the protection.
        ("clean", CHILD, True, False, Decision.WEARER),
        ("ambiguous", CHILD, True, False, Decision.POSSIBLE),
        ("clean", ADULT, True, False, Decision.NOT_HONOURED),
        ("clean", NO_FACE, True, False, Decision.NOT_HONOURED),  # masked robber
        (None, CHILD, True, False, Decision.VISIBLE),
        # Event mode: likely minors and uncertain people too, fail closed...
        (None, CHILD, False, True, Decision.MINOR),
        (None, NO_FACE, False, True, Decision.UNCERTAIN),
        (None, ADULT, False, True, Decision.VISIBLE),
        ("clean", ADULT, False, True, Decision.WEARER),
        # ...except in guarded zones, where "uncertain" protects nobody.
        (None, NO_FACE, True, True, Decision.VISIBLE),
        (None, CHILD, True, True, Decision.MINOR),
        ("clean", ADULT, True, True, Decision.NOT_HONOURED),
    ],
)
def test_decision_matrix(band, age, guarded, event, expected):
    track = _track(list(age))
    if band == "clean":
        _band(track)
    elif band == "ambiguous":
        _band(track, clean=0, ambiguous=1)
    policy = DecisionPolicy(guarded=guarded, protect_minors=event)
    assert policy.classify(track) is expected
    assert policy.should_blur(track) is (
        expected in {Decision.WEARER, Decision.POSSIBLE, Decision.MINOR, Decision.UNCERTAIN}
    )


def test_too_few_clean_sightings_is_still_protected():
    track = _band(_track([]), clean=1)
    assert DecisionPolicy(min_band_hits=2).classify(track) is Decision.POSSIBLE
    assert DecisionPolicy(min_band_hits=2).should_blur(track)


def test_overrides_win():
    policy = DecisionPolicy()
    wearer = _band(_track([]))
    bystander = _track([])
    assert policy.should_blur(bystander, Override.BLUR)
    assert not policy.should_blur(wearer, Override.KEEP)


def test_age_is_only_needed_where_it_can_change_a_decision():
    wearer, bystander = _band(_track([])), _track([])
    assert not needs_age(wearer, guarded=False, protect_minors=False)
    assert not needs_age(bystander, guarded=True, protect_minors=False)
    assert needs_age(wearer, guarded=True, protect_minors=False)
    assert needs_age(bystander, guarded=False, protect_minors=True)


def test_band_hit_storage_is_capped_but_counted():
    from safeframe.pipeline.types import MAX_BAND_HITS_STORED

    track = _band(_track([]), clean=MAX_BAND_HITS_STORED + 50)
    assert len(track.band_hits) == MAX_BAND_HITS_STORED
    assert track.band_confirmed == MAX_BAND_HITS_STORED + 50
    restored = Track.from_dict(track.to_dict())
    assert restored.band_confirmed == track.band_confirmed
    assert restored.band_hits == track.band_hits


@pytest.mark.parametrize("bad", [{"threshold": -0.1}, {"threshold": 1.5}, {"min_band_hits": 0}])
def test_policy_validates_parameters(bad):
    with pytest.raises(ValueError):
        DecisionPolicy(**bad)
