"""Attributing band sightings to people."""

from __future__ import annotations

from safeframe.pipeline.types import BandVia, Box
from safeframe.pipeline.wearers import attribute, frame_hits, match_poses, wrist_rois

from .conftest import ellipse_mask, marker_at, person, pose

LEFT = (20, 40, 100, 220)
RIGHT = (90, 40, 170, 220)  # overlaps LEFT


def test_wrist_keypoint_wins_over_overlapping_boxes():
    people = [person(1, *LEFT), person(2, *RIGHT)]
    poses = {1: pose(LEFT, (95, 130)), 2: pose(RIGHT, (160, 130))}
    att = attribute(marker_at(96, 132), people, poses)
    assert att.track_ids == (1,) and att.via is BandVia.WRIST


def test_two_wrists_at_the_band_protect_both():
    # Holding hands: both wrists are right at the band.
    people = [person(1, *LEFT), person(2, *RIGHT)]
    poses = {1: pose(LEFT, (95, 130)), 2: pose(RIGHT, (96, 131))}
    att = attribute(marker_at(95, 130), people, poses)
    assert set(att.track_ids) == {1, 2}


def test_falls_back_to_silhouette_then_box():
    mask = ellipse_mask(Box(*LEFT), 320, 240)
    people = [person(1, *LEFT, mask=mask), person(2, *RIGHT)]
    # Inside the overlap of the boxes, but only inside person 1's silhouette.
    att = attribute(marker_at(70, 130), people, {})
    assert att.track_ids == (1,) and att.via is BandVia.BODY

    boxed = attribute(marker_at(95, 60), [person(1, *LEFT), person(2, *RIGHT)], {})
    assert set(boxed.track_ids) == {1, 2} and boxed.via is BandVia.BOX

    orphan = attribute(marker_at(300, 10), people, {})
    assert orphan.track_ids == () and orphan.via is None


def test_frame_hits_keeps_best_evidence_per_person():
    people = [person(1, *LEFT), person(2, *RIGHT)]
    poses = [pose(LEFT, (40, 130)), pose(RIGHT, (160, 130))]
    dets = [marker_at(95, 60), marker_at(41, 131), marker_at(300, 10)]
    hits, orphan = frame_hits(7, dets, people, poses)
    assert orphan
    assert hits[1].via is BandVia.WRIST and not hits[1].ambiguous and hits[1].frame == 7
    # Person 2 only shares the ambiguous box sighting.
    assert hits[2].via is BandVia.BOX and hits[2].ambiguous


def test_pose_matching_is_one_to_one():
    people = [person(1, *LEFT), person(2, 200, 40, 280, 220)]
    poses = [pose((22, 42, 98, 218)), pose((205, 45, 280, 220)), pose((0, 0, 5, 5))]
    matched = match_poses(poses, people)
    assert matched[1] is poses[0] and matched[2] is poses[1]


def test_wrist_rois_are_scaled_to_the_person():
    rois = wrist_rois([pose((0, 0, 100, 500), (50, 250))])
    assert len(rois) == 1
    assert rois[0].width == rois[0].height == 0.22 * 500
