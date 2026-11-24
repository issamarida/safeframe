"""Sealed originals and the k-of-n trustee escrow."""

from __future__ import annotations

import contextlib
import itertools
import os

import pytest

from safeframe import seal
from safeframe.seal import (
    EscrowPublicKey,
    SealError,
    TrusteeShare,
    combine_shares,
    create_escrow,
    read_header,
    seal_file,
    split_secret,
    unseal_file,
)


def test_any_k_shares_rebuild_the_secret_and_fewer_do_not():
    secret = os.urandom(32)
    points = split_secret(secret, 3, 5)
    for subset in itertools.combinations(points, 3):
        assert combine_shares(subset) == secret
    for subset in itertools.combinations(points, 2):
        # The interpolated value usually doesn't even fit in 32 bytes.
        with contextlib.suppress(ValueError):
            assert combine_shares(subset) != secret


@pytest.mark.parametrize(("k", "n"), [(1, 3), (4, 3), (2, 256)])
def test_split_validates_parameters(k, n):
    with pytest.raises(ValueError):
        split_secret(b"x" * 32, k, n)


@pytest.fixture
def escrow():
    return create_escrow(3, 5)


@pytest.fixture
def sealed(tmp_path, escrow, monkeypatch):
    monkeypatch.setattr(seal, "CHUNK_BYTES", 1000)  # exercise many chunks
    public, _ = escrow
    original = tmp_path / "original.jpg"
    original.write_bytes(os.urandom(4500))
    published = tmp_path / "shared.jpg"
    published.write_bytes(b"blurred")
    envelope = tmp_path / "shared.jpg.sfseal"
    seal_file(
        original,
        envelope,
        public,
        reason="band honoured",
        published=published,
        context={"guarded": False},
        now=1_000_000.0,
    )
    return original, envelope


def test_round_trip_with_a_quorum(tmp_path, escrow, sealed):
    _, shares = escrow
    original, envelope = sealed
    out = tmp_path / "out.jpg"
    header = unseal_file(envelope, [shares[4], shares[0], shares[2]], out, now=1_000_001.0)
    assert out.read_bytes() == original.read_bytes()
    assert header.reason == "band honoured" and header.original_name == "original.jpg"
    assert header.published_sha256 == seal.hashlib.sha256(b"blurred").hexdigest()
    assert read_header(envelope).expires_at == 1_000_000.0 + 30 * 86_400


def test_envelope_does_not_contain_the_plaintext(sealed):
    original, envelope = sealed
    data = envelope.read_bytes()
    assert original.read_bytes()[:64] not in data


def test_too_few_or_foreign_shares_are_refused(tmp_path, escrow, sealed):
    _, shares = escrow
    _, envelope = sealed
    out = tmp_path / "out.jpg"
    with pytest.raises(SealError, match="needs 3"):
        unseal_file(envelope, shares[:2], out, now=1_000_001.0)
    with pytest.raises(SealError, match="needs 3"):
        unseal_file(envelope, [shares[0]] * 3, out, now=1_000_001.0)
    _, other = create_escrow(3, 5)
    with pytest.raises(SealError, match="belong"):
        unseal_file(envelope, other[:3], out, now=1_000_001.0)
    forged = [TrusteeShare(s.key_id, s.threshold, s.trustees, s.x, s.y + 1) for s in shares[:3]]
    with pytest.raises(SealError, match="rebuild"):
        unseal_file(envelope, forged, out, now=1_000_001.0)
    assert not out.exists()


def test_expired_envelopes_are_refused(tmp_path, escrow, sealed):
    _, shares = escrow
    _, envelope = sealed
    later = 1_000_000.0 + 31 * 86_400
    with pytest.raises(SealError, match="expired"):
        unseal_file(envelope, shares[:3], tmp_path / "o", now=later)
    unseal_file(envelope, shares[:3], tmp_path / "o", now=later, allow_expired=True)


def _flip(path, offset):
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


def test_tampering_is_detected(tmp_path, escrow, sealed):
    _, shares = escrow
    _, envelope = sealed
    pristine = envelope.read_bytes()
    out = tmp_path / "out.jpg"

    # Extending the expiry in the header breaks the wrapped key.
    text = pristine.replace(b'"expires_at":3592000.0', b'"expires_at":9592000.0')
    assert text != pristine
    envelope.write_bytes(text)
    with pytest.raises(SealError, match="altered"):
        unseal_file(envelope, shares[:3], out, now=1_000_001.0)

    envelope.write_bytes(pristine)
    _flip(envelope, len(pristine) - 20)
    with pytest.raises(SealError, match="altered"):
        unseal_file(envelope, shares[:3], out, now=1_000_001.0)

    envelope.write_bytes(pristine[:-1020])  # drop the final chunk
    with pytest.raises(SealError, match="truncated"):
        unseal_file(envelope, shares[:3], out, now=1_000_001.0)
    assert not out.exists()


def test_keys_and_shares_round_trip_through_json(escrow):
    public, shares = escrow
    assert EscrowPublicKey.from_json(public.to_json()) == public
    assert TrusteeShare.from_json(shares[0].to_json()) == shares[0]
    with pytest.raises(SealError):
        TrusteeShare.from_json('{"scheme": "something-else"}')


def test_not_an_envelope(tmp_path):
    bogus = tmp_path / "x.sfseal"
    bogus.write_bytes(b"hello world")
    with pytest.raises(SealError):
        read_header(bogus)
