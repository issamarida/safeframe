"""Sealed originals: keep the evidence without keeping the exposure.

When SafeFrame blurs someone because of their band, the unblurred original
is not destroyed. It is encrypted into a *sealed envelope* that nobody can
open alone: not the photographer, not the platform, not SafeFrame.

* The envelope key is wrapped to an **escrow public key** (X25519 + HKDF-SHA256
  + AES-256-GCM). The matching private key never exists in one place: it is
  split with Shamir's secret sharing into ``n`` trustee shares, any ``k`` of
  which can rebuild it (for example 3 of 5: a court, a child-protection body,
  a civil-liberties organisation, an independent auditor, the band issuer).
  Fewer than ``k`` shares reveal nothing about the key.
* Trustees combine shares only for a lawful request, such as a court order in
  a robbery investigation. A child's face never reaches the internet, but a
  witness's photo of a robbery can still become evidence.
* Envelopes carry an expiry (30 days by default). Partner devices delete
  expired envelopes, which destroys the evidence for good; :func:`unseal_file`
  also refuses expired envelopes.
* The header (expiry, reason, capture context, hash of the published copy) is
  authenticated: changing any of it makes the envelope unopenable. Every
  chunk is authenticated with its position and a final-chunk flag, so
  truncating or reordering the ciphertext is detected.

Envelope layout (``.sfseal``)::

    b"SFSEAL1\\n" | u32 header_len | header JSON | 12-byte wrap nonce
    | u32 wrapped_len | wrapped data key | chunks...
    chunk: u32 len | AES-GCM(data_key, nonce = 0^4 || u64 index,
                             aad = sha256(header) || u64 index || final flag)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import struct
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"SFSEAL1\n"
CHUNK_BYTES = 1 << 20
DEFAULT_RETENTION_DAYS = 30
SHARE_SCHEME = "safeframe-shamir-p521-v1"
# A Mersenne prime comfortably larger than any 32-byte secret.
PRIME = 2**521 - 1
_U32 = struct.Struct(">I")
_U64 = struct.Struct(">Q")


class SealError(RuntimeError):
    """The envelope is malformed, tampered with, expired, or the shares don't open it."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode(), validate=True)


def key_id_for(public: bytes) -> str:
    return hashlib.sha256(b"safeframe-escrow-v1" + public).hexdigest()[:16]


# --- Shamir secret sharing over GF(PRIME) -------------------------------------


def split_secret(secret: bytes, threshold: int, trustees: int) -> list[tuple[int, int]]:
    """``trustees`` points on a random degree ``threshold - 1`` polynomial whose
    constant term is ``secret``."""
    if not 2 <= threshold <= trustees <= 255:
        raise ValueError("need 2 <= threshold <= trustees <= 255")
    value = int.from_bytes(secret, "big")
    if value >= PRIME:
        raise ValueError("secret too large")
    coeffs = [value] + [secrets.randbelow(PRIME) for _ in range(threshold - 1)]
    points = []
    for x in range(1, trustees + 1):
        y = 0
        for c in reversed(coeffs):
            y = (y * x + c) % PRIME
        points.append((x, y))
    return points


def combine_shares(points: Sequence[tuple[int, int]], length: int = 32) -> bytes:
    """Lagrange interpolation at x = 0."""
    xs = [x for x, _ in points]
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate shares")
    secret = 0
    for i, (xi, yi) in enumerate(points):
        num = den = 1
        for j, xj in enumerate(xs):
            if i != j:
                num = num * xj % PRIME
                den = den * (xj - xi) % PRIME
        secret = (secret + yi * num * pow(den, -1, PRIME)) % PRIME
    if secret >= 1 << (8 * length):
        raise ValueError("shares do not belong together")
    return secret.to_bytes(length, "big")


# --- Keys ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EscrowPublicKey:
    key_id: str
    public: bytes

    def to_json(self) -> str:
        return json.dumps({"key_id": self.key_id, "x25519": _b64(self.public)}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> EscrowPublicKey:
        data = json.loads(text)
        public = _unb64(data["x25519"])
        if len(public) != 32 or key_id_for(public) != data["key_id"]:
            raise SealError("escrow public key is corrupt")
        return cls(data["key_id"], public)

    @classmethod
    def load(cls, path: Path) -> EscrowPublicKey:
        return cls.from_json(Path(path).read_text())


@dataclass(frozen=True, slots=True)
class TrusteeShare:
    key_id: str
    threshold: int
    trustees: int
    x: int
    y: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "scheme": SHARE_SCHEME,
                "key_id": self.key_id,
                "threshold": self.threshold,
                "trustees": self.trustees,
                "x": self.x,
                "y": format(self.y, "x"),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> TrusteeShare:
        data = json.loads(text)
        if data.get("scheme") != SHARE_SCHEME:
            raise SealError("not a SafeFrame trustee share")
        return cls(
            data["key_id"],
            int(data["threshold"]),
            int(data["trustees"]),
            int(data["x"]),
            int(data["y"], 16),
        )

    @classmethod
    def load(cls, path: Path) -> TrusteeShare:
        return cls.from_json(Path(path).read_text())


def create_escrow(threshold: int, trustees: int) -> tuple[EscrowPublicKey, list[TrusteeShare]]:
    """Key ceremony: a fresh escrow key, split ``threshold``-of-``trustees``.

    The whole private key exists only inside this function.
    """
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    key_id = key_id_for(public)
    points = split_secret(private.private_bytes_raw(), threshold, trustees)
    shares = [TrusteeShare(key_id, threshold, trustees, x, y) for x, y in points]
    return EscrowPublicKey(key_id, public), shares


def _rebuild_private(shares: Sequence[TrusteeShare], key_id: str) -> X25519PrivateKey:
    relevant = [s for s in shares if s.key_id == key_id]
    if not relevant:
        raise SealError("none of the shares belong to this envelope's escrow key")
    need = relevant[0].threshold
    unique = {s.x: s for s in relevant}
    if len(unique) < need:
        raise SealError(f"{len(unique)} distinct share(s) given; this key needs {need}")
    try:
        raw = combine_shares([(s.x, s.y) for s in unique.values()])
        private = X25519PrivateKey.from_private_bytes(raw)
    except ValueError as exc:
        raise SealError("the shares do not rebuild the escrow key") from exc
    if key_id_for(private.public_key().public_bytes_raw()) != key_id:
        raise SealError("the shares do not rebuild the escrow key")
    return private


def _kek(shared: bytes, ephemeral: bytes, escrow: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"safeframe-seal-v1|" + ephemeral + escrow,
    ).derive(shared)


# --- Envelopes ----------------------------------------------------------------


@dataclass(slots=True)
class SealHeader:
    key_id: str
    ephemeral_public: str
    created_at: float
    expires_at: float
    original_name: str
    original_bytes: int
    # SHA-256 of the blurred copy this envelope belongs to.
    published_sha256: str | None
    reason: str
    context: dict = field(default_factory=dict)
    version: int = 1

    def encode(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def decode(cls, data: bytes) -> SealHeader:
        try:
            fields = json.loads(data)
            header = cls(**fields)
        except (ValueError, TypeError) as exc:
            raise SealError("malformed envelope header") from exc
        if header.version != 1:
            raise SealError(f"unsupported envelope version {header.version}")
        return header


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_aad(header_hash: bytes, index: int, final: bool) -> bytes:
    return header_hash + _U64.pack(index) + (b"\x01" if final else b"\x00")


def _chunk_nonce(index: int) -> bytes:
    return b"\x00\x00\x00\x00" + _U64.pack(index)


def _chunks(fh) -> Iterable[tuple[bytes, bool]]:
    """Yield (chunk, is_final); always yields at least one (possibly empty) chunk."""
    current = fh.read(CHUNK_BYTES)
    while True:
        nxt = fh.read(CHUNK_BYTES)
        yield current, not nxt
        if not nxt:
            return
        current = nxt


def seal_file(
    original: Path,
    envelope: Path,
    escrow: EscrowPublicKey,
    *,
    reason: str,
    published: Path | None = None,
    context: dict | None = None,
    retention_days: float = DEFAULT_RETENTION_DAYS,
    original_name: str | None = None,
    now: float | None = None,
) -> SealHeader:
    """Encrypt ``original`` into ``envelope`` so only the escrow quorum can open it."""
    now = time.time() if now is None else now
    ephemeral = X25519PrivateKey.generate()
    eph_pub = ephemeral.public_key().public_bytes_raw()
    kek = _kek(
        ephemeral.exchange(X25519PublicKey.from_public_bytes(escrow.public)), eph_pub, escrow.public
    )
    header = SealHeader(
        key_id=escrow.key_id,
        ephemeral_public=_b64(eph_pub),
        created_at=now,
        expires_at=now + retention_days * 86_400,
        original_name=original_name or Path(original).name,
        original_bytes=Path(original).stat().st_size,
        published_sha256=sha256_file(published) if published is not None else None,
        reason=reason,
        context=context or {},
    )
    encoded = header.encode()
    data_key = AESGCM.generate_key(bit_length=256)
    wrap_nonce = os.urandom(12)
    wrapped = AESGCM(kek).encrypt(wrap_nonce, data_key, encoded)
    aead = AESGCM(data_key)
    header_hash = hashlib.sha256(encoded).digest()
    tmp = envelope.with_name(envelope.name + ".part")
    try:
        with Path(original).open("rb") as src, tmp.open("wb") as out:
            out.write(MAGIC + _U32.pack(len(encoded)) + encoded)
            out.write(wrap_nonce + _U32.pack(len(wrapped)) + wrapped)
            for index, (chunk, final) in enumerate(_chunks(src)):
                ct = aead.encrypt(_chunk_nonce(index), chunk, _chunk_aad(header_hash, index, final))
                out.write(_U32.pack(len(ct)) + ct)
        tmp.replace(envelope)
    finally:
        tmp.unlink(missing_ok=True)
    return header


def _read_exact(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise SealError("envelope is truncated")
    return data


def _read_prelude(fh) -> tuple[SealHeader, bytes, bytes, bytes]:
    if fh.read(len(MAGIC)) != MAGIC:
        raise SealError("not a SafeFrame sealed envelope")
    (n,) = _U32.unpack(_read_exact(fh, 4))
    if n > 1 << 20:
        raise SealError("envelope header too large")
    encoded = _read_exact(fh, n)
    wrap_nonce = _read_exact(fh, 12)
    (w,) = _U32.unpack(_read_exact(fh, 4))
    if w > 1024:
        raise SealError("wrapped key too large")
    return SealHeader.decode(encoded), encoded, wrap_nonce, _read_exact(fh, w)


def read_header(envelope: Path) -> SealHeader:
    """The envelope's (unverified) public header: expiry, reason, context."""
    with Path(envelope).open("rb") as fh:
        return _read_prelude(fh)[0]


def unseal_file(
    envelope: Path,
    shares: Sequence[TrusteeShare],
    output: Path,
    *,
    allow_expired: bool = False,
    now: float | None = None,
) -> SealHeader:
    """Rebuild the escrow key from a quorum of shares and decrypt the original."""
    now = time.time() if now is None else now
    with Path(envelope).open("rb") as fh:
        header, encoded, wrap_nonce, wrapped = _read_prelude(fh)
        if now > header.expires_at and not allow_expired:
            raise SealError("this envelope has expired and must be deleted, not opened")
        private = _rebuild_private(shares, header.key_id)
        eph_pub = _unb64(header.ephemeral_public)
        escrow_pub = private.public_key().public_bytes_raw()
        kek = _kek(
            private.exchange(X25519PublicKey.from_public_bytes(eph_pub)), eph_pub, escrow_pub
        )
        try:
            data_key = AESGCM(kek).decrypt(wrap_nonce, wrapped, encoded)
        except InvalidTag as exc:
            raise SealError("envelope header was altered or the key is wrong") from exc
        aead = AESGCM(data_key)
        header_hash = hashlib.sha256(encoded).digest()
        tmp = output.with_name(output.name + ".part")
        try:
            with tmp.open("wb") as out:
                index, finished = 0, False
                while True:
                    raw_len = fh.read(4)
                    if not raw_len:
                        break
                    if finished:
                        raise SealError("data after the final chunk")
                    if len(raw_len) != 4:
                        raise SealError("envelope is truncated")
                    (n,) = _U32.unpack(raw_len)
                    if n > CHUNK_BYTES + 16:
                        raise SealError("chunk too large")
                    ct = _read_exact(fh, n)
                    for final in (False, True):
                        try:
                            out.write(
                                aead.decrypt(
                                    _chunk_nonce(index), ct, _chunk_aad(header_hash, index, final)
                                )
                            )
                            finished = final
                            break
                        except InvalidTag:
                            continue
                    else:
                        raise SealError(f"chunk {index} was altered")
                    index += 1
                if not finished:
                    raise SealError("envelope is truncated")
            tmp.replace(output)
        finally:
            tmp.unlink(missing_ok=True)
    return header
