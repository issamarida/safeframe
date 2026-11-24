import json
import stat

from safeframe.cli import main
from safeframe.seal import EscrowPublicKey, seal_file


def test_band_sheet(tmp_path):
    out = tmp_path / "band.svg"
    assert main(["band", "-o", str(out)]) == 0
    svg = out.read_text()
    assert "<svg" in svg[:200] and "SafeFrame Band" in svg


def test_key_ceremony_seal_and_unseal(tmp_path, capsys):
    keys = tmp_path / "keys"
    assert main(["trustees", "init", "-k", "2", "-n", "3", "--out", str(keys)]) == 0
    shares = sorted(keys.glob("trustee-*.json"))
    assert len(shares) == 3
    assert all(stat.S_IMODE(s.stat().st_mode) == 0o600 for s in shares)
    assert json.loads(shares[0].read_text())["threshold"] == 2

    original = tmp_path / "IMG_0001.jpg"
    original.write_bytes(b"\xff\xd8 not really a photo")
    envelope = tmp_path / "IMG_0001.sfseal"
    public = EscrowPublicKey.from_json((keys / "escrow.pub.json").read_text())
    seal_file(original, envelope, public, reason="test")

    restored = tmp_path / "restored.jpg"
    # One share is refused with a one-line error, not a traceback.
    assert main(["unseal", str(envelope), "--share", str(shares[0]), "-o", str(restored)]) == 2
    assert capsys.readouterr().err.startswith("error:")
    assert not restored.exists()
    # Shares can follow a single --share flag.
    argv = ["unseal", str(envelope), "--share", str(shares[2]), str(shares[0]), "-o", str(restored)]
    assert main(argv) == 0
    assert restored.read_bytes() == original.read_bytes()


def test_bad_location_is_a_clean_error(tmp_path, capsys):
    photo = tmp_path / "x.jpg"
    photo.write_bytes(b"\xff\xd8")
    code = main(["analyze", str(photo), "-o", str(tmp_path / "a.json"), "--location", "north"])
    assert code == 2
    assert "error:" in capsys.readouterr().err
