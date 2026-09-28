"""Phase 5B-1 freeze guard: the files recorded in the experiment manifest
must not change during the experiment. Any modification fails CI."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_phase5b1_manifest import sha256_file  # noqa: E402


class TestPhase5b1Freeze(unittest.TestCase):
    def test_frozen_inputs_match_manifest(self):
        mp = ROOT / "experiments" / "phase5b1" / "manifest.json"
        if not mp.exists():
            self.skipTest("manifest not generated yet")
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        frozen = manifest.get("frozen_inputs", {})
        self.assertTrue(frozen, "manifest has no frozen inputs")
        for rel, rec in frozen.items():
            p = ROOT / rel
            self.assertTrue(p.exists(), rel)
            self.assertEqual(
                sha256_file(p), rec["sha256"],
                f"FROZEN INPUT CHANGED: {rel} — Phase 5B-1 forbids edits; "
                f"record an issue for Phase 5B-2 instead")


if __name__ == "__main__":
    unittest.main()
