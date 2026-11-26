import json
import os
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import saves  # noqa: E402


def make(root, cfg, cname, blobs, display=None):
    d = os.path.join(root, saves.encode(cfg), "local", "c-" + saves.encode(cname))
    os.makedirs(d)
    for n, data in blobs.items():
        with open(os.path.join(d, "b-" + saves.encode(n)), "wb") as f:
            f.write(data)
    if display:
        with open(os.path.join(d, ".display-name"), "wb") as f:
            f.write(display.encode())


class Saves(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a = os.path.join(self.tmp.name, "a")
        self.b = os.path.join(self.tmp.name, "b")
        self.zip = os.path.join(self.tmp.name, "s.zip")
        make(self.a, "TitleCfg", "Slot1", {"Data": b"\x00\x01" * 100, "meta": b"m"}, "My Save")
        make(self.a, "TitleCfg", "slot2", {"x": b"y"})
        # files the tool must never export
        with open(os.path.join(self.a, "TitleCfg.txt"), "w") as f:
            f.write("nope")
        with open(os.path.join(self.a, saves.encode("TitleCfg"), "local", "c-slot2", ".tmp-1"), "w") as f:
            f.write("tmp")

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *a):
        return saves.main(list(a))

    def test_encode_roundtrip(self):
        for n in ("a", "Slot 1", "ÄÖ/x", "A^b%"):
            self.assertEqual(saves.decode(saves.encode(n)), n)
        self.assertIsNone(saves.decode("A"))
        self.assertIsNone(saves.decode("%zz"))

    def test_export_import_roundtrip(self):
        self.assertEqual(self.run_main("export", self.zip, "--root", self.a), 0)
        with zipfile.ZipFile(self.zip) as z:
            names = z.namelist()
            man = json.loads(z.read("manifest.json"))
        self.assertEqual(man["format"], "pg6-saves")
        self.assertEqual(man["format_version"], 1)
        self.assertEqual(man["titles"][0]["config_id"], "TitleCfg")
        self.assertTrue(all(n == "manifest.json" or n.startswith("saves/") for n in names))
        self.assertFalse(any(".tmp" in n or n.endswith(".txt") for n in names))
        self.assertEqual(self.run_main("import", self.zip, "--root", self.b), 0)
        got = saves.scan(self.b)[0]["containers"]
        self.assertEqual([c["name"] for c in got], ["Slot1", "slot2"])
        self.assertEqual(got[0]["display_name"], "My Save")
        with open(got[0]["blobs"][0]["path"], "rb") as f:
            self.assertEqual(f.read(), b"\x00\x01" * 100)
        local = os.path.join(self.b, saves.encode("TitleCfg"), "local")
        self.assertEqual([f for f in os.listdir(local) if f.startswith(".")], [])

    def test_no_overwrite_without_force(self):
        self.run_main("export", self.zip, "--root", self.a)
        make(self.b, "TitleCfg", "Slot1", {"old": b"keep"})
        self.assertEqual(self.run_main("import", self.zip, "--root", self.b), 1)
        containers = saves.scan(self.b)[0]["containers"]
        self.assertEqual(len(containers), 1)  # nothing partially imported
        self.assertEqual([b["name"] for b in containers[0]["blobs"]], ["old"])
        self.assertEqual(self.run_main("import", self.zip, "--root", self.b, "--force"), 0)
        names = {c["name"]: [b["name"] for b in c["blobs"]] for c in saves.scan(self.b)[0]["containers"]}
        self.assertEqual(names["Slot1"], ["Data", "meta"])

    def test_dry_run_writes_nothing(self):
        self.run_main("export", self.zip, "--root", self.a)
        self.assertEqual(self.run_main("import", self.zip, "--root", self.b, "--dry-run"), 0)
        self.assertFalse(os.path.exists(self.b))

    def rewrite(self, mutate):
        self.run_main("export", self.zip, "--root", self.a)
        out = self.zip + ".bad"
        with zipfile.ZipFile(self.zip) as zin:
            items = {n: zin.read(n) for n in zin.namelist()}
        mutate(items)
        with zipfile.ZipFile(out, "w") as zout:
            for n, d in items.items():
                zout.writestr(n, d)
        return out

    def assert_rejected(self, mutate):
        bad = self.rewrite(mutate)
        self.assertEqual(self.run_main("import", bad, "--root", self.b), 1)
        self.assertFalse(os.path.exists(self.b))

    def test_rejects_corrupt_blob(self):
        def m(i):
            k = [n for n in i if n.endswith("b-^data")][0]
            i[k] = b"\x00" * len(i[k])
        self.assert_rejected(m)

    def test_rejects_extra_entry_and_traversal(self):
        self.assert_rejected(lambda i: i.update({"saves/x/evil.exe": b"x"}))
        self.assert_rejected(lambda i: i.update({"../evil": b"x"}))

    def test_rejects_bad_manifest(self):
        def ver(i):
            m = json.loads(i["manifest.json"])
            m["format_version"] = 99
            i["manifest.json"] = json.dumps(m).encode()

        def name(i):
            m = json.loads(i["manifest.json"])
            m["titles"][0]["containers"][0]["name"] = "../x"
            i["manifest.json"] = json.dumps(m).encode()
        self.assert_rejected(ver)
        self.assert_rejected(name)
        self.assert_rejected(lambda i: i.pop("manifest.json"))

    def test_rejects_oversized_blob(self):
        def m(i):
            k = [n for n in i if "/b-" in n][0]
            i[k] = b"0" * (saves.MAX_BLOB_BYTES + 1)
        self.assert_rejected(m)

    def test_config_id_filter(self):
        make(self.a, "Other", "s", {"b": b"1"})
        self.run_main("export", self.zip, "--root", self.a, "--config-id", "Other")
        self.run_main("import", self.zip, "--root", self.b)
        self.assertEqual([t["config_id"] for t in saves.scan(self.b)], ["Other"])

    def test_export_refuses_existing_zip(self):
        self.run_main("export", self.zip, "--root", self.a)
        self.assertEqual(self.run_main("export", self.zip, "--root", self.a), 1)

    def test_list(self):
        self.assertEqual(self.run_main("list", "--root", self.a), 0)


if __name__ == "__main__":
    unittest.main()
