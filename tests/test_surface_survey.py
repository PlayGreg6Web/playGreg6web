"""Tests for tools/surface_survey.py (synthetic provider source, survey, graph and trace only)."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import surface_survey as ss  # noqa: E402

PROVIDER = '''
constexpr Guid kAProvider = com::make_guid("e349bd1a-fc20-4e40-b99c-4178cc6b409f");
constexpr Guid kAIid = com::make_guid("1861cf2e-e18b-4834-a9f5-b4a4e6efb4cf");
constexpr Guid kBGuid = com::make_guid("37e56907-2f10-41e8-b72f-36edb185331a");
constexpr Guid kBIid = com::make_guid("bf2346b2-39af-4658-b5ea-44713c7e83b3");
constexpr Guid kCGuid = com::make_guid("bbfbdcc7-bfe7-409b-a5ca-edf054960b4d");
constexpr com::MethodDesc kSystemMethods[] = {
    SY::method<&SystemApi::GetConsoleId>("XSystemGetConsoleId"),
    SY::method<&SystemApi::GetXboxLiveSandboxId>("XSystemGetXboxLiveSandboxId"),
    SY::stub("XSystemGetAppSpecificDeviceId"),
};
constexpr auto kISystem =
    com::declare_interface<SystemApi>("xgameruntime.XSystem", kAIid, kSystemMethods, {.slot_count = 24});
constexpr auto kNetTable = slot_table<15>({
    {3, NW::stub("XNetworkingQueryPort")},
    {15, NW::method<&NetworkingApi::GetHint>("XNetworkingGetConnectivityHint")},
    {16, NW::stub("XNetworkingRegister")},
});
constexpr auto kINet = com::declare_interface<NetworkingApi>("xgameruntime.XNetworking", kBIid,
                                                             kNetTable.slots, {.slot_count = 32});
constexpr auto kIEvent = com::declare_interface<StubApi>(
    "xgameruntime.XGameEvent", kCGuid, {},
    {.slot_count = 16, .policy = com::SlotPolicy::ReturnFixed, .stub_result = 0});
const ApiEntry kApis[] = {
    {kAProvider, kAIid, &kISystem.info, [] { return make<SystemApi>(kISystem); }},
    {kBGuid, kBIid, &kINet.info, [] { return make<NetworkingApi>(kINet); }},
    {kCGuid, kCGuid, &kIEvent.info, [] { return make<StubApi>(kIEvent); }},
};
'''

SYSTEM = ("e349bd1a-fc20-4e40-b99c-4178cc6b409f", "1861cf2e-e18b-4834-a9f5-b4a4e6efb4cf")
NET = ("37e56907-2f10-41e8-b72f-36edb185331a", "bf2346b2-39af-4658-b5ea-44713c7e83b3")
EVENT = ("bbfbdcc7-bfe7-409b-a5ca-edf054960b4d",) * 2


def fn(entry, callees=(), tail=()):
    return {"entry": hex(entry), "callees": [hex(c) for c in callees], "tail_calls": [hex(c) for c in tail]}


def slot(n, site, callers=1):
    return {"slot": n, "sites": 1, "functions": 1, "caller_sites": callers, "min_args": 2,
            "examples": [hex(site)]}


def survey(*keys):
    return {"sources": [{"name": "qapi", "keys": [
        {"key": ["{%s}" % k[0], "{%s}" % k[1]], "slots": slots} for k, slots in keys]}]}


class ProviderTests(unittest.TestCase):
    def test_parse_tables_and_policies(self):
        p = {i["label"]: i for i in ss.parse_provider(PROVIDER)}
        self.assertEqual(len(p), 3)
        system = p["xgameruntime.XSystem"]
        self.assertEqual(ss.slot_status(system, 3), ("implemented", "XSystemGetConsoleId"))
        self.assertEqual(ss.slot_status(system, 5), ("trap", "XSystemGetAppSpecificDeviceId"))
        self.assertEqual(ss.slot_status(system, 6)[0], "trap")  # a gap traps
        net = p["xgameruntime.XNetworking"]
        self.assertEqual(ss.slot_status(net, 3)[0], "trap")
        self.assertEqual(ss.slot_status(net, 15)[0], "implemented")
        self.assertEqual(ss.slot_status(net, 9)[0], "trap")
        ev = p["xgameruntime.XGameEvent"]
        self.assertEqual(ss.slot_status(ev, 4)[0], "fixed")
        self.assertEqual(ss.slot_status(ev, 99)[0], "trap")  # beyond slot_count

    def test_hard_stop_rules(self):
        self.assertIsNotNone(ss.is_hard("xgameruntime.XStore", 20))
        self.assertIsNotNone(ss.is_hard("xgameruntime.XSystem", 5))
        self.assertIsNone(ss.is_hard("xgameruntime.XSystem", 3))
        self.assertIsNone(ss.is_hard("xgameruntime.XNetworking", 3))


class GraphTests(unittest.TestCase):
    def setUp(self):
        # 1000 -> 2000 -> 4000, 1000 tail-> 3000; 7000 -> 5000 -> 6000 (reached only via a trace edge)
        self.g = ss.Graph([fn(0x1000, [0x2000], [0x3000]), fn(0x2000, [0x4000]), fn(0x3000), fn(0x4000),
                           fn(0x5000, [0x6000]), fn(0x6000), fn(0x7000, [0x5000])])

    def test_forward_depths_and_function_of(self):
        d = self.g.forward_depths(0x1000)
        self.assertEqual(d, {0x1000: 0, 0x2000: 1, 0x3000: 1, 0x4000: 2})
        self.assertEqual(self.g.function_of(0x2fff), 0x2000)
        self.assertIsNone(self.g.function_of(0x500))

    def test_trace_adds_edge_and_executed(self):
        # return address 0x3010 lies inside 0x3000, which dispatched to 0x5000
        self.assertEqual(self.g.add_trace([(0x3010, 0x5000)]), 1)
        self.assertIn(0x6000, self.g.forward_depths(0x1000))
        self.assertEqual({0x3000, 0x5000}, self.g.executed)

    def test_ancestor_hops(self):
        self.g.executed.add(0x7000)
        self.assertEqual(self.g.ancestor_hops(0x6000, 6), 2)
        self.assertIsNone(self.g.ancestor_hops(0x6000, 1))
        self.assertIsNone(self.g.ancestor_hops(0x4000, 6))

    def test_parse_targets(self):
        self.assertEqual(ss.parse_targets("# c\n0x10 0x20\nbad\n30 40 # x\n"), [(0x10, 0x20), (0x30, 0x40)])


class BuildTests(unittest.TestCase):
    def test_ranking_tiers_and_exclusions(self):
        provider = ss.parse_provider(PROVIDER)
        g = ss.Graph([fn(0x1000, [0x2000]), fn(0x2000), fn(0x3000), fn(0x4000, [0x3000]), fn(0x9000)])
        g.executed.add(0x4000)
        sv = survey(
            (SYSTEM, [slot(2, 0x2010), slot(3, 0x2010), slot(5, 0x2020)]),
            (NET, [slot(3, 0x2030, callers=1), slot(16, 0x3010, callers=5), slot(9, 0x9010, callers=9),
                   slot(15, 0x2040)]),
            (EVENT, [slot(4, 0x2050)]),
            (("00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"), [slot(3, 0x2060)]),
        )
        res = ss.build_survey(provider, sv, g, 0x1000, ("xgameruntime.XSystem", 5), 6, None)
        names = [(r["name"], r["tier"]) for r in res["ranked"]]
        # tier A first, then B (0x3000 is a callee of an executed function), then C
        self.assertEqual(names, [("XNetworkingQueryPort", "A"), ("XNetworkingRegister", "B"), ("slot 9", "C")])
        self.assertEqual(res["current_stop"]["name"], "XSystemGetAppSpecificDeviceId")
        self.assertEqual(res["by_status"], {"implemented": 2, "trap": 4, "fixed": 1})
        self.assertEqual(len(res["unregistered_interface_slots"]), 1)

    def test_hard_stops_listed_separately_and_no_addresses_in_default_json(self):
        provider = ss.parse_provider(PROVIDER.replace("xgameruntime.XNetworking", "xgameruntime.XStore"))
        g = ss.Graph([fn(0x1000, [0x2000]), fn(0x2000)])
        sv = survey((NET, [slot(3, 0x2010)]))
        res = ss.build_survey(provider, sv, g, 0x1000, ("xgameruntime.XSystem", 5), 6, None)
        self.assertEqual(res["ranked"], [])
        self.assertEqual([r["name"] for r in res["hard_stops"]], ["XNetworkingQueryPort"])
        text = ss.to_json(res, with_addresses=False)
        self.assertNotIn("0x2010", text)
        self.assertNotIn("call_site_addresses", text)
        self.assertIn("0x2010", ss.to_json(res, with_addresses=True))
        self.assertIn("## Hard stops", ss.to_markdown(res, 10))


class ImportTests(unittest.TestCase):
    def test_unresolved_import_sites_found_through_iat_slot(self):
        # one executable section at rva 0x1000 holding `call [rip+disp]` to an IAT slot at rva 0x3000
        code = bytearray(0x20)
        code[4:6] = b"\xff\x15"
        code[6:10] = struct.pack("<i", 0x3000 - (0x1000 + 4 + 6))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.bin"
            p.write_bytes(bytes(code))
            layout = {"image_base": 0x140000000, "sections": [
                {"name": ".text", "va": 0x1000, "vsize": 0x20, "raw": 0, "rsize": 0x20, "exec": True}]}
            found = ss.find_iat_sites(p, layout, {0x3000: "a.dll!F"})
        self.assertEqual(found, {"a.dll!F": [0x140001004]})

    def test_summarise_imports(self):
        manifest = {"modules": [{"file": "t.exe", "role": "exe"}], "imports": [
            {"importer": "t.exe", "dll": "a.dll", "name": "F", "ordinal": None, "slot_rva": "0x3000",
             "binding": "unresolved", "reason": "denied"},
            {"importer": "t.exe", "dll": "a.dll", "name": None, "ordinal": 7, "slot_rva": "0x3008",
             "binding": "unresolved", "reason": "unmarshalable"},
            {"importer": "other.dll", "dll": "b.dll", "name": "G", "ordinal": None, "slot_rva": "0x10",
             "binding": "unresolved", "reason": "no_shim"},
            {"importer": "t.exe", "dll": "a.dll", "name": "H", "ordinal": None, "slot_rva": "0x3010",
             "binding": "host", "reason": None}]}
        s = ss.summarise_imports(manifest, None, None, None, {}, 6)
        self.assertEqual(s["by_binding"], {"unresolved": 3, "host": 1})
        self.assertEqual([r["import"] for r in s["unresolved_in_main_module"]], ["a.dll!#7", "a.dll!F"])


if __name__ == "__main__":
    unittest.main()
