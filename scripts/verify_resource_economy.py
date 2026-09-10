"""Resource economy teaching (Sep 7): the generalized producer map.

RESOURCE_ECONOMY unifies producers the way CRAVING_PRODUCERS unified
demand: every tap-cost resource maps to the stations spending it, the
families generating it, and the station raising its cap. Checks:
- map integrity (both resources; used_by stations are known station
  names; made_by/cap_by non-empty),
- the correction + folded-rule texts name the producers (no drift
  between the structured map and the taught prose),
- wiki Makes rates are banked in item_stats and survive popup writes.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = []
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def main() -> int:
    from planner.vision_drive import (RESOURCE_ECONOMY, RESOURCE_REJECTION,
                                      VisionDrivenPlanner, _resource_teaching)
    from planner.glossary import read_item_stats, write_item_stat

    # A1: all tap-cost resources present with full triples.
    for res in RESOURCE_ECONOMY:
        cfg = RESOURCE_ECONOMY.get(res, {})
        check(f"A1: {res} has used_by/made_by/cap_by",
              bool(cfg.get("used_by")) and bool(cfg.get("made_by"))
              and bool(cfg.get("cap_by")), str(sorted(cfg)))
    # A2: used_by stations are known station names (alias-normalized).
    from planner.vision_drive import _STATION_NAME_ALIASES
    for res, cfg in RESOURCE_ECONOMY.items():
        for st in cfg["used_by"]:
            norm = st.lower().replace(" ", "")
            check(f"A2: {res} used_by {st!r} is a known station",
                  norm in _STATION_NAME_ALIASES, norm)
    # A3: rejection reasons point at known resources.
    for reason, res in RESOURCE_REJECTION.items():
        check(f"A3: {reason} -> known resource {res}",
              res in RESOURCE_ECONOMY, res)
    # A4: teaching sentences name the board-relevant producers.
    slime_t = _resource_teaching("slime")
    check("A4: slime teaching names Zombies/Spiders/Ghouls",
          all(w in slime_t for w in ("Zombies", "Spiders", "Ghouls")),
          slime_t[:80])
    check("A4: slime teaching names the cap station",
          "Slime Vats" in slime_t)
    mana_t = _resource_teaching("mana")
    check("A4: mana teaching names Skeletons/Eye Monsters/Banshees",
          all(w in mana_t for w in ("Skeletons", "Eye Monsters", "Banshees")),
          mana_t[:80])
    check("A4: short form is one sentence",
          _resource_teaching("slime", short=True).count(".") == 1)
    # A4b: darkness (late game, knowledge-only) teaches producers too.
    dark_t = _resource_teaching("darkness")
    check("A4b: darkness teaching names Mummies/Bats/Imps",
          all(w in dark_t for w in ("Mummies", "Bats", "Imps")),
          dark_t[:80])
    check("A4b: darkness teaching names the cap station",
          "Darkness Stores" in dark_t)
    # A5: in-step correction teaches producers on resource rejections.
    c = VisionDrivenPlanner._correction("spawn_no_slime:0")
    check("A5: correction names slime producers",
          "Zombies" in c and "Spiders" in c, c[:100])
    c = VisionDrivenPlanner._correction("spawn_low_mana:0.10")
    check("A5: correction names mana producers",
          "Skeletons" in c and "Eye Monsters" in c, c[:100])
    # A6: folded learnings rules teach producers too.
    rules = VisionDrivenPlanner._REJECTION_RULE_TEXTS
    check("A6: folded slime rule names producers",
          "Zombies" in rules["spawn_no_slime"])
    check("A6: folded mana rule names producers",
          "Skeletons" in rules["spawn_low_mana"])
    # A7: wiki Makes rates banked (spot-checks from the Mana/Slime tables).
    stats = read_item_stats()
    check("A7: eyemonster_lvl1 makes 2 mana",
          (stats.get("eyemonster_lvl1", {}).get("makes") or {}).get("mana") == 2,
          str(stats.get("eyemonster_lvl1", {}).get("makes")))
    check("A7: skeleton_lvl5 makes 7 mana",
          (stats.get("skeleton_lvl5", {}).get("makes") or {}).get("mana") == 7)
    check("A7: zombie_lvl3 makes 4 slime",
          (stats.get("zombie_lvl3", {}).get("makes") or {}).get("slime") == 4)
    check("A7: mummy_lvl1 makes 1 darkness",
          (stats.get("mummy_lvl1", {}).get("makes") or {}).get("darkness") == 1)
    check("A7: bat_lvl4 makes 8 darkness",
          (stats.get("bat_lvl4", {}).get("makes") or {}).get("darkness") == 8)
    check("A7: imp_lvl2 makes 5 darkness",
          (stats.get("imp_lvl2", {}).get("makes") or {}).get("darkness") == 5)
    # A8: popup writes preserve wiki makes (isolated knowledge dir).
    with tempfile.TemporaryDirectory() as td:
        kd = Path(td)
        (kd).mkdir(exist_ok=True)
        write_item_stat("test_mon", feed=10, knowledge_dir=kd)
        import json
        p = kd / "item_stats.json"
        d = json.loads(p.read_text())
        d["test_mon"]["makes"] = {"mana": 3}
        p.write_text(json.dumps(d))
        write_item_stat("test_mon", feed=12, knowledge_dir=kd)
        d = json.loads(p.read_text())
        check("A8: popup write preserves makes",
              d["test_mon"].get("makes") == {"mana": 3}
              and d["test_mon"]["feed"] == 12,
              str(d["test_mon"]))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
