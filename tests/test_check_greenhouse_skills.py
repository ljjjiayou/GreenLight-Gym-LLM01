import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_greenhouse_skills.py"
SPEC = importlib.util.spec_from_file_location("check_greenhouse_skills", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _write_skill(root: Path, name: str, body: str | None = None) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    text = body or (
        "---\n"
        f"name: {name}\n"
        "description: Test greenhouse skill.\n"
        "---\n"
        "\n"
        "# Test Skill\n"
        "\n"
        "Use this skill for tests.\n"
    )
    (skill_dir / "SKILL.md").write_bytes(text.encode("utf-8"))


class TestCheckGreenhouseSkills(unittest.TestCase):
    def test_matching_repo_and_local_skills_are_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            local = root / "local"
            _write_skill(repo, "greenhouse-mainline-guard")
            _write_skill(
                local,
                "greenhouse-mainline-guard",
                "---\r\n"
                "name: greenhouse-mainline-guard\r\n"
                "description: Test greenhouse skill.\r\n"
                "---\r\n"
                "\r\n"
                "# Test Skill\r\n"
                "\r\n"
                "Use this skill for tests.\r\n",
            )

            report = module.build_report(repo, local)

        self.assertTrue(report["ok"])
        self.assertEqual(report["out_of_sync"], [])
        self.assertEqual(report["skill_count"], 1)

    def test_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            local = root / "local"
            _write_skill(repo, "greenhouse-mainline-guard")
            _write_skill(
                local,
                "greenhouse-mainline-guard",
                "---\n"
                "name: greenhouse-mainline-guard\n"
                "description: Changed description.\n"
                "---\n"
                "\n"
                "# Different\n",
            )

            report = module.build_report(repo, local)

        self.assertFalse(report["ok"])
        self.assertEqual(report["out_of_sync"], ["greenhouse-mainline-guard"])

    def test_invalid_frontmatter_blocks_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            local = root / "local"
            _write_skill(
                repo,
                "greenhouse-mainline-guard",
                "---\n"
                "name: wrong-name\n"
                "---\n"
                "\n"
                "# Broken\n",
            )
            _write_skill(local, "greenhouse-mainline-guard")

            report = module.build_report(repo, local)

        self.assertFalse(report["ok"])
        self.assertIn("frontmatter_name_mismatch", report["rows"][0]["repo"]["errors"])
        self.assertIn("missing_description", report["rows"][0]["repo"]["errors"])


if __name__ == "__main__":
    unittest.main()
