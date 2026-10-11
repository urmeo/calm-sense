import unittest
from contextlib import ExitStack
from unittest.mock import patch
from tempfile import TemporaryDirectory
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
PHANTOM_MODELS = [
    "Transformer",
    "BiLSTM",
    "CatBoost",
    "EfficientNet",
    "cross-modal attention",
]


class DocsTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_readme_names_no_untrained_models(self):
        lower = README.lower()
        found = [name for name in PHANTOM_MODELS if name.lower() in lower]
        self.assertFalse(
            found, f"README names models the project never trains: {found}"
        )

    def test_docs_make_no_best_overall_ranking_claim(self):
        self.assertNotIn(
            "best overall",
            README.lower(),
            "README makes a 'Best overall' ranking claim",
        )

    def test_readme_local_links_and_images_exist(self):
        refs = re.findall(r"\]\(([^\s)]+)\)", README) + re.findall(
            '(?:src|href)="([^"]+)"', README
        )
        for ref in refs:
            url = urlsplit(ref)
            if url.scheme or not url.path:
                continue
            self.assertTrue(
                (ROOT / unquote(url.path)).exists(), f"Missing README asset: {ref}"
            )

    def test_readme_table_update_is_idempotent(self):
        tmp_path = self.tmp_path
        from scripts import update_readme_tables as tables

        path = tmp_path / "README.md"
        path.write_text(README)
        self.stack.enter_context(patch.object(tables, "README", path))
        tables.main()
        self.assertEqual(path.read_text(), README)
        tables.main()
        self.assertEqual(path.read_text(), README)

    def test_readme_table_update_rejects_missing_markers(self):
        from scripts.update_readme_tables import _replace

        with self.assertRaisesRegex(SystemExit, "not found in README.md"):
            _replace("overview without generated tables", "calibration", "replacement")

    def test_referenced_notebooks_exist(self):
        refs = sorted(set(re.findall(r"notebooks/[\w./-]+\.ipynb", README)))
        missing = [ref for ref in refs if not (ROOT / ref).exists()]
        self.assertFalse(
            missing, f"Docs reference notebooks that do not exist: {missing}"
        )

    def test_no_nonexistent_numbered_notebook_series(self):
        self.assertFalse(
            re.search("notebooks/0[1-8]_", README),
            "README references a nonexistent numbered 01-08 notebook series",
        )
