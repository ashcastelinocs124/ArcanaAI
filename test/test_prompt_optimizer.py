import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ai import prompt_optimizer


class TestPromptOptimizer(unittest.TestCase):
    def _write_excel(self, path: Path, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        df.to_excel(path, index=False)

    def test_load_excel_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            excel_path = Path(tmp) / "data.xlsx"
            self._write_excel(excel_path, [{"input": "hi"}])
            with self.assertRaises(ValueError):
                prompt_optimizer.load_excel(excel_path, "input", "gold", None)

    def test_load_excel_resolves_regex_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            excel_path = Path(tmp) / "data.xlsx"
            self._write_excel(
                excel_path,
                [{"Input Text": "hi", "Expected Output": "ok", "Comments": "c"}],
            )
            _, input_col, gold_col, comments_col = prompt_optimizer.load_excel(
                excel_path, "input", "expected", "comment"
            )
            self.assertEqual(input_col, "Input Text")
            self.assertEqual(gold_col, "Expected Output")
            self.assertEqual(comments_col, "Comments")

    def test_optimize_row_improves_prompt(self):
        def fake_call(prompt: str, model: str) -> str:
            if prompt.startswith("You are optimizing"):
                return "Better prompt: {input}"
            if "Better prompt" in prompt:
                return "EXPECTED"
            return "WRONG"

        with patch("ai.prompt_optimizer._call_llm", side_effect=fake_call):
            final_template, output, score, iters = prompt_optimizer.optimize_row(
                prompt_template="Base prompt: {input}",
                input_text="Hello",
                gold_text="EXPECTED",
                eval_model="fake",
                optimizer_model="fake",
                target_score=0.9,
                max_iters=3,
            )

        self.assertEqual(final_template, "Better prompt: {input}")
        self.assertEqual(output, "EXPECTED")
        self.assertEqual(score, 1.0)
        self.assertEqual(iters, 2)

    def test_run_optimizer_writes_outputs(self):
        def fake_call(prompt: str, model: str) -> str:
            return "GOLD"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            excel_path = tmp_path / "data.xlsx"
            learning_path = tmp_path / "learning.md"
            output_path = tmp_path / "out.csv"
            self._write_excel(
                excel_path,
                [{"input": "x", "gold": "GOLD", "comments": "ok"}],
            )

            with patch("ai.prompt_optimizer._call_llm", side_effect=fake_call):
                results = prompt_optimizer.run_optimizer(
                    excel_path=excel_path,
                    prompt_template="Prompt: {input}",
                    input_col="input",
                    gold_col="gold",
                    comments_col="comments",
                    eval_model="fake",
                    optimizer_model="fake",
                    target_score=0.9,
                    max_iters=1,
                    learning_path=learning_path,
                    output_path=output_path,
                )

            self.assertTrue(learning_path.exists())
            self.assertTrue(output_path.exists())
            self.assertEqual(len(results), 1)
            self.assertIn("Run", learning_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
