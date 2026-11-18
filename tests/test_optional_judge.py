"""Offline regression checks for the core route with optional Part 5.

Run with: python -m unittest discover -s tests -v
Model and LangSmith calls are not made.
"""
import ast
import json
from pathlib import Path
import re
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ('LangSmith_Eval_Lab_Colab.ipynb',)


def cells(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))['cells']


def source(cell):
    return ''.join(cell['source'])


class OptionalJudgeTests(unittest.TestCase):
    def test_part6_quizzes_run_with_no_part5_or_core_runtime_state(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                captured = []
                quiz = ModuleType('jupyterquiz')
                quiz.display_quiz = lambda questions, **kwargs: captured.extend(questions)
                in_part6 = False
                with patch.dict(sys.modules, {'jupyterquiz': quiz}):
                    for cell in cells(name):
                        s = source(cell)
                        if '## Part 6 ·' in s:
                            in_part6 = True
                        if '## Part 7 ·' in s:
                            break
                        if in_part6 and cell['cell_type'] == 'code':
                            exec(s, {})
                self.assertEqual(len(captured), 4)
                self.assertTrue(all('question' in q for q in captured))

    def test_structure_and_shared_instructions(self):
        for name in NOTEBOOKS:
            notebook_cells = cells(name)
            texts = [source(c) for c in notebook_cells if c['cell_type'] == 'markdown']
            joined = '\n'.join(texts)
            self.assertEqual(re.findall(r'<a id="lab-part-(\d+)"></a>', joined), list('01234567'))
            self.assertEqual(joined.count('<br><br>'), 8)
            self.assertIn('Part 5 · Check the explanation and the judge (optional)', joined)
            self.assertIn('Parts 6 and 7 do not require Part 5', joined)
            for target in re.findall(r'\]\(#(lab-part-\d+)\)', joined):
                self.assertIn(f'<a id="{target}"></a>', joined)
            for cell in notebook_cells:
                if cell['cell_type'] == 'code':
                    compile(source(cell), name, 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


if __name__ == '__main__':
    unittest.main()
