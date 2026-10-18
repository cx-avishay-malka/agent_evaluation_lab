"""Offline regression checks for the core route with optional Part 5.

Run with: python -m unittest discover -s tests -v
Model and LangSmith calls are not made; reports use small representative results.
"""
import ast
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ('LangSmith_Eval_Lab_Colab.ipynb', 'LangSmith_Eval_Lab_Colab_solutions.ipynb')
FULL_NOTEBOOKS = tuple(name.replace('.ipynb', '_full.ipynb') for name in NOTEBOOKS)


def cells(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))['cells']


def source(cell):
    return ''.join(cell['source'])


def report_code(name):
    return next(source(c).split('# Validation below. You do not need to edit this part.\n', 1)[1]
                for c in cells(name) if c['cell_type'] == 'code' and 'REPORT = [' in source(c))


def core_state(name):
    # Use the notebook's actual input validator, without running service setup.
    setup = next(source(c) for c in cells(name) if 'def require(' in source(c))
    require_node = next(n for n in ast.parse(setup).body if isinstance(n, ast.FunctionDef) and n.name == 'require')
    state = {}
    exec(compile(ast.Module(body=[require_node], type_ignores=[]), '<require>', 'exec'), state)
    state.update(
        pd=pd, display=lambda value: None,
        COSTLY_MISTAKE='A missed vulnerability may be dismissed without review.',
        NOT_PROVEN='Production readiness.', NEXT_TEST='Test on new real findings.',
        UNSUPPORTED_NOTE='', JUDGE_LIMITATION='',
        D=SimpleNamespace(split_summary=lambda rows: {'dev': {'a': 13, 'b': 13}, 'held-out': {'a': 31, 'b': 31}}),
        R=SimpleNamespace(reply_table=lambda report: pd.DataFrame({'said unknown': [1, 0], 'no verdict': [0, 1]})),
        rows_v1=[], settings=SimpleNamespace(triage_model='test-model'), DATASET='test-data', DATASET_TAG='v1',
        CHOSEN='baseline_v1', WHY_CHOSEN='Fewer misses on dev.',
        held_out=SimpleNamespace(metrics={'recall_exploitability': .8, 'precision_exploitability': .75, 'fn': 6, 'fp': 8},
                                 dataset_snapshot={'examples_sha256': 'test-hash'}),
    )
    return state


class OptionalJudgeTests(unittest.TestCase):
    def test_core_report_without_any_part5_variables(self):
        for name in FULL_NOTEBOOKS:
            with self.subTest(notebook=name):
                state = core_state(name)
                exec(report_code(name), state)
                report = dict(state['REPORT'])
                self.assertIn('Not evaluated', report['Judge test recall, precision'])
                self.assertIn('Not evaluated', report['An unsupported explanation'])
                self.assertEqual(report['Held-out missed, false alarms'], '6, 8')

    def test_development_only_does_not_invent_a_judge_test(self):
        for name in FULL_NOTEBOOKS:
            with self.subTest(notebook=name):
                state = core_state(name)
                state.update(cal={'judge_recall_ungrounded': 1.0}, judge=object())
                exec(report_code(name), state)
                self.assertIn('Not evaluated', dict(state['REPORT'])['Judge test recall, precision'])

    def test_completed_judge_includes_results_and_failure_count(self):
        for name in FULL_NOTEBOOKS:
            with self.subTest(notebook=name):
                state = core_state(name)
                state.update(RIGHT_VERDICT_BAD_REASON='cal-033', UNSUPPORTED_NOTE='Invented function.',
                             JUDGE_LIMITATION='Small test and one failed call.',
                             judge_test={'judge_recall_ungrounded': .6, 'judge_precision_ungrounded': .75,
                                         'judge_failed_calls': 1})
                exec(report_code(name), state)
                report = dict(state['REPORT'])
                self.assertEqual(report['Judge test recall, precision'], '0.60, 0.75')
                self.assertIn('cal-033', report['An unsupported explanation'])
                self.assertEqual(report['Judge test failed calls (excluded from quality scores)'], '1')
                state['JUDGE_LIMITATION'] = ''
                with self.assertRaisesRegex(ValueError, 'JUDGE_LIMITATION'):
                    exec(report_code(name), state)

    def test_unfinished_core_directs_to_manual_report(self):
        for name in FULL_NOTEBOOKS:
            with self.subTest(notebook=name):
                state = core_state(name)
                del state['held_out']
                with self.assertRaisesRegex(ValueError, 'manual report'):
                    exec(report_code(name), state)

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
                self.assertEqual(len(captured), 3)
                self.assertTrue(all('question' in q for q in captured))

    def test_structure_and_shared_instructions(self):
        markdown = []
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
            markdown.append([t for t in texts if 'lab-part-' in t and '## Part' in t])
            for cell in notebook_cells:
                if cell['cell_type'] == 'code':
                    compile(source(cell), name, 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        self.assertEqual(markdown[0], markdown[1])
        self.assertEqual(report_code(FULL_NOTEBOOKS[0]), report_code(FULL_NOTEBOOKS[1]))


if __name__ == '__main__':
    unittest.main()
