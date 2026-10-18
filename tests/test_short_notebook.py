"""Keep shortened notebooks runnable without the removed baseline repeat."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['LangSmith_Eval_Lab_Colab.ipynb']

class ShortNotebookTests(unittest.TestCase):
    def test_short_route_has_no_repeat_dependencies(self):
        for name in NAMES:
            short = json.loads((ROOT / name).read_text(encoding='utf8'))
            for c in short['cells']:
                s=''.join(c['source'])
                self.assertNotIn('baseline rerun',s)
                if c['cell_type']=='code':
                    compile(s,name,'exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    names={n.id for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Name)}
                    self.assertFalse(names & {'rerun','repeat_delta','RERUN_CHANGED_ROWS','_repeat_table','SECOND_TAG','SECOND_VERSION_COUNT','precision'})
                    if 'METRICS = [' in s:
                        state=dict(baseline=SimpleNamespace(metrics={'recall_exploitability':.5,'precision_exploitability':.8}),
                            mine=SimpleNamespace(metrics={'recall_exploitability':.75,'precision_exploitability':.6}),
                            R=SimpleNamespace(changes=lambda *args:dict(fixed=['a'],broken=['b'])),
                            PREDICTION='recall up',PREDICTION_REASON='Flag uncertain findings.',
                            viz=SimpleNamespace(metric_card=lambda reports:None),prompts=[])
                        out=io.StringIO()
                        with contextlib.redirect_stdout(out):exec('METRICS = ['+s.split('METRICS = [',1)[1],state)
                        self.assertIn('+0.250 vs baseline',out.getvalue())
                        self.assertIn('1 fixed, 1 broken',out.getvalue())

if __name__=='__main__':unittest.main()
