"""Offline checks for feedback exercises; no model or LangSmith calls."""
import ast
import base64
import contextlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class FeedbackTests(unittest.TestCase):
    def test_exercises(self):
        for path in [ROOT / 'LangSmith_Eval_Lab_Colab.ipynb']:
            texts = [''.join(c['source']) for c in json.loads(path.read_text(encoding='utf8'))['cells']]
            setup = next(s for s in texts if 'def require(' in s)
            funcs = [n for n in ast.parse(setup).body if isinstance(n, ast.FunctionDef) and n.name in ('check', 'require')]
            state = {}
            exec(compile(ast.Module(body=funcs,type_ignores=[]),'helpers','exec'), state)
            payload = next(s for s in texts if '_PAYLOAD = (' in s)
            encoded = next(ast.literal_eval(n.value) for n in ast.parse(payload).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_PAYLOAD' for t in n.targets))
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)),mode='r:gz') as archive:
                report = archive.extractfile('src/lab/report.py').read().decode()
            funcs = [n for n in ast.parse(report).body if isinstance(n,ast.FunctionDef) and n.name in ('flips','changes')]
            rs = {'pd':pd,'EvalReport':object}
            exec(compile(ast.Module(body=funcs,type_ignores=[]),'report','exec'), rs)
            def row(id, predicted, expected, outcome):
                return SimpleNamespace(finding_id=id,predicted=predicted,expected=expected,outcome=outcome,repetition=0,pattern='test')
            baseline=SimpleNamespace(records=[row('a','unknown','exploitable','FN'),row('b','not exploitable','not exploitable','TN'),row('c','unknown','exploitable','FN')])
            after=SimpleNamespace(records=[row('a','exploitable','exploitable','TP'),row('b','exploitable','not exploitable','FP'),row('c','not exploitable','exploitable','FN')])
            state.update(R=SimpleNamespace(flips=rs['flips'],changes=rs['changes']),baseline=baseline,recall=after,rerun=after)
            def validation(marker):
                return next(s for s in texts if marker in s).split('# Validation below. You do not need to edit this part.\n')[1]
            def run(code):
                out=io.StringIO()
                with contextlib.redirect_stdout(out): exec(code,state)
                return out.getvalue()
            state.update(FN_TO_TP=' A ',TN_TO_FP='b')
            self.assertEqual(run(validation('FN_TO_TP =')).count('✅'),2)
            state['recall']=baseline
            state.update(FN_TO_TP=' NONE ',TN_TO_FP='none')
            self.assertEqual(run(validation('FN_TO_TP =')).count('✅'),2)
            state['recall']=after
            state.update(DISMISS_ACCURACY=.9,DISMISS_RECALL=0)
            self.assertEqual(run(validation('DISMISS_ACCURACY =')).count('✅'),2)
            state.update(DISMISS_ACCURACY=90,DISMISS_RECALL=None)
            self.assertEqual(run(validation('DISMISS_ACCURACY =')).count('❌'),2)

if __name__=='__main__': unittest.main()
