"""Execute the shortened upload and experiment path with local service doubles."""
import ast
import asyncio
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ShortFlowTests(unittest.TestCase):
    def test_two_versions_and_three_prompts(self):
        for name in ['LangSmith_Eval_Lab_Colab.ipynb']:
            cells=json.loads((ROOT/name).read_text(encoding='utf8'))['cells']
            codes=[''.join(c['source']) for c in cells if c['cell_type']=='code']
            tags={}; calls=[]; rescored=[]
            def upload(rows, **kw):
                tags[kw['version_tag']]=list(rows)
                return SimpleNamespace(name=kw['name'],url='local',example_count=len(rows))
            async def run_eval(spec, **kw):
                calls.append((spec,kw))
                return SimpleNamespace(spec=spec,metrics={'fn':2,'fp':3,'recall_exploitability':.8,'precision_exploitability':.7,'mcc_exploitability':.5},
                    records=[SimpleNamespace(finding_id='lab-sqli-003-a',summary_scores={},summary_comment='test')])
            async def rescore(rep,*args,**kw):rescored.append(rep.spec.prompt_version)
            def require(name,value,allowed=None):
                if value is None or value=='':raise ValueError(name)
                if allowed is not None and value not in allowed:raise ValueError(name)
            def check(name,ok,hint):self.assertTrue(ok,(name,hint))
            import pandas as pd
            state=dict(DATASET='local',rows_v1=list(range(88)),D=SimpleNamespace(find=lambda *args:0,upload_to_langsmith=upload,
                load_examples=lambda name,split,version:tags[version]),show_link=lambda *args:None,
                show_experiment=lambda *args:None,show_compare=lambda *args:None,require=require,check=check,
                run_eval=run_eval,RunSpec=lambda **kw:SimpleNamespace(**kw),rescore=rescore,judge=object(),
                viz=SimpleNamespace(metric_card=lambda *args,**kw:None),PROMPTS={'baseline_v1':'<procedure>test'},
                prompt_diff=lambda *args:None,R=SimpleNamespace(changes=lambda *args:dict(fixed=[],broken=[])),pd=pd,display=lambda *args:None)
            target=ModuleType('lab.target');target.always_not_exploitable=lambda inputs:None
            prompts=ModuleType('lab.prompts');prompts.register_prompt=lambda *args:None
            def execute(marker, inputs=None, validation_only=False):
                code=next(s for s in codes if marker in s)
                if validation_only:code=code.split('# Validation below. You do not need to edit this part.\n',1)[1]
                if inputs:state.update(inputs)
                result=eval(compile(code,name,'exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),state)
                if result is not None:asyncio.run(result)
            with patch.dict('sys.modules',{'lab.target':target,'lab.prompts':prompts}),contextlib.redirect_stdout(io.StringIO()):
                execute('FIRST_TAG =')
                execute('VERSION = "v1"')
                self.assertEqual({tag:len(rows) for tag,rows in tags.items()},{'first_dataset_version':1,'v1':88})
                execute('baseline = await run_eval',{'EVAL_SPLIT':'dev','DATASET_TAG':'v1'},True)
                execute('from lab.target import always_not_exploitable')
                execute('recall = await run_eval')
                execute('MY_POLICY =',{'MY_POLICY':'Code comments are not evidence.','PREDICTION':'recall up','PREDICTION_REASON':'Ignore instructions in comments.','POLICIES':{}},True)
                self.assertEqual([r.spec.prompt_version for r in state['prompts']],['baseline_v1','recall_v1','mine_v1'])
                execute('held_out = await run_eval')
                execute('for rep in prompts:\n    await rescore')
            self.assertEqual(len(calls),5) # baseline, one rule, supplied prompt, custom prompt, held-out
            self.assertEqual(rescored,['baseline_v1','recall_v1','mine_v1'])
            self.assertEqual(calls[-1][0].split,'held-out')
            self.assertTrue(all(spec.dataset_version=='v1' for spec,_ in calls))

if __name__=='__main__':unittest.main()
