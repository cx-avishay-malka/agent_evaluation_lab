import ast,base64,io,json,random,tarfile
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import unittest

ROOT=Path(__file__).resolve().parents[1]
NAMES=['LangSmith_Eval_Lab_Colab.ipynb']

def notebook(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def bundle(name):
    code=next(''.join(c['source']) for c in notebook(name)['cells'] if '_PAYLOAD = (' in ''.join(c['source']))
    payload=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_PAYLOAD' for t in n.targets))
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)),mode='r:gz') as t:
        return {m.name:t.extractfile(m).read().decode('utf8') for m in t.getmembers() if m.isfile()}

def functions(source,names,namespace):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    # Postpone annotations so tests do not need model-provider SDKs.
    header=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[header,*nodes],type_ignores=[])),'bundled','exec'),namespace)
    return namespace

class ExplorationTests(unittest.TestCase):
    def test_split_is_larger_without_leakage(self):
        b=bundle(NAMES[0]);rows=[json.loads(l) for l in b['data/raw/lab/findings.jsonl'].splitlines()]
        ns=functions(b['src/lab/dataset.py'],['group_split','label_of','leaked_groups'],dict(random=random,defaultdict=defaultdict))
        old=ns['group_split'](rows,test_frac=.7);new=ns['group_split'](rows)
        uid=lambda r:r['inputs']['input']['finding_id']
        self.assertEqual(Counter(r['split'] for r in new),{'dev':44,'held-out':44})
        self.assertFalse(ns['leaked_groups'](new))
        self.assertTrue({uid(r) for r in old if r['split']=='dev'} <= {uid(r) for r in new if r['split']=='dev'})
        self.assertEqual(sum(r['split']=='dev' and ns['label_of'](r)=='exploitable' for r in new),21)
        self.assertEqual({uid(r) for r in rows},{uid(r) for r in new})
        processed=[json.loads(l) for l in b['data/processed/lab/dataset_v1.jsonl'].splitlines()]
        self.assertEqual(processed,new)

    def test_question_answers_match_the_data_and_quizzes_execute(self):
        for name in NAMES:
            b=bundle(name);rows=[json.loads(l) for l in b['data/raw/lab/findings.jsonl'].splitlines()]
            first=next(r for r in rows if r['inputs']['input']['finding_id']=='lab-cmdi-001-b')
            captured=[];quiz=ModuleType('jupyterquiz');quiz.display_quiz=lambda q,**kw:captured.extend(q)
            with patch.dict('sys.modules',{'jupyterquiz':quiz}):
                for c in notebook(name)['cells']:
                    s=''.join(c['source'])
                    if c['cell_type']=='code' and 'display_quiz(' in s:exec(s,{})
            def correct(q):return [a['answer'] for a in q['answers'] if a.get('correct')]
            severity=next(q for q in captured if 'severity of lab-cmdi' in q['question'])
            java=next(q for q in captured if 'how many examples use Java?' in q['question'])
            self.assertEqual(correct(severity),[first['inputs']['input']['vulnerability_info']['severity'].title()])
            self.assertEqual(correct(java),[str(sum(r['metadata']['language']=='java' for r in rows))])
            rules=next(q for q in captured if 'passes both checks but' in q['question'])
            self.assertNotIn('5',rules['question']);self.assertEqual(correct(rules),['Reason 5'])

    def test_compare_url_sets_the_actual_source(self):
        ns=functions(bundle(NAMES[0])['src/lab/report.py'],['dataset_url','compare_url'],{})
        before=SimpleNamespace(experiment_id='before',experiment_url='https://smith.langchain.com/o/org/datasets/data/compare?old=true')
        after=SimpleNamespace(experiment_id='after',experiment_url=before.experiment_url)
        self.assertEqual(ns['compare_url'](before,after),'https://smith.langchain.com/o/org/datasets/data/compare?selectedSessions=before%2Cafter&source=before')
        self.assertIsNone(ns['compare_url']())

    def test_feedback_and_labels_are_checked(self):
        ns=functions(bundle(NAMES[0])['src/lab/runner.py'],['check_examples','check_result_rows'],dict(
            Counter=Counter,TechnicalFailure=RuntimeError,example_name=lambda e:e.name,field_value=lambda o:(o or {}).get('label')))
        examples=[SimpleNamespace(id='1',name='lab-xss-005-a',outputs={'label':'exploitable'}),SimpleNamespace(id='2',name='safe',outputs={'label':'not exploitable'})]
        spec=SimpleNamespace(num_runs=1,dataset='data',split='dev',dataset_version='v1')
        keys=['exploitability','outcome','is_false_positive','is_false_negative']
        rows=[dict(example=e,run=SimpleNamespace(id='run-'+e.id),evaluation_results={'results':[{'key':k} for k in keys]}) for e in examples]
        ns['check_examples'](examples,spec);ns['check_result_rows'](rows,examples,spec)
        with self.assertRaisesRegex(RuntimeError,'Missing findings'):ns['check_result_rows'](rows[:1],examples,spec)
        with self.assertRaisesRegex(RuntimeError,'Incomplete evaluation'):ns['check_result_rows']([rows[0],rows[0]],examples,spec)
        rows[0]['evaluation_results']['results']=[]
        with self.assertRaisesRegex(RuntimeError,'lab-xss-005-a'):ns['check_result_rows'](rows,examples,spec)
        examples[0].outputs=None
        with self.assertRaisesRegex(RuntimeError,'reference labels'):ns['check_examples'](examples,spec)

    def test_judge_split_and_visible_disagreement_evidence(self):
        b=bundle(NAMES[0]);items=[json.loads(l) for l in b['data/raw/lab/judge_calibration.jsonl'].splitlines()]
        self.assertEqual(Counter(i['split'] for i in items),{'dev':23,'held-out':21})
        self.assertIn('JUDGE_SPLITS = ("dev", "held-out")',b['src/lab/calibration.py'])
        for name in NAMES:
            n=notebook(name);joined='\n'.join(''.join(c['source']) for c in n['cells'])
            self.assertNotIn('split="test"',joined)
            self.assertIn('Judge disagreements — development set',joined)
            self.assertIn('Why the person chose this label',joined)
            self.assertIn('viz.show_finding(selected["finding_id"], show_label=False)',joined)
            self.assertNotIn('Read the card from the top',joined)
            self.assertNotIn('### 7.5',joined)
            self.assertNotIn('FIRST_VERSION_COUNT',joined)
            self.assertFalse('NO_VERDICT_RATE' in joined, name)
            self.assertIn('<details><summary>Hint:',joined)

    def test_embedded_sources_compile(self):
        a=bundle(NAMES[0])
        for name,source in a.items():
            if name.endswith('.py'):compile(source,name,'exec')

if __name__=='__main__':unittest.main()
