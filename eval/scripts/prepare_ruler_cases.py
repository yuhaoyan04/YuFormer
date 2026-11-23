#!/usr/bin/env python3
"""Freeze paired RULERv1 cases with the actual training tokenizer."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml
from tokenizers import Tokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_dependencies(nltk_data, *, needs_sentences):
    inputs={};versions={}
    # Pin code, lexicons and package metadata as well as the tokenizer artifact.
    # A word-list or sentence-boundary change otherwise changes the actual cases
    # while leaving the upstream generator and its random seed identical.
    for name in ('nltk','wonderwords','tokenizers','jsonlines','tenacity','numpy','scipy','pyyaml'):
        dist=importlib.metadata.distribution(name);versions[name]=dist.version
        for item in dist.files or []:
            if Path(str(item)).suffix not in ('.py','.txt','.json','.so') and item.name not in ('METADATA','RECORD'):
                continue
            path=Path(dist.locate_file(item)).resolve()
            if path.is_file():inputs[str(path)]=digest(path)
    if needs_sentences:
        if nltk_data is None:raise ValueError('essay cases require an explicit frozen NLTK sentence-model directory')
        base=Path(nltk_data).resolve()/'tokenizers/punkt_tab/english'
        for name in ('abbrev_types.txt','collocations.tab','ortho_context.tab','sent_starters.txt'):
            path=base/name
            if not path.is_file():raise ValueError('frozen English sentence-model asset is absent: '+str(path))
            inputs[str(path)]=digest(path)
    return inputs,versions


def select_length_conditioned_cases(rows,tokenizer,*,seq,generation_tokens,samples):
    if len({r['index'] for r in rows})!=len(rows):raise ValueError('raw RULER sample identities are duplicated')
    selected=[];lengths=[]
    for row in rows:
        count=len(tokenizer.encode(row['input']+row.get('answer_prefix',''),add_special_tokens=False).ids)
        if (not row['outputs'] or any(not isinstance(x,str) or not x for x in row['outputs'])
                or count+generation_tokens>seq):
            raise ValueError('raw RULER prompt/answer/hard generation budget does not close')
        if count>=seq*.75 and len(selected)<samples:
            selected.append(row);lengths.append(count)
    if len(selected)!=samples:raise ValueError('fixed oversampling cannot cover the unchanged75pct context-length floor')
    return selected,lengths


def prepare(root,ruler,output,*,lengths,tasks,samples,seed,nltk_data=None,corpus_admission=None,oversample_factor=1):
    root,ruler,output=map(lambda p:Path(p).resolve(),(root,ruler,output))
    if output.exists():raise ValueError('preserve previous case preparation attempts')
    if (type(samples)is not int or samples<=0 or not lengths or len(set(lengths))!=len(lengths)
            or any(s not in (4096,16384,65536) for s in lengths)
            or not tasks or len(set(tasks))!=len(tasks)
            or type(oversample_factor)is not int or oversample_factor not in (1,2)):
        raise ValueError('explicit positive case count and supported context roster are required')
    lock=json.loads((ruler/'upstream.lock.json').read_bytes())
    if lock['revision']!='db43f92ff634bbce1c33d1fb15858aec4bba509a':raise ValueError('RULER revision mismatch')
    for name,row in lock['files'].items():
        if digest(ruler/name)!=row['sha256']:raise ValueError('pinned upstream generator changed before import')
    configs=yaml.safe_load((ruler/'scripts/synthetic.yaml').read_bytes())
    if len(configs)!=13 or not set(tasks)<=set(configs):raise ValueError('RULER task roster mismatch')
    path=ruler/'scripts/data/synthetic/constants.py'
    spec=importlib.util.spec_from_file_location('_pinned_ruler_templates',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gate_path=root/'data/TOKENIZER_READY.json';gate=json.loads(gate_path.read_bytes())
    tokenizer_path=Path(gate['tokenizer_path'])
    if not tokenizer_path.is_absolute():tokenizer_path=root/tokenizer_path
    if gate['status']!='PASS' or digest(tokenizer_path)!=gate['tokenizer_sha256']:raise ValueError('training tokenizer bytes changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    inputs={str(ruler/name):row['sha256'] for name,row in lock['files'].items()}
    for p in [Path(__file__),Path(__file__).with_name('exec_ruler_generator.py'),
              root/'eval/src/yuformer_eval/ruler_tokenizer.py',gate_path,tokenizer_path,ruler/'upstream.lock.json']:
        inputs[str(p)]=digest(p)
    # Corpus assets are checked before any task can begin its length-filling
    # loop. Empty/missing essay inputs must never become an unbounded generator.
    required=set()
    for task in tasks:
        cfg=configs[task]
        if cfg['args'].get('type_haystack')=='essay':required.add('PaulGrahamEssays.json')
        if cfg['task']=='qa':required.add('squad.json' if cfg['args']['dataset']=='squad' else 'hotpotqa.json')
    for name in required:
        p=ruler/'scripts/data/synthetic/json'/name
        body=json.loads(p.read_bytes())
        if name=='PaulGrahamEssays.json' and len(body.get('text',''))<10000:raise ValueError('empty/short RULER essay corpus')
        if name!='PaulGrahamEssays.json' and len(body)<1:raise ValueError('empty RULER QA corpus')
        inputs[str(p)]=digest(p)
    if any(configs[task]['task']=='common_words_extraction' for task in tasks):
        lock_path=ruler/'english_words.lfs.lock.json';dictionary_lock=json.loads(lock_path.read_bytes())
        dictionary=ruler/'scripts/data/synthetic/json/english_words.json'
        if (dictionary_lock.get('status')!='PASS_EXACT_ORIGINAL_GIT_LFS_DICTIONARY'
                or dictionary_lock.get('revision')!=lock['revision']
                or dictionary.stat().st_size!=dictionary_lock.get('bytes')
                or digest(dictionary)!=dictionary_lock.get('sha256')
                or digest(dictionary_lock['pointer_file'])!=dictionary_lock.get('pointer_sha256')):
            raise ValueError('original Git LFS word dictionary has not been admitted')
        for path in (dictionary,lock_path,Path(dictionary_lock['pointer_file'])):
            inputs[str(path)]=digest(path)
    if required:
        if corpus_admission is None:raise ValueError('full original corpus download admission must be explicit')
        admission_path=Path(corpus_admission).resolve();admission=json.loads(admission_path.read_bytes())
        downloads=admission.get('downloaded',[])
        if (admission.get('status')!='PASS_ORIGINAL_RULER_V1_DOWNLOAD_ASSETS_ALL_URLS'
                or len(downloads)!=admission.get('urls_required')
                or len({row['original_url'] for row in downloads})!=len(downloads)):
            raise ValueError('original RULER corpus URL roster is incomplete')
        inputs[str(admission_path)]=digest(admission_path)
        for name in required:
            p=ruler/'scripts/data/synthetic/json'/name
            if admission.get('assets',{}).get(str(p))!=digest(p):raise ValueError('admitted original corpus changed')
        for row in downloads+admission.get('qa_downloads',[]):
            p=Path(row['file'])
            if digest(p)!=row['sha256']:raise ValueError('archived original corpus source bytes changed')
            inputs[str(p)]=row['sha256']
        for name,sha in admission.get('source_inputs',{}).items():
            if digest(name)!=sha:raise ValueError('original corpus parser/URL roster changed')
            inputs[name]=sha
    dependency_inputs,versions=runtime_dependencies(nltk_data,needs_sentences=any(
        configs[task]['args'].get('type_haystack')=='essay' for task in tasks))
    inputs.update(dependency_inputs)
    output.mkdir(parents=True)
    def guard():
        if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('RULER generator/tokenizer/corpus changed during case preparation')
    manifest={'schema_version':1,'status':'PREPARING_IMMUTABLE_PAIRED_RULER_CASES','benchmark':'RULERv1',
        'revision':lock['revision'],'model_template':'base','seed':seed,'samples_per_task':samples,
        'task_roster':list(tasks),'lengths':lengths,'tokenizer_sha256':gate['tokenizer_sha256'],
        'scope':'FULL_ORIGINAL_13_TASK_500_CASE_PROTOCOL' if set(tasks)==set(configs) and samples==500 and oversample_factor==1 else 'PRETRAIN_STAGE_DIAGNOSTIC_SUBSET_NOT_PUBLISHED_BENCHMARK_SCORE',
        'oversample_factor':oversample_factor,
        'selection_policy':'first_N_original_cases_with_actual_native_prompt_at_least75pct_context_and_reserved_generation_budget_no_model_metric_selection',
        'runtime_dependency_versions':versions,'python_version':sys.version,
        'nltk_data':str(Path(nltk_data).resolve()) if nltk_data else None,
        'source_inputs':inputs,'files':[]}
    for seq in lengths:
        for task in tasks:
            guard();cfg=configs[task];base=module.TASKS[cfg['task']]
            template=base['template']+base.get('answer_prefix','')
            folder=output/str(seq);folder.mkdir(exist_ok=True)
            command=[sys.executable,str(Path(__file__).with_name('exec_ruler_generator.py')),
                '--ruler-root',str(ruler),'--task-script',cfg['task'],
                '--save_dir',str(folder),'--save_name',task,'--subset','validation',
                '--tokenizer_path',str(tokenizer_path),'--tokenizer_type','hf',
                '--max_seq_length',str(seq),'--tokens_to_generate',str(base['tokens_to_generate']),
                '--num_samples',str(samples*oversample_factor),'--random_seed',str(seed),'--template',template]
            for name,value in cfg['args'].items():command.extend(['--'+name,str(value)])
            if cfg['task']=='qa':command.extend(['--pre_samples','0'])
            log=folder/(task+'.generation.log')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            if nltk_data is not None:env['NLTK_DATA']=str(Path(nltk_data).resolve())
            with log.open('w') as handle:
                subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,
                    check=True,timeout=1800)
            raw_path=folder/task/'validation.jsonl';raw_rows=[json.loads(line) for line in raw_path.read_text().splitlines()]
            if len(raw_rows)!=samples*oversample_factor:raise ValueError('raw RULER sample roster has missing identities')
            rows,actual_lengths=select_length_conditioned_cases(raw_rows,tokenizer,seq=seq,
                generation_tokens=base['tokens_to_generate'],samples=samples)
            inputs[str(raw_path)]=digest(raw_path)
            path=folder/task/'admitted.validation.jsonl'
            with path.open('x') as handle:
                for row in rows:handle.write(json.dumps(row,sort_keys=True)+'\n')
                handle.flush();os.fsync(handle.fileno())
            with path.open('rb') as handle:os.fsync(handle.fileno())
            manifest['files'].append({'file':str(path),'sha256':digest(path),'task':task,'seq_len':seq,
                'samples':samples,'generation_tokens':base['tokens_to_generate'],
                'original_sample_count':len(raw_rows),'raw_file':str(raw_path),'raw_sha256':digest(raw_path),
                'admitted_original_indices':[r['index'] for r in rows],
                'input_tokens_min':min(actual_lengths),'input_tokens_max':max(actual_lengths)})
            (output/'progress.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    guard();manifest['status']='PASS_IMMUTABLE_PAIRED_RULER_CASE_MANIFEST'
    (output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--ruler-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--lengths',type=int,nargs='+',default=[4096,16384,65536])
    parser.add_argument('--tasks',nargs='+');parser.add_argument('--samples',type=int,default=32);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--nltk-data',type=Path);parser.add_argument('--corpus-admission',type=Path)
    parser.add_argument('--oversample-factor',type=int,choices=[1,2],default=1)
    args=parser.parse_args();tasks=args.tasks or list(yaml.safe_load((args.ruler_root/'scripts/synthetic.yaml').read_bytes()))
    result=prepare(args.root,args.ruler_root,args.output,lengths=args.lengths,tasks=tasks,samples=args.samples,seed=args.seed,
        nltk_data=args.nltk_data,corpus_admission=args.corpus_admission,oversample_factor=args.oversample_factor)
    print(json.dumps({'status':result['status'],'scope':result['scope'],'files':len(result['files'])},sort_keys=True))
