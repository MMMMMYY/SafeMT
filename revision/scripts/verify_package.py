#!/usr/bin/env python3
"""Verify the released revision evidence and its links to the paper outputs.

Checks original-label joins, the exact audit samples, independent pre-resolution
agreement, Table III/VIII totals, the 200-image classifier reference, local images,
canonical MR inputs, and regenerated statistical outputs when available.
"""
import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from package_paths import *
from fractions import Fraction
import math

def agreement(a, b):
    assert len(a) == len(b) and len(a) > 0
    n=len(a);same=sum(x==y for x,y in zip(a,b));ca=Counter(a);cb=Counter(b)
    pe=sum(Fraction(ca[x]*cb[x],n*n) for x in set(ca)|set(cb))
    return {'n':n,'agree':same,'disagree':n-same,'agreement':same/n,'kappa':None if pe==1 else float((Fraction(same,n)-pe)/(1-pe))}

def wilson(k, n):
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def joined(a, b, keys):
    key=lambda r:tuple(r[k] for k in keys)
    aa={key(r):r for r in a};bb={key(r):r for r in b}
    assert len(aa)==len(a) and len(bb)==len(b) and aa.keys()==bb.keys()
    return [(r,bb[key(r)]) for r in a]

def build_classifier_summary():
    """Section VI: 200-image accuracy; NudeNet for sex, Q16 for other categories.

    The four disagreements use the supplied adjudication labels. The exact
    published sample IDs, rather than an inferred seed, define this 200-image audit.
    """
    a=read_csv(AUDIT/'classifier_images_A_200.csv');b=read_csv(AUDIT/'classifier_images_B_200.csv');final=read_csv(AUDIT/'classifier_images_final_200.csv')
    ab=joined(a,b,['image']);ag=agreement([x['unsafe'] for x,y in ab],[y['unsafe'] for x,y in ab])
    positive=lambda s:s.strip().lower() in {'1','unsafe','true','nsfw','nude','inappropriate'}
    counts=Counter();by_classifier={}
    for r in final:
        col='nudenet_label' if r['category']=='sexual_content' else 'q16_label'
        pred=int(positive(r[col]));truth=int(r['final_unsafe']);counts[['TN','FP','FN','TP'][truth*2+pred]]+=1
    def stats(rows,col):
        c=Counter()
        for r in rows:c[['TN','FP','FN','TP'][int(r['final_unsafe'])*2+int(positive(r[col]))]]+=1
        return {'n':len(rows),'counts':dict(c),'accuracy':(c['TP']+c['TN'])/len(rows)}
    by_classifier['Q16']=stats([r for r in final if r['category']!='sexual_content'],'q16_label')
    by_classifier['NudeNet']=stats([r for r in final if r['category']=='sexual_content'],'nudenet_label')
    return {'n':len(final),'agreement':ag,'correct':counts['TP']+counts['TN'],'accuracy':(counts['TP']+counts['TN'])/len(final),'accuracy_ci95':wilson(counts['TP']+counts['TN'],len(final)),'confusion':dict(counts),'by_classifier':by_classifier,'by_model':dict(Counter(r['model'] for r in final))}

def checksum(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=None);args=ap.parse_args();checks={}
    aa=read_csv(AUDIT/'flagged_pairs_A.csv');bb=read_csv(AUDIT/'flagged_pairs_B.csv');final=read_csv(AUDIT/'flagged_pairs_final.csv');pairs=joined(aa,bb,['pair_id'])
    assert len(final)==len(pairs)==300 and len({r['pair_id'] for r in final})==300
    am={r['pair_id']:r for r in aa};bm={r['pair_id']:r for r in bb}
    allowed={'refusal':{'genuine','other'},'safety_label':{'genuine','safety_label_error'},'semantic':{'genuine','clip_error'}}
    for r in final:
        k=r['pair_id'];assert r['A_judgment']==am[k]['judgment'] and r['B_judgment']==bm[k]['judgment'];assert r['final_judgment'] in allowed[r['trigger']]
        for f in ['source_prompt','followup_prompt','trigger','model','prompt_type']:assert r[f]==am[k][f]==bm[k][f]
    a=agreement([x['judgment'] for x,y in pairs],[y['judgment'] for x,y in pairs]);assert a['agree']==292 and abs(a['kappa']-.9192408641227539)<1e-12
    assert Counter(r['final_judgment'] for r in final)=={'genuine':246,'safety_label_error':21,'clip_error':33}
    assert Counter(r['model'] for r in final)=={'GPT-Image':65,'Gemini-Image':89,'SDXL':146}
    design=read_json(AUDIT/'sampling_design.json')
    for tr,by in design['sample_by_trigger_context'].items():
        for pt,n in by.items():assert sum(r['trigger']==tr and r['prompt_type']==pt for r in final)==n
    checks['flagged_audit']={'n':300,'independent_agree':292,'kappa':a['kappa'],'metadata_and_labels':'preserved'}
    classifier=build_classifier_summary();assert classifier['n']==200 and classifier['correct']==168 and classifier['agreement']['agree']==196 and abs(classifier['agreement']['kappa']-.9588265568708183)<1e-12
    decisions=read_csv(AUDIT/'classifier_images_adjudication_200.csv');assert len(decisions)==4
    cfinal=read_csv(AUDIT/'classifier_images_final_200.csv');decision={r['image']:r for r in decisions}
    for r in cfinal:
        if r['A_unsafe']==r['B_unsafe']:assert r['final_unsafe']==r['A_unsafe']
        else:assert r['final_unsafe']==decision[r['image']]['final_unsafe']
    checks['classifier_audit']={'n':200,'correct':168,'accuracy':.84,'kappa':classifier['agreement']['kappa']}
    mr=read_csv(MR_VALIDATION/'mr_adjudicated.csv');ma=read_csv(MR_VALIDATION/'mr_annotations_A.csv');mb=read_csv(MR_VALIDATION/'mr_annotations_B.csv');mab=joined(ma,mb,['mr_id','prompt_index']);assert len(mr)==len(mab)==240
    assert Counter(r['mr_id'] for r in mr)=={m:30 for m in MR_ORDER}
    assert all(sum(r['mr_id']==m and r['prompt_type']==p for r in mr)==10 for m in MR_ORDER for p in ['sensitive','expressive','neutral'])
    counts={'scene':sum(r['semantics_preserved']=='Y' for r in mr),'policy':sum(r['safety_intent_preserved']=='Y' for r in mr),'joint':sum(r['semantics_preserved']==r['safety_intent_preserved']=='Y' for r in mr)}
    # Verify lineage rather than accepting a fixed paper total from another export.
    final_by_id={(r['mr_id'],r['prompt_index']):r for r in mr}
    assert len(final_by_id)==len(mr)
    decisions=read_csv(MR_VALIDATION/'mr_adjudication_decisions.csv')
    decision_by_field={(r['mr_id'],r['prompt_index'],r['field']):r for r in decisions}
    assert len(decision_by_field)==len(decisions)
    disputed=set();agreed=0
    for ar,br in mab:
        k=(ar['mr_id'],ar['prompt_index']);fr=final_by_id[k]
        for field,allowed in [('semantics_preserved',{'Y','N'}),('safety_intent_preserved',{'Y','N','L'})]:
            assert fr[field] in allowed
            if ar[field]==br[field]:
                assert fr[field]==ar[field],('Changed A/B agreement',k,field)
                agreed+=1
            else:
                dk=(*k,field);disputed.add(dk);dr=decision_by_field[dk]
                assert (dr['A'],dr['B'],dr['final'])==(ar[field],br[field],fr[field])
                assert dr['reason'].strip()
    assert decision_by_field.keys()==disputed
    rec=read_csv(MR_VALIDATION/'mr_label_reconciliation.csv')
    rec_by_field={(r['mr_id'],r['prompt_index'],r['field']):r for r in rec}
    assert len(rec_by_field)==len(rec) and rec_by_field.keys()==disputed
    for dk,r in rec_by_field.items():
        dr=decision_by_field[dk]
        assert r['A_B_agree']=='false' and all(r[f]==dr[f] for f in ['A','B','final'])
    lineage=read_json(MR_VALIDATION/'mr_adjudication.json')
    for item in lineage['input_files']:
        assert checksum(REV/item['file'])==item['sha256']
    assert checksum(MR_VALIDATION/'mr_adjudicated.csv')==lineage['final_sha256']
    assert checksum(MR_VALIDATION/'mr_adjudication_decisions.csv')==lineage['decisions_sha256']
    assert lineage['unanimous_fields_retained']==agreed and lineage['disputed_fields_resolved']==len(disputed)
    checks['MR_validation']={'n':len(mr),'final_counts':counts,'reconciliation_fields':len(rec),'unanimous_fields_retained':agreed,'lineage':'all agreements retained; all disagreements have decisions'}
    source=load_source_prompts();follow={m:load_followups(m) for m in MR_ORDER};assert len(source)==270 and all(len(v)==270 for v in follow.values())
    auto=read_json(MR_VALIDATION/'mr_validation_safemt.json');assert len(auto['per_pair'])==2160
    for r in auto['per_pair']:
        assert r['source_prompt']==source[r['prompt_index']]
        assert r['followup_prompt']==follow[r['mr_id']][r['prompt_index']]
    checks['canonical_prompts']={'sources':270,'followups':2160,'matches_automated_validation':True}
    images=read_json(AUDIT/'image_index.json')
    for d in images.values():assert (REV/d['preview']).is_file() and checksum(REV/d['preview'])==d['preview_sha256']
    for r in read_csv(AUDIT/'figure_03_examples.csv'):
        for side in ['source','followup']:assert checksum(REV/r[side+'_image'])==r[side+'_sha256']
    checks['images']={'audit_previews':len(images),'figure3_originals':8,'hashes_match':True}
    target_slots={(i,r) for i in range(SOURCE_PROMPT_COUNT) for r in range(1,REPETITION_COUNT+1)}
    for m in MODEL_ORDER:
        assert len(read_jsonl(MAIN_RESULTS[m]))==2160
        attempts=read_jsonl(REP_MANIFESTS[m]);latest={}
        assert {(r['prompt_index'],r['rep']) for r in attempts}==target_slots
        for r in attempts:
            k=(r['prompt_index'],r['rep'])
            if k not in latest or r['status'] in {'generated','blocked'} or latest[k]['status']=='error':latest[k]=r
        valid={i:sorted(r for (j,r),v in latest.items() if j==i and v['status']!='error') for i in range(SOURCE_PROMPT_COUNT)}
        expected={(i,ra,rb) for i,reps in valid.items() for a,ra in enumerate(reps) for rb in reps[a+1:]}
        pairs=read_jsonl(BASELINE/f'{m}_baseline_pairs.jsonl')
        actual={(r['prompt_index'],r['rep_a'],r['rep_b']) for r in pairs}
        assert len(actual)==len(pairs) and actual==expected
    checks['repetition_protocol']={'source_prompts_per_model':SOURCE_PROMPT_COUNT,'repetitions_per_prompt':REPETITION_COUNT,'manifest_slots':'all planned slots represented; pair evidence matches recorded non-error outcomes'}
    for rel,frozen in [('baseline_recomputed/revision_analysis.json',BASELINE/'revision_analysis.json'),('stage2_recomputed/stage2_control.json',STAGE2/'stage2_control.json')]:
        p=GENERATED/rel
        if p.exists():assert read_json(p)==read_json(frozen);checks[rel]='exact match'
    missing=[]
    for p in [ROOT/'README.md',*REV.rglob('*.md')]:
        if 'generated' in p.relative_to(ROOT).parts:continue
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in link or link.startswith('#'):continue
            target=link.split('#',1)[0]
            if target and not (p.parent/target).exists():missing.append({'file':str(p.relative_to(ROOT)),'link':link})
    assert not missing,missing
    for p in (REV/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    checks['links_and_scripts']={'markdown_links':'all local targets exist','python_syntax':'passed'}
    checks['paper_reference']=read_json(REV/'paper_reference.json')
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
