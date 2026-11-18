#!/usr/bin/env python3
"""Rebuild the revised paper's numerical tables from the packaged evidence.

Table III: build_mr_table(), automated scores plus the documented A/B final merge.
Table IV: build_stage1_table(), original Stage-1 result records.
Tables V--VII: build_control_tables(), repeated-prompt/bootstrap outputs.
Table VIII: build_flag_table(), supplied final flag judgments.
Section VI classifier audit: build_classifier_summary(), the exact 200-image sample.

No annotations are inferred or changed. Raw A/B labels determine agreement;
final-label files determine preservation, classifier accuracy and audit counts.
"""
import argparse
import json
import math
import statistics
from collections import Counter
from fractions import Fraction
from pathlib import Path
from package_paths import *

PT_ORDER = ['sensitive', 'expressive', 'neutral']
TR_ORDER = ['refusal', 'safety_label', 'semantic']
TR_NAMES = {'refusal':'Refusal','safety_label':'Safety label','semantic':'Semantics'}
PT_NAMES = {'sensitive':'Sensitive','expressive':'Expressive','neutral':'Natural'}

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

def tex_table(caption, headers, rows, align, label):
    return '\n'.join([r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(map(str,row))+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])

def build_mr_table(out):
    """Table III: final Scene/Policy/Joint counts; A/B kappa is pre-adjudication."""
    a=read_csv(MR_VALIDATION/'mr_annotations_A.csv');b=read_csv(MR_VALIDATION/'mr_annotations_B.csv');final=read_csv(MR_VALIDATION/'mr_adjudicated.csv')
    ab=joined(a,b,['mr_id','prompt_index'])
    ag={f:agreement([x[f] for x,y in ab],[y[f] for x,y in ab]) for f in ['semantics_preserved','safety_intent_preserved']}
    auto=read_json(MR_VALIDATION/'mr_validation_safemt.json')
    # Verify that the stored automated aggregate is recoverable from all 270x8 records.
    assert len(auto['per_pair'])==2160
    records=[];tex=[]
    for mr in MR_ORDER:
        rows=[r for r in final if r['mr_id']==mr];raw=[r for r in auto['per_pair'] if r['mr_id']==mr]
        sim=statistics.mean(r['bertscore_f1'] for r in raw);spr=sum(r['preserve_entailment'] for r in raw)/len(raw)
        assert len(raw)==270 and len(rows)==30 and abs(sim-auto['summary'][mr]['avg_sim_bertscore_f1'])<1e-10
        scene=sum(r['semantics_preserved']=='Y' for r in rows);policy=sum(r['safety_intent_preserved']=='Y' for r in rows)
        joint=lambda rs:sum(r['semantics_preserved']=='Y' and r['safety_intent_preserved']=='Y' for r in rs)
        contexts={pt:{'joint':joint([r for r in rows if r['prompt_type']==pt]),'n':sum(r['prompt_type']==pt for r in rows)} for pt in PT_ORDER}
        assert all(v['n']==10 for v in contexts.values())
        d={'mr_id':mr,'mr':MR_NAMES[mr],'n':len(rows),'sim':sim,'spr':spr,'scene':scene,'policy':policy,'joint':joint(rows),'contexts':contexts};records.append(d)
        tex.append([MR_NAMES[mr],f'{sim:.3f}',f'{100*spr:.1f}',f'{scene}/30',f'{policy}/30',*[f'{contexts[p]["joint"]}/10' for p in PT_ORDER],f'{joint(rows)}/30'])
    total={k:sum(r[k] for r in records) for k in ['n','scene','policy','joint']}
    total['contexts']={p:sum(r['contexts'][p]['joint'] for r in records) for p in PT_ORDER}
    tex.append(['Overall','--','--',f'{total["scene"]}/240',f'{total["policy"]}/240',*[f'{total["contexts"][p]}/80' for p in PT_ORDER],f'{total["joint"]}/240'])
    (out/'table_iii_mr_validation.tex').write_text(tex_table('Semantics-preserving validation of all eight MRs.',['MR','SIM','SPR (\\%)','Scene','Policy','Sensitive','Expressive','Neutral','Joint'],tex,'lrrrrrrrr','tab:mr-validation'))
    return {'agreement':ag,'rows':records,'total':total}

def build_flag_table(out):
    """Table VIII: genuine is interpreted separately for each flag trigger.

    Semantic genuine means a verified core-content difference. It is not, by
    itself, an independently verified policy violation or a causal safety mechanism.
    """
    a=read_csv(AUDIT/'flagged_pairs_A.csv');b=read_csv(AUDIT/'flagged_pairs_B.csv');final=read_csv(AUDIT/'flagged_pairs_final.csv')
    ab=joined(a,b,['pair_id']);ag=agreement([x['judgment'] for x,y in ab],[y['judgment'] for x,y in ab]);rows=[];tex=[]
    for tr in TR_ORDER:
        for pt in ['expressive','neutral','sensitive']:
            sub=[r for r in final if r['trigger']==tr and r['prompt_type']==pt];c=Counter(r['final_judgment'] for r in sub);n=len(sub)
            d={'trigger':tr,'prompt_type':pt,'n':n,'genuine':c['genuine'],'classifier_errors':c['safety_label_error'],'clip_errors':c['clip_error'],'false_alarm_ratio':(n-c['genuine'])/n};rows.append(d)
            tex.append([TR_NAMES[tr],PT_NAMES[pt],n,d['genuine'],'--' if tr=='semantic' else d['classifier_errors'],'--' if tr=='safety_label' else d['clip_errors'],f'{100*d["false_alarm_ratio"]:.2f}'])
    c=Counter(r['final_judgment'] for r in final);n=len(final);tex.append(['Total','',n,c['genuine'],c['safety_label_error'],c['clip_error'],f'{100*(n-c["genuine"])/n:.2f}'])
    (out/'table_viii_flag_adjudication.tex').write_text(tex_table('Final adjudication of 300 flags in Stage 1.',['Trigger','Prompt type','$n$','Genuine','Classifier errors','CLIP errors','False alarm rate (\\%)'],tex,'llrrrrr','tab:flag-adjudication'))
    return {'agreement':ag,'rows':rows,'counts':dict(c),'sample_genuine_ratio':c['genuine']/n,'sample_genuine_ci95':wilson(c['genuine'],n),'by_model':dict(Counter(r['model'] for r in final))}

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

def build_stage1_table(out):
    """Table IV: original complete 270x8 Stage-1 records, before any manual audit."""
    result={};tex=[]
    for m in MODEL_ORDER:
        rows=read_jsonl(MAIN_RESULTS[m]);assert len(rows)==2160
        n=len(rows);st=Counter(r['stage'] for r in rows);nbp=n-st['level1_mismatch']-st['level1_both_blocked'];nl3=st['level3_consistent']+st['level3_drift'];flag=sum(r['final_consistent'] is False for r in rows)
        result[m]={'n':n,'flagged':flag,'IR':flag/n,'CR':1-flag/n,'response_consistency':1-st['level1_mismatch']/n,'label_consistency':1-st['level2_mismatch']/nbp,'semantic_consistency':st['level3_consistent']/nl3,'stages':dict(st)}
        # Full Table IV breakdown, retaining each check's own conditional denominator.
        strata={}
        ptype=lambda r:{'plain_benign':'neutral','expressive_benign':'expressive'}.get(r['prompt_type'],r['prompt_type'])
        for field,values in [('prompt_type',PT_ORDER),('category',sorted({r['category'] for r in rows})),('mr_id',MR_ORDER)]:
            for value in values:
                sub=[r for r in rows if (ptype(r) if field=='prompt_type' else r[field])==value];c=Counter(r['stage'] for r in sub);nn=len(sub);nb=nn-c['level1_mismatch']-c['level1_both_blocked'];nl=c['level3_consistent']+c['level3_drift'];nf=sum(r['final_consistent'] is False for r in sub)
                strata[field+':'+value]={'n':nn,'flagged':nf,'IR':nf/nn,'CR':1-nf/nn,'response_consistency':1-c['level1_mismatch']/nn,'label_consistency':1-c['level2_mismatch']/nb if nb else None,'semantic_consistency':c['level3_consistent']/nl if nl else None}
        result[m]['strata']=strata
        tex.append([MODEL_NAMES[m],n,flag,f'{100*flag/n:.2f}',f'{100*(1-flag/n):.2f}'])
    (out/'table_iv_stage1_overall.tex').write_text(tex_table('Stage-1 overall results on all 270 source prompts and eight MRs.',['Model','Pairs','Flagged','IR (\\%)','CR (\\%)'],tex,'lrrrr','tab:stage1-overall'))
    return result

def build_control_tables(out):
    """Tables V/VI/VII: each observed rate uses its matched repetition population.

    Observed and repeated-generation rates use matching evaluable prompt IDs.
    Bootstrap intervals and BH-adjusted tests are stored in revision_analysis.json.
    """
    analysis=read_json(BASELINE/'revision_analysis.json');tex=[]
    for m in MODEL_ORDER:
        s=analysis[m]['strata']['overall'];o=s['observed'];b=s['baseline'];e=s['excess']['flagged']
        cells=[MODEL_NAMES[m]]
        for key in ['resp_mis','label_mis','sem_mis','flagged']:cells += [f'{100*o[key]:.2f}',f'{100*b[key]:.2f}']
        cells += [f'{100*e["value"]:+.2f} [{100*e["lo"]:.2f}, {100*e["hi"]:.2f}]'];tex.append(cells)
    (out/'table_v_repeated_generation.tex').write_text(tex_table('Transformed versus repeated-prompt flag rates at $\\epsilon=0.30$ (\\%).',['Model','T refusal','B refusal','T label','B label','T semantic','B semantic','T overall','B overall','Excess IR [95\\% CI]'],tex,'lrrrrrrrrl','tab:repeated-generation'))
    tex=[]
    for mr in MR_ORDER:
        cells=[MR_NAMES[mr]]
        for m in MODEL_ORDER:
            e=analysis[m]['strata']['mr:'+mr]['excess']['flagged'];cells.append(f'{100*e["value"]:+.2f} [{100*e["lo"]:.2f}, {100*e["hi"]:.2f}]')
        tex.append(cells)
    (out/'table_vi_mr_excess.tex').write_text(tex_table('MR-wise excess IR over identical-prompt repetitions at $\\epsilon=0.30$ (percentage points; 95\\% CI).',['MR',*[MODEL_NAMES[m] for m in MODEL_ORDER]],tex,'llll','tab:mr-excess'))
    tex=[]
    for kind,ok,bk in [('Overall IR','observed_flagged','baseline_flagged'),('Conditional semantic flags','observed_sem','baseline_sem')]:
        for eps in [.20,.25,.30,.35,.40]:
            row=[kind,f'{eps:.2f}']
            for m in MODEL_ORDER:
                x=analysis[m]['threshold'][f'{eps:.3f}'];row += [f'{100*x[ok]:.2f}',f'{100*x[bk]:.2f}']
            tex.append(row)
    (out/'table_vii_threshold_sensitivity.tex').write_text(tex_table('Threshold sensitivity of overall IR and conditional semantic flags (\\%).',['Metric','$\\epsilon$','GPT T','GPT B','Gemini T','Gemini B','SDXL T','SDXL B'],tex,'llrrrrrr','tab:threshold-sensitivity'))
    return analysis

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=REV/'tables');args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    mr=build_mr_table(out);flags=build_flag_table(out);classifier=build_classifier_summary();stage1=build_stage1_table(out);controls=build_control_tables(out)
    result={'table_iii':mr,'table_iv':stage1,'table_viii':flags,'classifier_audit_200':classifier,'baseline_coverage':{m:{k:controls[m][k] for k in ['n_prompts_covered','n_pairs_observed','n_pairs_baseline']} for m in MODEL_ORDER}}
    (out/'paper_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    md=['# Recomputed paper results','', 'All values below are computed from the published files.','',f'- Table III: Scene {mr["total"]["scene"]}/240; Policy {mr["total"]["policy"]}/240; Joint {mr["total"]["joint"]}/240.',f'- MR pre-adjudication kappa: scene {mr["agreement"]["semantics_preserved"]["kappa"]:.6f}; policy {mr["agreement"]["safety_intent_preserved"]["kappa"]:.6f}.',f'- Table VIII: {flags["counts"]["genuine"]} genuine; {flags["counts"]["safety_label_error"]} classifier errors; {flags["counts"]["clip_error"]} CLIP errors.',f'- Flag audit pre-adjudication kappa: {flags["agreement"]["kappa"]:.6f}; agreement {100*flags["agreement"]["agreement"]:.2f}%.',f'- Classifier audit: {classifier["correct"]}/200 correct ({100*classifier["accuracy"]:.1f}%); kappa {classifier["agreement"]["kappa"]:.6f}; agreement {100*classifier["agreement"]["agreement"]:.1f}%.','', 'The MR final file retains all A/B agreements and resolves only the three documented disputed fields. See `04_mr_validation/mr_adjudication_decisions.csv`.']
    (out/'paper_results.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'table_iii_total':mr['total'],'table_viii_counts':flags['counts'],'flag_kappa':flags['agreement']['kappa'],'classifier_accuracy':classifier['accuracy'],'classifier_kappa':classifier['agreement']['kappa']},ensure_ascii=False))

if __name__=='__main__':main()
