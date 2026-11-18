#!/usr/bin/env python3
"""Figures 4 and 5: plot the corrected, packaged all-pairs saliency records.

Figure 4 histograms use individual records and the same metrics() function as
revision_stage2_control.py. Figure 5 reads the corresponding frozen control
summary; its displayed percentages are never hard-coded in this plotting script.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'safemt-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from package_paths import REV, STAGE2, MODEL_ORDER, MODEL_NAMES, read_jsonl, read_json
from revision_stage2_control import metrics

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})

def panel(ax, values, kind, title):
    ax.set_title(title,fontsize=8)
    if not values:
        ax.text(.5,.5,'N/A',transform=ax.transAxes,ha='center',va='center',fontsize=18,color='#666666')
        ax.set_xlim(0,1);ax.set_ylim(0,1);return
    ax.hist(values,bins=20 if kind!='shift' else 30,color='#547CA0' if kind!='MR' else '#C86262',edgecolor='white',linewidth=.35)
    ax.set_ylabel('Count');ax.tick_params(labelsize=7)
    if kind=='response':
        n=len(values);k=sum(x<=.30 for x in values);ax.axvline(.30,color='#222222',linestyle='--',linewidth=.8);ax.set_xlim(0,1);ax.set_xlabel('Normalized anchor rank')
        note=f'Top 30%: {k}/{n} ({100*k/n:.1f}%)'
    elif kind=='MR':
        n=len(values);k=sum(x<.5 for x in values);ax.axvline(.5,color='#222222',linestyle='--',linewidth=.8);ax.set_xlim(0,1);ax.set_xlabel('MR rank')
        note=f'More salient half:\n{k}/{n} ({100*k/n:.1f}%)'
    else:
        n=len(values);k=sum(x<0 for x in values);l=sum(x>0 for x in values);ax.axvline(0,color='#B43B3B',linestyle='--',linewidth=.8);ax.set_xlabel('Anchor shift')
        note=f'More: {100*k/n:.1f}%\nLess: {100*l/n:.1f}%'
    ax.text(.97,.96,note,ha='right',va='top',transform=ax.transAxes,fontsize=7,bbox={'facecolor':'white','alpha':.85,'edgecolor':'#dddddd','pad':3})
    ax.spines[['top','right']].set_visible(False)

def figure4(out):
    # Paper Figure 4, 3 systems x 6 panels. Refusal mismatches have one image only.
    fig,axes=plt.subplots(3,6,figsize=(16,8.0));details={}
    for ri,model in enumerate(MODEL_ORDER):
        raw=read_jsonl(STAGE2/f'stage2_{model}_saliency_allpairs.jsonl')
        paired=[x for x in (metrics(r) for r in raw) if x]
        d={}
        for ci,side in enumerate(['source','followup']):
            vals=[r['anchor_rank_rel'] for r in raw if r.get('single_side')==side and r.get('anchor_rank_rel') is not None]
            panel(axes[ri,ci],vals,'response',f'{side.capitalize()} generated')
            d[side]={'n':len(vals),'top30':sum(v<=.30 for v in vals),'values':vals}
        for offset,stage,title in [(2,'level2_mismatch','Safety label'),(4,'level3_drift','Semantic')]:
            sub=[r for r in paired if r['stage']==stage]
            mr=[r['delta_top_rel'] for r in sub];shifts=[r['anchor_shift'] for r in sub]
            panel(axes[ri,offset],mr,'MR',title+': MR rank');panel(axes[ri,offset+1],shifts,'shift',title+': anchor shift')
            d[stage]={'n':len(sub),'mr_top_half':sum(v<.5 for v in mr),'anchor_more':sum(v<0 for v in shifts),'anchor_less':sum(v>0 for v in shifts)}
        axes[ri,0].annotate(MODEL_NAMES[model],xy=(-.44,.5),xycoords='axes fraction',rotation=90,ha='center',va='center',fontsize=12,fontweight='bold')
        details[model]=d
    fig.tight_layout(w_pad=1.3,h_pad=2.0)
    for ext in ['pdf','png']:fig.savefig(out/f'figure_04_saliency.{ext}',dpi=220,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
    plt.close(fig)
    (out/'figure_04_metrics.json').write_text(json.dumps(details,indent=2)+'\n')

def figure5(out):
    # Paper Figure 5: all-pairs control, a different population from the audit sample.
    data=read_json(STAGE2/'stage2_control.json');y=np.arange(3)[::-1]
    fig,ax=plt.subplots(figsize=(7.2,2.7))
    for i,m in enumerate(MODEL_ORDER):
        f=data[m]['flagged']['mr_component_salient_half_pct'];u=data[m]['unflagged']['mr_component_salient_half_pct'];d=data[m]['test_mr_component_salient_half']['diff_pp']
        ax.plot([u,f],[y[i],y[i]],color='#999999',linewidth=2)
        ax.scatter([u],[y[i]],s=55,color='#2B6CB0',label='Unflagged' if i==0 else None,zorder=3)
        ax.scatter([f],[y[i]],s=55,color='#B43B3B',label='Flagged' if i==0 else None,zorder=3)
        for x in [u,f]:ax.text(x,y[i]+.18,f'{x:.1f}',ha='center',fontsize=10)
        ax.text(91,y[i],f'+{d:.1f} pp',va='center',fontsize=10)
    ax.set_yticks(y,[MODEL_NAMES[m] for m in MODEL_ORDER]);ax.set_xlim(69,96);ax.set_ylim(-.5,2.65);ax.set_xlabel('MR-modified component in the more salient half (%)')
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.25);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2,frameon=False)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(out/f'figure_05_flagged_unflagged.{ext}',dpi=220,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=REV/'figures');args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    figure4(args.output_dir);figure5(args.output_dir);print('Wrote Figure 4, Figure 5, and record-level panel counts to',args.output_dir)

if __name__=='__main__':main()
