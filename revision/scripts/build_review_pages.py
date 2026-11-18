#!/usr/bin/env python3
"""Build portable read-only audit pages and the Figure 3 example panel.

Paper: Table VIII/Section V-B flagged pairs; Section VI classifier audit;
Table III MR validation; Figure 3 same-label semantic examples.
Images are shared local previews; original PNG identifiers and hashes are retained.
"""
import html
import json
from pathlib import Path
from package_paths import REV,AUDIT,MR_VALIDATION,read_csv,read_json

def esc(v):return html.escape(str(v))
STYLE='''body{font:15px/1.6 Arial,sans-serif;max-width:1240px;margin:30px auto;padding:0 20px;color:#182936}h1{font-size:27px}h2{font-size:20px}a{color:#164c76}.card{border:1px solid #ccd5de;border-radius:8px;margin:22px 0;padding:20px;break-inside:avoid}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}.pair img,.single img{display:block;max-width:100%;height:auto;max-height:480px;margin:auto}.text{white-space:pre-wrap;overflow-wrap:anywhere}.labels{background:#eef3f7;padding:10px 14px;margin:12px 0}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #d3dce5;text-align:left}summary{cursor:pointer;font-weight:bold}code{overflow-wrap:anywhere}@media(max-width:700px){.pair{grid-template-columns:1fr}}'''

def page(title,body):return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+STYLE+'</style><h1>'+esc(title)+'</h1><nav><a href="../index.html">Revision evidence index</a></nav>'+body+'</html>'

def main():
    index=read_json(AUDIT/'image_index.json')
    def image(raw,caption):
        d=index.get(raw)
        if d:return '<a href="../'+esc(d['preview'])+'"><img loading="lazy" src="../'+esc(d['preview'])+'" alt="'+esc(caption)+'"></a>'
        return '<p>No generated image for this response. See the recorded response evidence.</p>'
    rows=read_csv(AUDIT/'flagged_pairs_final.csv');cards=[]
    for r in rows:
        sides=[]
        for s in ['source','followup']:
            visual = '<p>No generated image for the recorded response.</p>' if r[s+'_blocked'].strip().lower() in ['true','1'] else image(r[s+'_original'],s)
            sides.append('<section><h3>'+s.capitalize()+'</h3>'+visual+'<p class="text">'+esc(r[s+'_prompt'])+'</p></section>')
        cards.append('<article class="card"><h2>'+esc(r['pair_id'])+'</h2><p>'+esc(r['model'])+' · '+esc(r['trigger'])+' · '+esc(r['prompt_type'])+'</p><div class="labels">A: '+esc(r['A_judgment'])+' | B: '+esc(r['B_judgment'])+' | Final: '+esc(r['final_judgment'])+'</div><div class="pair">'+''.join(sides)+'</div></article>')
    (AUDIT/'flagged_pairs_review.html').write_text(page('Flagged-pair audit: 300 cases','<p>Table VIII. Final labels and original independent A/B labels are shown separately.</p>'+''.join(cards)))
    cards=[]
    for r in read_csv(AUDIT/'classifier_images_final_200.csv'):
        cards.append('<article class="card"><h2>'+esc(Path(r['image']).name)+'</h2><p>'+esc(r['model'])+' · '+esc(r['category'])+' · '+esc(r['prompt_type'])+'</p><div class="labels">A: '+esc(r['A_unsafe'])+' | B: '+esc(r['B_unsafe'])+' | Final: '+esc(r['final_unsafe'])+'</div><div class="single">'+image(r['image'],'Classifier audit image')+'</div></article>')
    (AUDIT/'classifier_images_review.html').write_text(page('Classifier audit: 200 images','<p>Section VI. 1 = unsafe; 0 = safe. The corresponding classifier labels are retained in the CSV.</p>'+''.join(cards)))
    aa={(r['mr_id'],r['prompt_index']):r for r in read_csv(MR_VALIDATION/'mr_annotations_A.csv')};bb={(r['mr_id'],r['prompt_index']):r for r in read_csv(MR_VALIDATION/'mr_annotations_B.csv')};cards=[]
    for r in read_csv(MR_VALIDATION/'mr_adjudicated.csv'):
        k=(r['mr_id'],r['prompt_index']);a=aa[k];b=bb[k]
        label=lambda z:esc(z['semantics_preserved'])+' / '+esc(z['safety_intent_preserved'])
        cards.append('<article class="card"><h2>'+esc(r['mr'])+' · prompt '+esc(r['prompt_index'])+'</h2><p>'+esc(r['prompt_type'])+'</p><div class="labels">Scene / Policy — A: '+label(a)+' | B: '+label(b)+' | Final: '+label(r)+'</div><div class="pair"><section><h3>Source</h3><p class="text">'+esc(r['source_prompt'])+'</p></section><section><h3>Follow-up</h3><p class="text">'+esc(r['followup_prompt'])+'</p></section></div></article>')
    (MR_VALIDATION/'mr_review.html').write_text(page('MR validation: 240 pairs','<p>Table III. All agreed A/B labels are retained. Only the three disputed fields are resolved; <a href="mr_adjudication_review.html">view the three decisions</a>.</p>'+''.join(cards)))
    sample={(r['mr_id'],r['prompt_index']):r for r in read_csv(MR_VALIDATION/'mr_sample.csv')};cards=[]
    for r in read_csv(MR_VALIDATION/'mr_adjudication_decisions.csv'):
        z=sample[(r['mr_id'],r['prompt_index'])]
        cards.append('<article class="card"><h2>'+esc(z['mr'])+' / '+esc(r['prompt_index'])+'</h2><div class="labels">'+esc(r['field'])+' - A: '+esc(r['A'])+' | B: '+esc(r['B'])+' | Final: '+esc(r['final'])+'</div><div class="pair"><section><h3>Source prompt</h3><p class="text">'+esc(z['source_prompt'])+'</p></section><section><h3>Follow-up prompt</h3><p class="text">'+esc(z['followup_prompt'])+'</p></section></div><p>'+esc(r['reason'])+'</p></article>')
    (MR_VALIDATION/'mr_adjudication_review.html').write_text(page('MR adjudication: three disputed fields','<p>All 477 agreed field labels are retained. These decisions resolve only the three A/B disagreements.</p>'+''.join(cards)))
    rows=read_csv(AUDIT/'figure_03_examples.csv');body=['<table><thead><tr><th>Source prompt and image</th><th>Follow-up prompt and image</th><th>Semantic distance</th></tr></thead><tbody>']
    for r in rows:
        body.append('<tr><td><p class="text">'+esc(r['source_prompt'])+'</p><img width="260" src="../'+esc(r['source_image'])+'"></td><td><p class="text">'+esc(r['followup_prompt'])+'</p><img width="260" src="../'+esc(r['followup_image'])+'"></td><td>'+esc(r['semantic_distance'])+'<p><code>'+esc(r['pair_id'])+'</code></p></td></tr>')
    body.append('</tbody></table>')
    (REV/'figures/figure_03_examples.html').write_text(page('Figure 3: same-label semantic examples',''.join(body)))
    links=[('PAPER_MAP.md','Paper-to-artifact map'),('EXPERIMENTS.md','Experiment settings and query dates'),('README.md','Revision README'),('RUNME.md','Reproduction commands'),('tables/paper_results.md','Recomputed numerical results'),('03_manual_audit/flagged_pairs_review.html','Review the 300 flagged pairs'),('03_manual_audit/classifier_images_review.html','Review the 200 classifier images'),('04_mr_validation/mr_review.html','Review the 240 MR pairs'),('04_mr_validation/mr_adjudication_review.html','Review the three MR decisions'),('figures/figure_03_examples.html','Figure 3 examples'),('figures/figure_04_saliency.pdf','Figure 4 corrected saliency panels'),('figures/figure_05_flagged_unflagged.pdf','Figure 5 flagged/unflagged control')]
    (REV/'index.html').write_text(page('SafeMT revision evidence','<ul>'+''.join('<li><a href="'+p+'">'+esc(t)+'</a></li>' for p,t in links)+'</ul>').replace('<nav><a href="../index.html">Revision evidence index</a></nav>',''))
    print('Built 300-pair, 200-image, 240-MR review pages and Figure 3 example page.')

if __name__=='__main__':main()
