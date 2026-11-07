"""Read saved password-v2 evaluations, compute the numbers of the password-scaling table ("Password-game
scaling and the role of the evaluation bank"), and draw its result figure.

Reads the runs under $RESULTS_ROOT (default artifacts/results/); writes password_v2_verified.json and
password_constant_forecast_verified.json to $RESULTS_ROOT/summaries/ and password_v2_errors.pdf to $FIGURES_ROOT
(default artifacts/figures/). The "source" of each row is its run directory relative to $RESULTS_ROOT.
No model is loaded or evaluated. Run on a Slurm CPU node with the project's Python environment.
"""
from pathlib import Path
import json
import math
import os
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO=Path(os.environ.get('REPO_ROOT',Path(__file__).resolve().parents[2]))
RESULTS=Path(os.environ.get('RESULTS_ROOT',REPO/'artifacts'/'results'))
SUMMARIES=RESULTS/'summaries'
OUT=Path(os.environ.get('FIGURES_ROOT',REPO/'artifacts'/'figures'))
SEEDS=[11,17,23,29,31,37,41,43,47,53]
rows=[]
for size in ['0.6b','1.7b','4b','8b']:
    for seed in (SEEDS if size=='0.6b' else SEEDS[:3]):
        tag='' if size=='0.6b' else '_qwen3_'+size.replace('.','_')
        root=f'lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296{tag}_seed{seed}_'
        suffix='20260725' if size!='8b' else {11:'besteval_pw8b_s11at89',17:'besteval_pw8b_s17at80',23:'besteval_pw8b_s23at126'}[seed]
        sources=[('ours',root+suffix+f'/seed_{seed}')]
        if size in ['4b','8b']:
            sources.append(('sft',f'lquant_40slots_nopwseed_strictv2_sft_union_qwen3_{size}_seed{seed}/seed_{seed}'))
        for method,relative in sources:
            p=RESULTS/relative
            trained=json.loads((p/'test_results.json').read_text())
            base=json.loads((p/'baseline_test_results.json').read_text())
            assert len(trained)==len(base)==7 and set(trained)==set(base)
            err=st.mean(v['forecast_sq_log10_error'] for v in trained.values())
            berr=st.mean(v['forecast_sq_log10_error'] for v in base.values())
            kl=json.loads((p/'kl_eval_results.json').read_text())
            cfg=json.loads((p/'config.json').read_text())
            rows.append(dict(size=size,seed=seed,method=method,source=relative,error=err,baseline_error=berr,fold=berr/err,
                risk_reduction_decades=st.mean((base[k]['actual_worst_log_prob']-trained[k]['actual_worst_log_prob'])/math.log(10) for k in trained),
                baseline_ceiling_error=st.mean((v['actual_worst_log_prob']/math.log(10))**2 for v in base.values()),
                trained_ceiling_error=st.mean((v['actual_worst_log_prob']/math.log(10))**2 for v in trained.values()),
                baseline_absolute_error=st.mean(v['forecast_abs_log10_error'] for v in base.values()),
                kl=kl['single_token'],kl_prompts=kl.get('n_prompts'),kl_datasets=kl.get('datasets'),
                prompt_bank=cfg.get('prompt_bank_root'),lr=cfg.get('lr'),steps=cfg.get('steps'),kl_coef=cfg.get('kl_coef'),
                checkpoint='training-selected' if size=='8b' and method=='ours' else 'final'))

def meansem(x):return dict(mean=st.mean(x),sem=st.stdev(x)/math.sqrt(len(x)),min=min(x),max=max(x))
summary={}
for size in ['0.6b','1.7b','4b','8b']:
    for method in ['ours','sft']:
        subset=[r for r in rows if r['size']==size and r['method']==method]
        if not subset:continue
        summary[f'{size}/{method}']={'n':len(subset),**{k:meansem([r[k] for r in subset]) for k in ['error','baseline_error','fold','risk_reduction_decades','kl']}}
SUMMARIES.mkdir(parents=True,exist_ok=True)
(SUMMARIES/'password_v2_verified.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2)+'\n')
controls={}
for size in ['0.6b','1.7b','4b','8b']:
    subset=[r for r in rows if r['size']==size and r['method']=='ours']
    controls[size]={'pairs':7*len(subset),'sources':[r['source'] for r in subset],
        'constant_ceiling_pretrained_error':st.mean(r['baseline_ceiling_error'] for r in subset),
        'constant_ceiling_trained_error':st.mean(r['trained_ceiling_error'] for r in subset),
        'pretrained_forecast_mean_absolute_error_decades':st.mean(r['baseline_absolute_error'] for r in subset)}
(SUMMARIES/'password_constant_forecast_verified.json').write_text(json.dumps(controls,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(5.5,2.15),layout='constrained')
for method,label,color,offset,marker in [('pretrained','Pretrained','#626262',-.22,'s'),('sft','Direct-risk SFT','#C36A19',0,'^'),('ours','Forecastability','#2369A1',.22,'o')]:
    for j,size in enumerate(['0.6b','1.7b','4b','8b']):
        subset=[r for r in rows if r['size']==size and r['method']==('ours' if method=='pretrained' else method)]
        if not subset:continue
        vals=[r['baseline_error' if method=='pretrained' else 'error'] for r in subset]
        jitter=np.linspace(-.055,.055,len(vals))
        ax.scatter(j+offset+jitter,vals,s=15,alpha=.45,marker=marker,color=color,zorder=3)
        stat=meansem(vals)
        ax.errorbar(j+offset,stat['mean'],yerr=stat['sem'],fmt=marker,color=color,markersize=5,capsize=3,zorder=4,label=label if j==(2 if method=='sft' else 0) else None)
ax.set(yscale='log',xticks=range(4),xticklabels=['0.6B ($n=10$)','1.7B ($n=3$)','4B ($n=3$)','8B ($n=3$)'],ylabel='Squared log$_{10}$ forecast error',xlabel='Qwen3 size and forecastability-training seed count',ylim=(.006,130))
ax.grid(axis='y',alpha=.16)
handles,labels=ax.get_legend_handles_labels(); order=[labels.index(x) for x in ['Pretrained','Direct-risk SFT','Forecastability']]
ax.legend([handles[i] for i in order],[labels[i] for i in order],frameon=False,ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.2))
OUT.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT/'password_v2_errors.pdf',metadata={'Author':''})
plt.close(fig)
print(json.dumps(summary,indent=2))
