"""Two-panel diagnostics from sealed derived loss and seed summaries only."""
from pathlib import Path
import argparse, csv, hashlib, json, os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'figure_updates_v4/.mplconfig'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SEEDS = [417,42,123,2027,3407]
BLUE='#2A7F9E'; ORANGE='#C65D3B'; INK='#263238'; GRID='#D7DFE3'

# STIXGeneral provides serif typography consistent with the manuscript.
def require_matplotlib_panel_alignment(fig, *, vertical_groups=(), tolerance_pt=1.5):
    """Fail if stacked panels do not share their left and right plot edges."""
    tolerance=tolerance_pt/fig.get_figwidth()/72
    for group in vertical_groups:
        boxes=[ax.get_position() for ax in group]
        if len(boxes)<2 or any(max(abs(a.x0-b.x0),abs(a.x1-b.x1))>tolerance for a,b in zip(boxes,boxes[1:])):
            raise ValueError('Stacked panels do not share left/right plot edges')


def require(ok, message):
    if not ok: raise ValueError(message)


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'figures/evaluation_v3')
    args=ap.parse_args()
    candidates=[ROOT/'reproducibility/derived_ten_method',ROOT/'submission_jksu_v1/reviewer_supplement/reproducibility/derived_ten_method']
    data=args.data_dir or next(p for p in candidates if (p/'mae_by_seed.csv').exists())
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    score={(r['model'],int(r['horizon_months'])):float(r['mae_pp']) for r in rows(data/'mae_by_model_horizon.csv')}
    sr=[r for r in rows(data/'mae_by_seed.csv') if r['model'] in ['BEAH','Direct'] and int(r['horizon_months'])==3]
    sm={(r['model'],int(r['seed'])):float(r['mae_pp']) for r in sr}
    require(set(sm)=={(m,s) for m in ['BEAH','Direct'] for s in SEEDS},'Invalid seed set')
    for m in ['BEAH','Direct']:
        require(abs(np.mean([sm[m,s] for s in SEEDS])-score[m,3])<1e-12,'Seed aggregate mismatch')
    loss=[r for r in rows(data/'origin_paired_losses.csv') if int(r['horizon_months'])==3 and r['model'] in ['B3','BEAH','Direct']]
    lm={(r['model'],r['origin_id']):r for r in loss}
    require(len(lm)==27,'Expected 27 method-origin losses')
    origins=sorted([r for r in loss if r['model']=='B3'],key=lambda r:r['target_reference_month'])
    require([r['target_reference_month'] for r in origins]==[f'2026-{i:02d}' for i in range(1,10)],'Unexpected target months')
    contrasts=[]
    for b in origins:
        a=lm['BEAH',b['origin_id']];d=lm['Direct',b['origin_id']]
        require(a['target_reference_month']==d['target_reference_month']==b['target_reference_month'],'Unpaired target')
        for ref,other in [('B3',b),('Direct',d)]:
            contrasts.append(dict(origin_id=b['origin_id'],target_reference_month=b['target_reference_month'],horizon_months=3,contrast='BEAH_minus_'+ref,loss_difference_pp=float(a['absolute_error_pp'])-float(other['absolute_error_pp'])))
    require(all(np.isfinite(r['loss_difference_pp']) for r in contrasts),'Invalid loss')
    for ref in ['B3','Direct']:
        rr=[r['loss_difference_pp'] for r in contrasts if r['contrast']=='BEAH_minus_'+ref]
        require(abs(np.mean(rr)-(score['BEAH',3]-score[ref,3]))<1e-12,'Mean contrast mismatch')
    mpl.rcParams.update({'font.family':'STIXGeneral','font.size':10.5,'axes.labelsize':11.0,
        'axes.titlesize':11.0,'axes.titleweight':'bold','xtick.labelsize':10.3,'ytick.labelsize':10.3,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'axes.edgecolor':GRID,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'cpi-head-diagnostics-v2'})
    fig,(ax,bx)=plt.subplots(2,1,figsize=(6.1,5.0),dpi=400,gridspec_kw={'height_ratios':[1,1.5]})
    fig.subplots_adjust(left=.15,right=.97,bottom=.12,top=.94,hspace=.58)
    jitter=[-.12,-.06,0,.06,.12]
    for s,j in zip(SEEDS,jitter):
        ax.plot([j,1+j],[sm['BEAH',s],sm['Direct',s]],color='#B7C5CD',lw=1.1,zorder=1)
    for x,m,c in [(0,'BEAH',BLUE),(1,'Direct',ORANGE)]:
        ax.scatter([x+j for j in jitter],[sm[m,s] for s in SEEDS],s=40,c=c,edgecolors='white',linewidth=.6,zorder=3)
        ax.scatter([x],[score[m,3]],marker='D',s=72,facecolors='white',edgecolors=c,linewidths=1.9,zorder=4)
    ax.axhline(score['B3',3],ls=(0,(4,3)),color=INK,lw=1.1)
    ax.text(1.30,score['B3',3]+.009,'B3',ha='right',va='bottom',fontsize=10.5)
    ax.set_xlim(-.35,1.35);ax.set_ylim(.375,.442);ax.set_yticks([.38,.40,.42,.44])
    ax.set_xticks([0,1],['BEAH','Direct']);ax.set_ylabel('h = 3 MAE (pp)')
    ax.set_title('(a) Fitting sensitivity',loc='left',pad=8,fontsize=11.0)
    ax.grid(axis='y',color=GRID,lw=.7,alpha=.6)
    ax.spines[['top','right']].set_visible(False)
    bx.axvline(0,color='#77868E',ls=(0,(3,3)),lw=1.1)
    for j,b in enumerate(origins):
        for off,ref,marker in [(-.14,'B3','o'),(.14,'Direct','^')]:
            v=next(r['loss_difference_pp'] for r in contrasts if r['origin_id']==b['origin_id'] and r['contrast']=='BEAH_minus_'+ref)
            color=BLUE if v<0 else ORANGE if v>0 else INK
            bx.plot([0,v],[j+off,j+off],color=color,alpha=.65,lw=1.4,zorder=1)
            bx.scatter([v],[j+off],s=37,marker=marker,c=color,edgecolors='white',linewidth=.5,zorder=3)
    bx.set_yticks(range(9),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'])
    bx.set_ylim(8.6,-.7);bx.set_xlim(-.33,.18);bx.set_xticks([-.3,-.15,0,.15])
    bx.set_xlabel('BEAH − reference absolute loss (pp)');bx.set_ylabel('2026 target month')
    bx.set_title('(b) Origin-paired differences',loc='left',pad=25,fontsize=11.0)
    require_matplotlib_panel_alignment(fig,vertical_groups=[[ax,bx]])
    handles=[Line2D([],[],linestyle='none',marker=mk,color=INK,markerfacecolor='white',markersize=6,label=label) for mk,label in [('o','Reference: B3'),('^','Reference: Direct')]]
    bx.legend(handles=handles,frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.005),fontsize=10.3,handletextpad=.4,columnspacing=1.0,borderaxespad=0)
    bx.grid(axis='y',color=GRID,lw=.6,alpha=.4);bx.spines[['top','right']].set_visible(False)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text():continue
        bb=t.get_window_extent(renderer)
        require(not bb.width or (bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1),'Text outside canvas: '+t.get_text())
    stem=out/'head_robustness'
    fig.savefig(stem.with_suffix('.png'),dpi=400,facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'),facecolor='white',metadata={'Author':None,'Creator':'Matplotlib','CreationDate':None,'ModDate':None})
    fig.savefig(stem.with_suffix('.svg'),facecolor='white',metadata={'Creator':'Matplotlib','Date':None})
    plt.close(fig)
    for name,records in [('head_robustness_seed_source.csv',sr),('head_robustness_pair_source.csv',contrasts)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report={'source_skill':'scibox-figure','source_sha256':{n:sha(data/n) for n in ['mae_by_seed.csv','mae_by_model_horizon.csv','origin_paired_losses.csv']},
        'script_sha256':sha(Path(__file__)),'seed_points':10,'paired_differences':18,
        'observation_unit':'nine target months; seeds are repeated fits to these same targets',
        'negative_loss_difference_favors_BEAH':True,'means_match_sealed_summaries_tolerance':1e-12,
        'minimum_native_font_pt':10.3,'text_canvas_bounds':'pass','raw_targets_read':False,
        'new_training_inference_scoring_or_tests':False,'outputs':{p.name:sha(p) for p in sorted(out.glob('head_robustness*')) if p.suffix in {'.pdf','.png','.svg','.csv'}}}
    (out/'head_robustness_handoff.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'seed_points':10,'paired_differences':18,'aggregate_checks':'pass','text_bounds':'pass'}))


if __name__=='__main__':main()
