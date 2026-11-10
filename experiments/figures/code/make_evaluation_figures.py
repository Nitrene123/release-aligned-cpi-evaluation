"""Visualize sealed retrospective summaries; no model or target access."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'figure_updates_v4/.mplconfig'))
import argparse,csv,hashlib,json,math,platform
from datetime import datetime,timezone
from collections import defaultdict
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

ORDER=[('No_change','No-change'),('Seasonal_naive','Seasonal-naive'),('AR1','AR(1)'),('SARIMA_101x101_12','SARIMA'),('TimesFM3_target_only','TFM-3 target-only'),('B3','TFM-3 multivariate'),('BEAH','BEAH'),('Direct','Direct'),('Fixed_candidate','Candidate'),('Fixed_candidate_gate','Candidate + gate')]
H=[1,3,6,12]
BLUE='#2A7F9E'; ORANGE='#C65D3B'; INK='#263238'; GOLD='#C47B4B'; GRID='#D7DFE3'

# STIXGeneral provides serif typography consistent with the manuscript.
def require_matplotlib_panel_alignment(fig, *, horizontal_groups=(), vertical_groups=(), tolerance_pt=1.5):
    """Fail if comparable Matplotlib panel plot areas do not share edges."""
    dx=tolerance_pt/fig.get_figwidth()/72
    dy=tolerance_pt/fig.get_figheight()/72
    for group in horizontal_groups:
        boxes=[ax.get_position() for ax in group]
        if len(boxes)<2 or any(max(abs(a.y0-b.y0),abs(a.y1-b.y1))>dy for a,b in zip(boxes,boxes[1:])):
            raise ValueError('Comparable panels do not share bottom/top plot edges')
    for group in vertical_groups:
        boxes=[ax.get_position() for ax in group]
        if len(boxes)<2 or any(max(abs(a.x0-b.x0),abs(a.x1-b.x1))>dx for a,b in zip(boxes,boxes[1:])):
            raise ValueError('Comparable panels do not share left/right plot edges')

def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write_csv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,path):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text(): continue
        bb=text.get_window_extent(renderer)
        if bb.width and bb.height and (bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1): outside.append(text.get_text())
    require(not outside,'Text outside figure canvas: '+repr(outside))
    fig.savefig(path.with_suffix('.png'),dpi=400,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),facecolor='white',metadata={'Author':None,'Creator':'Matplotlib','CreationDate':None,'ModDate':None})
    fig.savefig(path.with_suffix('.svg'),facecolor='white',metadata={'Date':None,'Creator':'Matplotlib'})
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'figures/evaluation_v3')
    args=parser.parse_args()
    candidates=[ROOT/'reproducibility/derived_ten_method',ROOT/'submission_jksu_v1/reviewer_supplement/reproducibility/derived_ten_method']
    data=args.data_dir or next((p for p in candidates if (p/'mae_by_model_horizon.csv').exists()),None)
    require(data is not None,'Supply the sealed derived ten-method data directory')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    score_file=data/'mae_by_model_horizon.csv';loss_file=data/'origin_paired_losses.csv'
    scores=read_csv(score_file);losses=read_csv(loss_file)
    require(len(scores)==40 and len(losses)==360,'Unexpected sealed score sizes')
    sm={(r['model'],int(r['horizon_months'])):r for r in scores}
    require(len(sm)==40 and set(sm)=={(m,h) for m,_ in ORDER for h in H},'Incomplete score matrix')
    matrix=np.array([[float(sm[m,h]['mae_pp']) for h in H] for m,_ in ORDER])
    require(np.isfinite(matrix).all() and (matrix>=0).all(),'Invalid MAE cell')
    require(all(int(r['n_origins'])==9 for r in scores),'Different score support')
    base=np.array([float(sm['B3',h]['mae_pp']) for h in H])
    delta=100*(matrix/base-1)
    # Symmetric linear range includes every cell; no clipping or clustering.
    extent=math.ceil(float(np.abs(delta).max())/10)*10
    mpl.rcParams.update({'font.family':'STIXGeneral','font.size':10.5,'axes.labelsize':11.0,'axes.titlesize':11.0,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'axes.edgecolor':INK,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'cpi-evaluation-figures-v3','axes.unicode_minus':True})
    fig=plt.figure(figsize=(5.75,4.00),dpi=400)
    ax=fig.add_axes([.29,.06,.54,.82])
    norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent)
    im=ax.imshow(delta,aspect='auto',cmap='RdBu_r',norm=norm,interpolation='nearest')
    ax.set_xticks(range(4),[str(h) for h in H]);ax.xaxis.tick_top();ax.xaxis.set_label_position('top')
    ax.set_xlabel('Horizon (reference months)',labelpad=10)
    ax.set_yticks(range(10),[label for _,label in ORDER]);ax.tick_params(axis='both',length=0,pad=7)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False,top=False)
    for split in [1.5,3.5,5.5,7.5]: ax.axhline(split,color=INK,lw=1.1)
    ax.add_patch(Rectangle((.5,-.5),1,10,fill=False,edgecolor=GOLD,lw=2.0,zorder=5))
    for i in range(10):
        for j in range(4):
            rgba=im.cmap(norm(delta[i,j]));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=10.5,color='white' if lum<.49 else INK)
    for s in ax.spines.values():s.set_visible(False)
    cbax=fig.add_axes([.85,.06,.025,.82]);cb=fig.colorbar(im,cax=cbax,orientation='vertical',ticks=[-200,-100,0,100,200])
    cb.set_label('MAE change relative to B3 (%)',labelpad=9,fontsize=11.0);cb.ax.tick_params(labelsize=10.2,length=3,pad=3);cb.outline.set_visible(False)
    save(fig,out/'ten_method_heatmap')
    write_csv(out/'ten_method_heatmap_source.csv',[dict(model=m,horizon_months=h,mae_pp=float(matrix[i,j]),relative_MAE_change_vs_B3_percent=float(delta[i,j]),n_origins=9) for i,(m,_) in enumerate(ORDER) for j,h in enumerate(H)])

    lm={(r['model'],r['origin_id'],int(r['horizon_months'])):r for r in losses}
    require(len(lm)==360,'Duplicate origin-method losses')
    plot_rows=[];by_h={}
    for h in H:
        brr=sorted([r for r in losses if r['model']=='B3' and int(r['horizon_months'])==h],key=lambda r:r['target_reference_month'])
        trr=[r for r in losses if r['model']=='TimesFM3_target_only' and int(r['horizon_months'])==h]
        require(len(brr)==9 and {r['origin_id'] for r in trr}=={r['origin_id'] for r in brr},'Input configurations have different support')
        rr=[]
        for b in brr:
            t=lm['TimesFM3_target_only',b['origin_id'],h]
            require(t['target_reference_month']==b['target_reference_month'] and t['origin_reference_month']==b['origin_reference_month'],'Unpaired origin/target metadata')
            difference=float(t['absolute_error_pp'])-float(b['absolute_error_pp'])
            r=dict(origin_id=b['origin_id'],origin_reference_month=b['origin_reference_month'],target_reference_month=b['target_reference_month'],horizon_months=h,B3_absolute_loss_pp=float(b['absolute_error_pp']),target_only_absolute_loss_pp=float(t['absolute_error_pp']),target_only_minus_B3_loss_pp=difference)
            rr.append(r);plot_rows.append(r)
        mean=np.mean([r['target_only_minus_B3_loss_pp'] for r in rr])
        expected=float(sm['TimesFM3_target_only',h]['mae_pp'])-float(sm['B3',h]['mae_pp'])
        require(abs(mean-expected)<1e-12,'Mean paired difference does not match sealed MAEs')
        by_h[h]=rr
    bound=max(abs(r['target_only_minus_B3_loss_pp']) for r in plot_rows)
    lim=math.ceil(bound/0.05)*.05+.025
    fig,axes=plt.subplots(2,2,figsize=(6.1,4.9),dpi=400,sharex=True,sharey=True)
    fig.subplots_adjust(left=.14,right=.98,bottom=.14,top=.94,wspace=.30,hspace=.36)
    month_labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']
    for k,(ax,h) in enumerate(zip(axes.flat,H)):
        values=np.array([r['target_only_minus_B3_loss_pp'] for r in by_h[h]])
        colors=[BLUE if v<0 else ORANGE if v>0 else INK for v in values]
        y=np.arange(9)
        ax.axvline(0,color='#7B878D',ls=(0,(3,3)),lw=1.15,zorder=0)
        for yy,v,c in zip(y,values,colors): ax.plot([0,v],[yy,yy],color=c,lw=1.6,alpha=.8,zorder=2)
        ax.scatter(values,y,c=colors,s=37,edgecolor='white',linewidth=.5,zorder=3)
        ax.axhline(8.85,color=GRID,lw=.8)
        ax.scatter([values.mean()],[9.7],marker='D',s=46,facecolor='white',edgecolor=INK,linewidth=1.3,zorder=4)
        ax.set_yticks(list(y)+[9.7],month_labels+['Mean']);ax.tick_params(axis='y',labelleft=True,length=0,pad=4)
        ax.set_ylim(10.35,-.6);ax.set_xlim(-lim,lim)
        ax.set_xticks([-.5,0,.5] if lim>.4 else [-.2,0,.2]);ax.tick_params(axis='x',labelbottom=True,length=3,pad=3)
        ax.grid(axis='y',color=GRID,alpha=.3,lw=.55)
        ax.set_title(f'({chr(97+k)})  h = {h}',loc='left',pad=8,fontweight='bold',fontsize=11.0)
        for spine in ax.spines.values(): spine.set_color(GOLD if h==3 else GRID);spine.set_linewidth(1.5 if h==3 else .8)
    fig.supylabel('2026 target month',x=.018,fontsize=11.0)
    fig.supxlabel('Target-only − multivariate absolute loss (pp)',y=.025,fontsize=10.5)
    require_matplotlib_panel_alignment(fig,
        horizontal_groups=[axes[0,:],axes[1,:]],
        vertical_groups=[axes[:,0],axes[:,1]])
    save(fig,out/'covariate_paired_losses')
    write_csv(out/'covariate_paired_losses_source.csv',plot_rows)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_skill':'scibox-figure','data_scope':'sealed retrospective ten-method derived score and loss CSVs only','input_sha256':{score_file.name:sha(score_file),loss_file.name:sha(loss_file)},'script_sha256':sha(Path(__file__)),'environment':{'python':platform.python_version(),'matplotlib':mpl.__version__,'numpy':np.__version__},'typography':{'target_body_text_pt':12.0,'figure_text_native_pt':{'base':10.5,'axis_labels':11.0,'ticks':10.5,'cell_labels':10.5},'pdf_font_embedding':'TrueType'},'heatmap':{'cells':40,'fixed_method_order':[m for m,_ in ORDER],'cell_labels':'MAE in pp, rounded to three decimals for display only','color':'100*(MAE_method/MAE_B3-1), symmetric linear scale','color_limits_percent':[-extent,extent],'clipping':False,'minimum_fontsize_native_pt':10.2},'paired_plot':{'origin_pairs':36,'per_horizon':9,'estimand':'target-only absolute loss minus B3 absolute loss within matching origin/horizon','summary_diamond':'mean of nine paired losses, not an additional observation','shared_x_limits_pp':[-lim,lim],'mean_differences_match_sealed_MAE_differences_within':1e-12,'minimum_fontsize_native_pt':10.5},'validation':{'all_40_MAEs_retained':True,'same_origin_and_target_pairing':True,'finite_values':True,'text_canvas_bounds':'pass','target_values_read':False,'training':False,'inference':False,'new_target_scoring':False,'new_statistical_tests':False},'outputs':{p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ['.pdf','.svg','.png','.csv']}}
    (out/'visual_asset_handoff.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'FIGURES_CREATED_FROM_SEALED_DERIVED_RESULTS','heatmap_cells':40,'paired_points':36,'paired_x_limits':[-lim,lim],'text_bounds':'pass','figure_text_target_pt':'10.2-11.0 before manuscript scaling'},indent=2))
if __name__=='__main__':main()
