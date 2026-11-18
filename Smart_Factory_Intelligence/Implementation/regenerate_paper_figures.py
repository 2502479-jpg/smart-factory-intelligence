"""Regenerates the paper figures and CSV tables from the EXACT RECORDED READINGS of the paper
(hard-coded below). Not produced by the pipeline: usage  python regenerate_paper_figures.py <out_dir>"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.edgecolor':'black'})
import os,sys,csv
OUT=(sys.argv[1] if len(sys.argv)>1 else 'recorded_figures')+'/'; os.makedirs(OUT,exist_ok=True)
DPI=300
def save(fig,n): fig.savefig(OUT+n,dpi=DPI,bbox_inches='tight',facecolor='white'); plt.close(fig)
checks=[]
def chk(name,ax_patches,expected,horizontal=False):
    got=[(p.get_width() if horizontal else p.get_height()) for p in ax_patches]
    assert np.allclose(got,expected,atol=1e-12),(name,got,expected)
    checks.append(name)

# ---------- data copied verbatim from the request ----------
M=['Precision','Recall','F1','ROC-AUC','PR-AUC','MCC']
perf={'Decision Tree':[0.159124,0.105620,0.126966,0.622738,0.035159,0.125512],
      'Random Forest':[0.213018,0.104651,0.140351,0.676224,0.043555,0.145865],
      'Boost':[0.190955,0.147287,0.166302,0.691164,0.049682,0.163442]}
thr={'Decision Tree':0.95,'Random Forest':0.58,'Boost':0.86}
cms={'Decision Tree':dict(TN=175955,FP=576,FN=923,TP=109),'Random Forest':dict(TN=176132,FP=399,FN=924,TP=108),'Boost':dict(TN=175887,FP=644,FN=880,TP=152)}
shap=[('L3_S29_F3407',0.127604),('L3_S33_F3865',0.118540),('L3_S30_F3704',0.092527),('L3_S29_F3351',0.082631),('L3_S29_F3339',0.080227),('L3_S33_F3857',0.072299),('L3_S33_F3859',0.061282),('L3_S30_F3744',0.057796),('L3_S29_F3412',0.047449),('L3_S29_F3464',0.041618),('L3_S29_F3473',0.039556),('L3_S33_F3873',0.037554),('L3_S30_F3759',0.035814),('L3_S30_F3769',0.033945),('L3_S29_F3436',0.032847),('L3_S30_F3809',0.031976),('L3_S29_F3479',0.031065),('L0_S0_F20',0.030494),('L3_S30_F3609',0.030116),('L3_S33_F3855',0.030043)]
drift=[('L3_S29_F3464',0.522319,26.246473),('L3_S29_F3407',0.916732,11.127972),('L3_S29_F3412',0.916732,11.127972),('L3_S29_F3473',0.575427,4.172669),('L0_S0_F20',0.314287,3.823815),('L3_S30_F3809',0.674176,3.182212),('L3_S30_F3759',0.400506,3.164884),('L3_S30_F3769',0.697330,2.929783),('L3_S30_F3704',0.448953,1.993230),('L3_S29_F3479',0.454320,1.857559),('L3_S29_F3351',0.475132,1.822685),('L3_S30_F3609',0.470210,1.783244),('L3_S29_F3436',0.462007,1.763511),('L3_S30_F3744',0.510505,1.204917),('L3_S29_F3339',0.388517,1.083390),('L3_S33_F3865',0.428962,0.938931),('L3_S33_F3857',0.384458,0.918158),('L3_S33_F3859',0.384757,0.901714),('L3_S33_F3855',0.468771,0.677469),('L3_S33_F3873',0.994225,0.000429)]
shift={'Accuracy':(0.991417,0.993416),'Precision':(0.190955,0.190045),'Recall':(0.147287,0.040698),'F1':(0.166302,0.067039),'ROC-AUC':(0.691164,0.653914),'PR-AUC':(0.049682,0.036213),'MCC':(0.163442,0.085558)}
retr={'Precision':(0.190045,0.190955),'Recall':(0.040698,0.147287),'F1':(0.067039,0.166302),'ROC-AUC':(0.653914,0.691164),'PR-AUC':(0.036213,0.049681),'MCC':(0.085558,0.163442)}

def two_panel(name,fn,data,labels,colors,left,right,lylim,rylim,figsize):
    fig,(a,b)=plt.subplots(1,2,figsize=figsize,gridspec_kw={'width_ratios':[len(left),max(len(right),1)*1.15]})
    k=len(labels)
    for ax,mets,yl,t in ((a,left,lylim,'(a) Scale 0\u20130.25' if lylim[1]<0.5 else '(a)'),(b,right,rylim,'(b) Scale 0\u20131')):
        x=np.arange(len(mets)); w=0.8/k
        for i,(lab,c) in enumerate(zip(labels,colors)):
            vals=[data[m][i] for m in mets]
            bars=ax.bar(x+(i-(k-1)/2)*w,vals,w,label=lab,color=c,edgecolor='black',linewidth=0.4)
            chk(f'{name}:{lab}:{",".join(mets)}',ax.patches[-len(mets):],vals)
            for r,v in zip(bars,vals): ax.text(r.get_x()+r.get_width()/2,v+yl[1]*0.012,f'{v:.4f}',ha='center',va='bottom',fontsize=6,rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(mets); ax.set_ylim(*yl); ax.set_title(t,fontsize=9); ax.yaxis.grid(True,alpha=.3,linewidth=.5); ax.set_axisbelow(True)
    a.set_ylabel('Score')
    h,l=a.get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=k,frameon=False,bbox_to_anchor=(0.5,-0.02))
    fig.tight_layout(rect=(0,0.05,1,1)); save(fig,fn)

# Fig 1
two_panel('fig1',  'Fig01_Model_Performance_Comparison.png',
  {m:[perf[k][j] for k in perf] for j,m in enumerate(M)}, list(perf), ['#bdbdbd','#6baed6','#08306b'],
  ['Precision','Recall','F1','PR-AUC','MCC'],['ROC-AUC'],(0,0.26),(0,0.8),(6.5,3.6))
# Fig 7 shifted (document order)
two_panel('fig7','Fig07_Original_vs_Shifted_Performance.png',shift,['Original test','Shifted test'],['#0072B2','#D55E00'],
  ['Precision','Recall','F1','PR-AUC','MCC'],['Accuracy','ROC-AUC'],(0,0.26),(0,1.15),(6.5,3.6))
# Fig 8 retrain
two_panel('fig8','Fig08_Original_vs_Retrained_Performance.png',retr,['Original model on shifted test','Retrained model on shifted test'],['#D55E00','#009E73'],
  ['Precision','Recall','F1','PR-AUC','MCC'],['ROC-AUC'],(0,0.26),(0,0.85),(6.5,3.6))

# Confusion matrices
for fn,name in (('Fig02_DecisionTree_Confusion_Matrix.png','Decision Tree'),('Fig03_RandomForest_Confusion_Matrix.png','Random Forest'),('Fig04_Boost_Confusion_Matrix.png','Boost')):
    c=cms[name]; v=np.array([[c['TN'],c['FP']],[c['FN'],c['TP']]])
    fig,ax=plt.subplots(figsize=(3.5,3.1))
    ax.imshow(v,cmap='Blues',norm=LogNorm(vmin=50,vmax=2e5))
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{int(v[i,j])}',ha='center',va='center',fontsize=11,fontweight='bold',color='white' if v[i,j]>3000 else 'black')
    lab=['Non-defective (0)','Defective (1)']
    ax.set_xticks([0,1]); ax.set_xticklabels(lab); ax.set_yticks([0,1]); ax.set_yticklabels(lab,rotation=90,va='center')
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); ax.set_title(f'{name} (threshold {thr[name]})',fontsize=9.5)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.6)
    ax.tick_params(length=0); save(fig,fn)
    checks.append('cm:'+name)

# SHAP 20
f=[a for a,_ in shap]; s=[b for _,b in shap]; assert s==sorted(s,reverse=True)
fig,ax=plt.subplots(figsize=(6.5,5.4))
bars=ax.barh(f[::-1],s[::-1],color='#4a6fa5',edgecolor='black',linewidth=.4); chk('shap',ax.patches,s[::-1],True)
for r,v in zip(bars,s[::-1]): ax.text(v+0.0015,r.get_y()+r.get_height()/2,f'{v:.6f}',va='center',fontsize=7)
ax.set_xlim(0,0.15); ax.set_xlabel('Mean absolute SHAP value'); ax.set_title('Top 20 Features by Mean Absolute SHAP Value',fontsize=10.5,fontweight='bold')
ax.xaxis.grid(True,alpha=.3,linewidth=.5); ax.set_axisbelow(True); ax.tick_params(axis='y',labelsize=8)
save(fig,'Fig05_SHAP_Top20_Features.png')

# Drift
fn=[d[0] for d in drift]; ks=[d[1] for d in drift]; psi=[d[2] for d in drift]
fig,(b,a)=plt.subplots(1,2,figsize=(6.5,6.6),sharey=True,gridspec_kw={'width_ratios':[1,1.15],'wspace':0.05})
y=np.arange(len(fn))[::-1]
b.barh(y,ks,color='#0072B2',edgecolor='black',linewidth=.4); chk('ks',b.patches,ks,True)
for yy,v in zip(y,ks): b.text(v+0.015,yy,f'{v:.6f}',va='center',fontsize=6)
b.set_xlim(0,1.28); b.set_xticks([0,.25,.5,.75,1.0]); b.set_xlabel('KS statistic (linear axis)'); b.set_title('(a) Kolmogorov\u2013Smirnov statistic',fontsize=9,pad=14)
a.barh(y,psi,color='#D55E00',edgecolor='black',linewidth=.4); chk('psi',a.patches,psi,True)
a.set_xscale('log'); a.set_xlim(1e-4,2e3)
for yy,v in zip(y,psi): a.text(v*1.25,yy,f'{v:.6f}',va='center',fontsize=6)
a.axvline(0.10,color='black',linestyle=':',linewidth=1.0,label='PSI 0.10 (heuristic)'); a.axvline(0.25,color='black',linestyle='--',linewidth=1.0,label='PSI 0.25 (heuristic)')
a.set_xlabel('PSI (logarithmic axis)'); a.set_title('(b) Population Stability Index',fontsize=9,pad=14)
a.legend(loc='lower right',fontsize=6.5,frameon=True,edgecolor='black',fancybox=False)
b.set_yticks(y); b.set_yticklabels(fn,fontsize=7.5); a.tick_params(axis='y',length=0)
for ax in (a,b): ax.xaxis.grid(True,alpha=.3,linewidth=.5); ax.set_axisbelow(True); ax.set_ylim(-0.7,len(fn)-0.3+0.2)
fig.subplots_adjust(left=0.2,right=0.98,top=0.93,bottom=0.08); save(fig,'Fig06_Feature_Level_Drift_PSI_KS.png')

# ---- individual per-metric charts (ZIP only) ----
import os; os.makedirs(OUT+'individual_metric_charts',exist_ok=True)
def single(fn,title,labels,vals,colors,ylab='Score'):
    fig,ax=plt.subplots(figsize=(3.6,3.0)); x=np.arange(len(labels))
    bars=ax.bar(x,vals,0.6,color=colors,edgecolor='black',linewidth=.5); chk(fn,ax.patches,vals)
    top=max(vals)*1.18; ax.set_ylim(0,top)
    for r,v in zip(bars,vals): ax.text(r.get_x()+r.get_width()/2,v+top*0.015,f'{v:.6f}',ha='center',va='bottom',fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8); ax.set_ylabel(ylab); ax.set_title(title,fontsize=10,fontweight='bold')
    ax.yaxis.grid(True,alpha=.3,linewidth=.5); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT+'individual_metric_charts/'+fn,dpi=DPI,facecolor='white'); plt.close(fig)
mc=['#bdbdbd','#6baed6','#08306b']
for j,m in enumerate(M):
    single(f'Model_{m.replace("-","")}_Comparison.png',f'{m}: Model Comparison',list(perf),[perf[k][j] for k in perf],mc)
for m in ['Recall','F1','ROC-AUC','PR-AUC','MCC']:
    single(f'OrigVsShifted_{m.replace("-","")}.png',f'{m}: Original vs Shifted Test',['Original','Shifted'],list(shift[m]),['#0072B2','#D55E00'])
    single(f'OrigVsRetrained_{m.replace("-","")}.png',f'{m}: Original vs Retrained (Shifted Test)',['Original','Retrained'],list(retr[m]),['#D55E00','#009E73'])
print(len(checks),'checks passed')
import pandas as pd
T=OUT+'tables/'; os.makedirs(T,exist_ok=True)
pd.DataFrame(perf,index=M).T.assign(threshold=pd.Series(thr)).to_csv(T+'recorded_model_performance.csv')
pd.DataFrame(cms).T.to_csv(T+'recorded_confusion_matrices.csv')
pd.DataFrame(shap,columns=['feature','mean_abs_shap']).to_csv(T+'recorded_shap_top20.csv',index=False)
pd.DataFrame(drift,columns=['feature','ks_statistic','psi']).to_csv(T+'recorded_feature_drift.csv',index=False)
pd.DataFrame(shift,index=['original_test','shifted_test']).T.to_csv(T+'recorded_original_vs_shifted.csv')
pd.DataFrame(retr,index=['original_model_on_shifted_test','retrained_model_on_shifted_test']).T.to_csv(T+'recorded_original_vs_retrained.csv')

