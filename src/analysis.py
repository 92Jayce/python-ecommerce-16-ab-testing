"""第16讲 用户级A/B实验与不确定性
业务问题：B版是否值得采用，结果有多不确定？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

from scipy.stats import fisher_exact,chisquare
# %% 一位随机访客一行；不要拿同一个用户的多次访问当独立样本
ab=load_tables()['experiment'];assert ab.visitor_id.is_unique
summary=ab.groupby('variant').agg(n=('visitor_id','size'),success=('converted','sum'),revenue_cents=('revenue_cents','sum'))
summary['conversion']=summary.success/summary.n
summary['revenue_per_visitor_yuan']=summary.revenue_cents/summary.n/100
print(summary.to_string())
nA,nB=map(int,summary.loc[['A','B'],'n']);a,b=map(int,summary.loc[['A','B'],'success'])
pA,pB=a/nA,b/nB;delta=pB-pA
test=fisher_exact([[b,nB-b],[a,nA-a]],alternative='two-sided')
srm=chisquare([nA,nB],f_exp=[len(ab)/2,len(ab)/2])

# %% Wilson单组区间，Newcombe两组差值区间（不依赖正态Wald近似）
def wilson(x,n,z=1.959963984540054):
    p=x/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half
la,ua=wilson(a,nA);lb,ub=wilson(b,nB)
low=delta-np.sqrt((pB-lb)**2+(ua-pA)**2)
high=delta+np.sqrt((ub-pB)**2+(pA-la)**2)
result=pd.DataFrame([{'absolute_lift':delta,'relative_lift':delta/pA,'difference_95_low':low,
    'difference_95_high':high,'fisher_two_sided_p':float(test.pvalue),'sample_ratio_p':float(srm.pvalue)}])
show(result);save_table(summary.reset_index(),16,'ab_groups');save_table(result,16,'ab_inference')
fig,ax=plt.subplots(figsize=(6,3.5));rates=np.array([pA,pB]);lo=np.array([la,lb]);hi=np.array([ua,ub])
ax.errorbar([0,1],rates*100,yerr=np.vstack([(rates-lo)*100,(hi-rates)*100]),fmt='o',color=BLUE,capsize=4)
ax.set_xticks([0,1],['A','B']);ax.set(title='Synthetic randomised experiment: conversion',ylabel='Conversion (%) with Wilson 95% interval')
save_figure(fig,16,'ab_conversion')
print('演示独立模拟随机试验；预先指定主指标，不在看到p值后随意停止或筛选多个显著指标。还需核查成本、护栏与分流。')
