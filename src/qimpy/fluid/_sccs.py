import numpy as np, math
def compute_calc(nC,rMin,rMax,eps):
    s=np.ones_like(nC)
    logD=math.log(rMax/rMin)
    f=np.log(rMax/nC)/logD
    t=f-np.sin(2*np.pi*f)/(2*np.pi)
    s=(eps**t-1)/(eps-1)
    s[nC>=rMax]=0 
    s[nC<=rMin]=1
    return s
  
def propagateGradient_calc(nC,gs,rMin,rMax,eps):
    g=np.zeros_like(nC)
    m=(nC>rMin)&(nC<rMax)
    logD=math.log(rMax/rMin)
    f=np.log(rMax/nC[m])/logD
    fr=-1/(nC[m]*logD)
    t=f-np.sin(2*np.pi*f)/(2*np.pi)
    tf=1-np.cos(2*np.pi*f)
    st=np.log(eps)*eps**t/(eps-1)
    g[m]+=gs[m]*st*tf*fr
    return g
