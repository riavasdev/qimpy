import numpy as np
import torch 
import math

def compute_shape_function(self, nC,rMin,rMax,eps,n: FieldR) -> FieldR):
    s=torch.ones_like(nC)
    logD=math.log(rMax/rMin)
    f=np.log(rMax/nC)/logD
    t=f-torch.sin(2*torch.pi*f)/(2*torch.pi)
    s=(eps**t-1)/(eps-1)
    s[nC>=rMax]=0 
    s[nC<=rMin]=1
    return s
  
def shape_gradient(nC,gs,rMin,rMax,eps):
    g=torch.zeros_like(nC)
    m=(nC>rMin)&(nC<rMax)
    logD=math.log(rMax/rMin)
    f=torch.log(rMax/nC[m])/logD
    fr=-1/(nC[m]*logD)
    t=f-torch.sin(2*np.pi*f)/(2*np.pi)
    tf=1-torch.cos(2*np.pi*f)
    st=torch.log(eps)*eps**t/(eps-1)
    g[m]+=gs[m]*st*tf*fr
    return g
