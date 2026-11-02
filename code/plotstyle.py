"""Local publication plotting defaults; replaces missing external style helper."""
import matplotlib.pyplot as plt
COL1=3.4
COL2=7.0
PALETTE=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
MARKERS=['o','s','^','D','v','P']
LINES=['-','--','-.',':']
def use_paper_style():
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
def save(fig,path):
 fig.savefig(path+'.pdf',bbox_inches='tight');plt.close(fig)
