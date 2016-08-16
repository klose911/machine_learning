import scipy as sp
data = sp.genfromtxt("web_traffic.tsv", delimiter="\t") 

x = data[:,0]
y = data[:,1] 

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)] 





def error(f, x, y):
    return sp.sum((f(x)-y)**2)  

# polyfit() function returns the parameters of the fitted model function fp1 
# by setting full to True, we also get additional background information on the fitting process.
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True) 
print("Model parameters: %s" % fp1) # Model parameters: [   2.59619213  989.02487106] 
f1 = sp.poly1d(fp1)  
#print(error(f1, x, y)) # 317389767.34
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting

f2p = sp.polyfit(x, y, 2) 
f2 = sp.poly1d(f2p)  

f10p = sp.polyfit(x, y, 100) 
f10 = sp.poly1d(f10p) 
#print(error(f2, x, y)) #179983507.878

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()

plt.plot(fx, f1(fx), linewidth=4)
plt.plot(fx, f2(fx), linewidth=4)
plt.plot(fx, f10(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left") 
plt.legend(["d=%i" % f2.order], loc="upper left") 
plt.show()

