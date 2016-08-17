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
fx = sp.linspace(0, x[-1], 1000) # generate X-values for plotting

f2p = sp.polyfit(x, y, 2) 
f2 = sp.poly1d(f2p)  

f10p = sp.polyfit(x, y, 100) 
f10 = sp.poly1d(f10p) 
#print(error(f2, x, y)) #179983507.878


inflection = 3.5*7*24 # calculate the inflection point in hours
xa = x[:inflection] # data before the inflection point
ya = y[:inflection]
xb = x[inflection:] # data after
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fax = sp.linspace(0,x[-1], inflection) 
fbx = sp.linspace(inflection, x[-1], 1000) 
fb = sp.poly1d(sp.polyfit(xb, yb, 1)) 
fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
#print("Error inflection=%f" % (fa_error + fb_error)) 
#Error inflection=132950348.197616

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.autoscale(tight=False)
plt.grid()

plt.plot(fx, f1(fx), linewidth=4)
plt.plot(fx, f2(fx), linewidth=4)
plt.plot(fx, f10(fx), linewidth=4) 
plt.plot(fax, fa(fax), linewidth=4) 
plt.plot(fbx, fb(fbx), linewidth=4) 
plt.legend( ["d=%i" % f1.order, "d=%i" % f2.order, "d=%i" % f10.order, 
             "d=%i" % fa.order, "d=%i" % fb.order], loc="upper left") 

plt.show()

