#!/usr/bin/python
# -*- coding: utf-8 -*-
# Filename: scipy_tutorial.py

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
fx_1 = sp.linspace(0,1000,1000)

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
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fax = sp.linspace(0,inflection, 1000) 
fbx = sp.linspace(inflection, 1000, 1000)

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
#print("Error inflection=%f" % (fa_error + fb_error)) 
#Error inflection=132950348.197616

fa2 = sp.poly1d(sp.polyfit(xa, ya, 2))
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))
fa2_error = error(fa2, xa, ya)
fb2_error = error(fb2, xb, yb)
# 第１周到第３.5周，随着指数升高error是降低的　underfit　
# 3.5周到第４周估算 error从指数２升高，error是升高的　
# 得出结论指数２是预测最可能正确的模型

# 预测指数２时候达到100000点击的方程
print(fb2-100000)
#0.07893 x - 84.69 x - 7.563e+04
from scipy.optimize import fsolve
reached_max = fsolve(fb2-100000, 800)/(7*24)
print("100,000 hits/hour expected at week %f" % reached_max[0])
#100,000 hits/hour expected at week 9.837964 

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.autoscale(tight=False)
plt.grid()

# plt.plot(fx, f1(fx_1), linewidth=4)
# plt.plot(fx, f2(fx_1), linewidth=4)
# plt.plot(fx, f10(fx), linewidth=4) 
# plt.plot(fax, fa(fax), linewidth=4) 
# plt.plot(fbx, fb(fbx), linewidth=4)
plt.plot(fax, fa2(fax), linewidth=4) 
plt.plot(fbx, fb2(fbx), linewidth=4) 
plt.legend( ["d=%i" % fa2.order, "d=%i" % fb2.order], loc="upper left") 

plt.show()


