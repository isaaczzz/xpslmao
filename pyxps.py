# -*- coding: utf-8 -*-
"""
Isaac Zakaria
20 February 2022

Rev: 22 March 2022
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit.models import PseudoVoigtModel

# TO-DO
# Constrain peak spacings for fits to doublets
# Multiple regions for Shirley background OR integrate Shirley into lmfit
# Add FWHM and area calculations to xpslmao (or add a function to do post-processing after fit results have been stored?)

def xpslmao(path, xlim=False, ylim=False, xticks=False, yticks=False, c1s=False, bg=False, plot=True, plotFits=True, dim=(3.25,3.25), color='k', savefig=False, fit=False, fitBEGuess=False):
    xdim = dim[0]
    ydim = dim[1]
    
    # regex \t+ skips over tabs of length 1 or longer
    data = pd.read_csv(path, delimiter=r"\t+", skiprows = 2, engine="python")
    # sometimes, CSV contains repeats of headers in the middle of data
    # in this case, all entries are imported as strings instead of floats
    # pd.to_numeric with errors="coerce" converts all entries to floats
    # entries that cannot be converted to floats are converted to NaN and dropped
    data = data.apply(lambda x: pd.to_numeric(x, errors="coerce"))
    data = data.dropna()
    
    # 
    # dropCols = [col for col in data.columns if "K.E." in col or "Counts" in col or col == "Background" or col == "Envelope"]
    # dropCols = [col for col in data.columns if col == "Background" or col == ]
    
    # data = data.drop(columns = dropCols)
    
    colors = [[98/255,146/255,190/255],[194/255,104/255,47/255],[133/255,127/255,188/255]]
    
    # cut off K.E. through Envelope (counts) columns
    cutoff = data.columns.get_loc("B.E.")
    data.drop(data.iloc[:, 0:cutoff], axis = 1, inplace = True)
    
    if c1s != False: # shift data according to position of C 1s peak, referenced from Perkin-Elmer Handbook of XPS
        data["B.E."] = data["B.E."] + (285 - c1s)
        fitCols = [col for col in data.columns if "None" in col and col != "None"]
        for col in fitCols: # shift included fits, if applicable
            data[col] = data[col] + (285 - c1s)
    
    fig = 0
    ax = 0
    result = 0
    
    if xlim != False: # trim data so that binding energies fall within the bounds of xlim
        if xlim[1] > xlim[0]: # upper bound must be first
            xlim = [xlim[1], xlim[0]]
        ixmin = data[data["B.E."] < xlim[0]]["B.E."].idxmax()
        ixmax = data[data["B.E."] > xlim[1]]["B.E."].idxmin()
        x = data["B.E."].iloc[ixmin:ixmax]
        y = data["CPS"].iloc[ixmin:ixmax]
    else:
        x = data["B.E."]
        y = data["CPS"]
    
    imax = y.idxmax()
    maxBE = x.loc[imax]
    
    # get binding energies and cps as numpy arrays for shirley background function
    x = x.to_numpy()
    y = y.to_numpy()
    
    if bg == "Included":
        autoBG = data["Background CPS"]
    if bg == "Shirley":
        autoBG = shirley_calculate(x, y)
    
    if fit == False:
        result = 0
    else: # fit an arbirary number of pseudo-Voigt peaks
        model = PseudoVoigtModel(prefix="p1")
        fractionDict = {"p1fraction": 0.3}
        fitBEDict = {}
        heightDict = {}
        fwhmDict = {}
        if fit > 1:
            for k in range(2, fit+1):
                model += PseudoVoigtModel(prefix="p"+str(k))
                fractionDict["p"+str(k)+"fraction"] = 0.3

        # for k in range(1,fit+1):
            # heightDict["p"+str(k)+"amplitude"] = max(y)-min(autoBG)
            # print(max(y))
            # fwhmDict["p"+str(k)+"sigma"] = 0.01
            # if fitBEGuess:
            #     fitBEDict["p"+str(k)+"center"] = fitBEGuess[k-1]
                    
        params = model.make_params(**fractionDict, **fitBEDict, **heightDict, **fwhmDict)
        for k in range(1, fit+1):
            params["p"+str(k)+"fraction"].vary = False
            
            if fitBEGuess:
                params.add(name="p"+str(k)+"center", value=fitBEGuess[k-1], min=xlim[1], max=xlim[0])
            
                params.add(name="p"+str(k)+"amplitude", value=(max(y)-min(autoBG))*1.5, min=0)
                params.add(name="p"+str(k)+"sigma", value=1.5, min=0, max=5)
            
        # print(params)
        init = model.eval(params, x=x)
        result = model.fit(y-autoBG, params, x=x) # 30% Lorentzian
    
    
    if plot:
        
        fig, ax = plt.subplots()
        fig.set_size_inches(xdim,ydim)
        if bg == False:
            autoBG = 0
        elif bg == "Included":
            sns.lineplot(data=data, x="B.E.", y="Background CPS", ax=ax, style=True, dashes=[(5,2)], legend=False, color='tab:gray')
        else:
            ax.plot(x, autoBG, 'k--', color='tab:gray')
        if plotFits:
            k = 0
            fitCols = [col for col in data.columns if "None" in col and col != "None"]
            for col in fitCols:   
                sns.lineplot(data=data, x="B.E.", y=col, ax=ax, style=True, legend=False, color=colors[k])
                k += 1
        sns.lineplot(data=data, x="B.E.", y="CPS", ax=ax, style=True, legend=False, color=color)
        if fit:
            # plt.plot(x, result.best_fit+autoBG, color=colors[0])
            comps = result.eval_components()
            # print(comps)
            for k in range(1,fit+1):
                plt.plot(x, comps["p"+str(k)]+autoBG, color=colors[k-1])
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (CPS)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if xlim == False and ylim == False:
            ax.autoscale(enable = True,axis='both')
            ax.invert_xaxis()
        elif xlim == False:
            ax.autoscale(enable=True,axis='x')
            ax.invert_xaxis()
        elif ylim == False:
            ax.autoscale(enable=True,axis='y')
        if xticks != False:
            plt.xticks(xticks)
        if yticks == False:
            plt.yticks([])
        elif yticks == True:
            plt.yticks()
        else:
            plt.yticks(yticks)
        # ax.set_aspect(1)
        ax.tick_params(direction='in')
        fig.set_size_inches(xdim,ydim)
        plt.tight_layout()
        
        if savefig != False:
            plt.savefig(savefig)
            
        return data, fig, ax, maxBE, result
    else:
        return data, maxBE, result
    
def stackSpectra(pathList, colorList, xlim=False, ylim=False, xlabel=True, ylabel=True, xticks=False, yticks=False, shift=0.0, yshift=0.0, bg=True, plot=True, plotFits=True, dim=(3.25,3.25), color='k', savefig=False, fontsize=12, c1sList=False, normalize=True):
    dataList = list()
    maxBEList = list()
    fig, ax = plt.subplots()
    dy = 0
    for k in range(0,len(pathList)):
        path = pathList[k]
        color = colorList[k]
        data, maxBE, result = xpslmao(path, plot=False)
        dataList.append(data) # save data before applying shifts!
        maxBEList.append(maxBE)
        
        if normalize == True:
            maxCPS = max(data["CPS"])
            data["CPS"] = data["CPS"]/maxCPS
        elif normalize == False:
            data["CPS"] = data["CPS"]
        elif type(normalize) is float:
            data["CPS"] = data["CPS"]/normalize
        elif type(normalize) is list:
            data["CPS"] = data["CPS"]/normalize[k]
            
        if type(yshift) == list or type(yshift) == tuple:
            dy += yshift[k]
            data["CPS"] = data["CPS"] + dy
        else:
            data["CPS"] = data["CPS"] + k*yshift
        if c1sList != False: # shift data according to position of C 1s peak, referenced from Perkin-Elmer Handbook of XPS
            c1s = c1sList[k]
            data["B.E."] = data["B.E."] + (285 - c1s)
        sns.lineplot(data=data, x="B.E.", y="CPS", ax=ax, style=True, legend=False, color=color)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if xlim == False and ylim == False:
            ax.autoscale(enable = True,axis='both')
            ax.invert_xaxis()
        elif xlim == False:
            ax.autoscale(enable=True,axis='x')
            ax.invert_xaxis()
        elif ylim == False:
            ax.autoscale(enable=True,axis='y')
        if xticks != False:
            plt.xticks(xticks)
        if yticks == False:
            plt.yticks([])
        elif yticks == True:
            plt.yticks()
        else:
            plt.yticks(yticks)
        # ax.set_aspect(xdim/ydim)
        plt.minorticks_on()
        if xlabel:
            ax.set_xlabel("Binding Energy (eV)", fontsize=fontsize)
        else:
            ax.set_xlabel("")
        if ylabel:
            ax.set_ylabel("Intensity (CPS)", fontsize=fontsize)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='y', which='minor', bottom=False)
        ax.tick_params(axis='both', labelsize=12)
        ax.tick_params(which='both',direction='in', top=True)
        # ax.tick_params(direction='in', top=True)
        fig.set_size_inches(*dim)
        plt.tight_layout()
        
    if savefig != False:
        plt.savefig(savefig)
    return dataList, fig, ax, maxBEList


def shirley_calculate(x, y, tol=1e-6, maxit=50):
# https://github.com/kaneod/physics/blob/master/python/specs.py

	# Make sure we've been passed arrays and not lists.
	#x = array(x)
	#y = array(y)

	# Sanity check: Do we actually have data to process here?
	#print(any(x), any(y), (any(x) and any(y)))
	if not (any(x) and any(y)):
		print("One of the arrays x or y is empty. Returning zero background.")
		return x * 0

	# Next ensure the energy values are *decreasing* in the array,
	# if not, reverse them.
	if x[0] < x[-1]:
		is_reversed = True
		x = x[::-1]
		y = y[::-1]
	else:
		is_reversed = False

	# Locate the biggest peak.
	maxidx = abs(y - y.max()).argmin()
	
	# It's possible that maxidx will be 0 or -1. If that is the case,
	# we can't use this algorithm, we return a zero background.
	if maxidx == 0 or maxidx >= len(y) - 1:
		print("Boundaries too high for algorithm: returning a zero background.")
		return x * 0
	
	# Locate the minima either side of maxidx.
	lmidx = abs(y[0:maxidx] - y[0:maxidx].min()).argmin()
	rmidx = abs(y[maxidx:] - y[maxidx:].min()).argmin() + maxidx

	xl = x[lmidx]
	yl = y[lmidx]
	xr = x[rmidx]
	yr = y[rmidx]
	
	# Max integration index
	imax = rmidx - 1
	
	# Initial value of the background shape B. The total background S = yr + B,
	# and B is equal to (yl - yr) below lmidx and initially zero above.
	B = y * 0
	B[:lmidx] = yl - yr
	Bnew = B.copy()
	
	it = 0
	while it < maxit:
		# Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
		ksum = 0.0
		for i in range(lmidx, imax):
			ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1] - 2 * yr - B[i] - B[i + 1])
		k = (yl - yr) / ksum
		# Calculate new B
		for i in range(lmidx, rmidx):
			ysum = 0.0
			for j in range(i, imax):
				ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] + y[j + 1] - 2 * yr - B[j] - B[j + 1])
			Bnew[i] = k * ysum
		# If Bnew is close to B, exit.
		#if norm(Bnew - B) < tol:
		B = Bnew - B
		#print(it, (B**2).sum(), tol**2)
		if (B**2).sum() < tol**2:
			B = Bnew.copy()
			break
		else:
			B = Bnew.copy()
		it += 1

	if it >= maxit:
		print("Max iterations exceeded before convergence.")
	if is_reversed:
		#print("Shirley BG: tol (ini = ", tol, ") , iteration (max = ", maxit, "): ", it)
		return (yr + B)[::-1]
	else:
		#print("Shirley BG: tol (ini = ", tol, ") , iteration (max = ", maxit, "): ", it)
		return yr + B