# coding: utf8

import os
import scipy
from scipy import constants,integrate
from numpy import testing
from pyregpt import *

nthreads = 8

def load_pklin():

	return scipy.loadtxt('ref_pk_lin.dat',unpack=True)
	
def load_reference_terms_2loop(a='delta',b='theta'):
	dtype = ['k','G1a_1loop','G1a_2loop','G1b_1loop','G1b_2loop','pkcorr_G2_tree_tree','pkcorr_G2_tree_1loop','pkcorr_G2_1loop_1loop','pkcorr_G3_tree','sigma_v2']
	dtype = [(key,'f8') for key in dtype]
	ref = scipy.loadtxt('ref_terms_2loop_{}_{}.dat'.format(a,b),dtype=dtype)
	return ref

def load_reference_spectrum_2loop(a='delta',b='theta'):
	dtype = ['k','pk_nowiggle','pk_lin','pk','error','Dgrowth']
	dtype = [(key,'f8') for key in dtype]
	ref = scipy.loadtxt('ref_spectrum_2loop_{}_{}.dat'.format(a,b),dtype=dtype)
	return ref

def load_reference_spectrum_1loop(a='delta',b='theta'):
	dtype = ['k','pk_nowiggle','pk_lin','pk','error','Dgrowth']
	dtype = [(key,'f8') for key in dtype]
	ref = scipy.loadtxt('ref_spectrum_1loop_{}_{}.dat'.format(a,b),dtype=dtype)
	return ref
	
def save_reference_terms_bias_1loop():
	pyregpt = Bias1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)]
	pyregpt.set_terms(k)
	pyregpt.run_terms(nthreads=nthreads)
	scipy.savetxt('self_terms_bias_1loop.dat',scipy.concatenate([pyregpt[key][:,None] for key in pyregpt.FIELDS],axis=-1))

def load_reference_terms_bias_1loop():
	dtype = [(key,'f8') for key in Bias1Loop.FIELDS]
	ref = scipy.loadtxt('self_terms_bias_1loop.dat',dtype=dtype)
	return ref

def save_reference_terms_A_1loop():
	pyregpt = A1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)]
	pyregpt.set_terms(k)
	pyregpt.run_terms(nthreads=nthreads)
	scipy.savetxt('self_terms_A_1loop.dat',scipy.concatenate([pyregpt['k'][:,None],pyregpt['pk']],axis=-1))

def load_reference_terms_A_1loop():
	dtype = [('k','f8'),('pk',('f8',5))]
	ref = scipy.loadtxt('self_terms_A_1loop.dat',dtype=dtype)
	return ref

def save_reference_terms_A_2loop():
	pyregpt = A2Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)]
	k = scipy.asarray([k[5],k[42]])
	pyregpt.set_terms(k)
	pyregpt.set_precision(calculation='gamma1_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='gamma2_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='A_2loop_q',min=-2,max=-1.)
	pyregpt.run_terms(nthreads=nthreads)
	scipy.savetxt('self_terms_A_2loop.dat',scipy.concatenate([pyregpt['k'][:,None],pyregpt['pk']],axis=-1))

def load_reference_terms_A_2loop(self=True):
	dtype = [('k','f8'),('pk',('f8',5))]
	if self:
		ref = scipy.loadtxt('self_terms_A_2loop.dat',dtype=dtype)
		return ref
	else:
		ref_I = scipy.loadtxt('ref_terms_A_2loop_I_sd.dat',dtype=dtype)
		ref_II = scipy.loadtxt('ref_terms_A_2loop_II_sd.dat',dtype=dtype)
		ref_I['pk'] += ref_II['pk']
		return ref_I[200:220:10]

def load_reference_terms_B_2loop():
	dtype = [('k','f8'),('pk',('f8',9))]
	ref = scipy.loadtxt('ref_terms_B_2loop.dat',dtype=dtype)
	return ref

def test_gauss_legendre():

	pyregpt = PyRegPT()
	
	start,end,n = 0.1,10.,10
	xref,wref = pyregpt.nodes_weights_gauss_legendre(start,end,n)
	
	for a,b in [(2.,.0),(0.2,1.)]:
		x,w = pyregpt.nodes_weights_gauss_legendre(a*start+b,a*end+b,10)
		testing.assert_allclose(a*xref+b,x,rtol=1e-7,atol=1e-7)
		testing.assert_allclose(a*wref,w,rtol=1e-7,atol=1e-7)
	
	x = scipy.linspace(0.,2*constants.pi,100)
	y = scipy.sin(x)
	ref = integrate.trapz(y,x=x,axis=0)
	
	xref,wref = pyregpt.nodes_weights_gauss_legendre(x[0],x[-1],len(x))
	test = scipy.sum(scipy.sin(xref)*wref,axis=0)
	
	testing.assert_allclose(ref,test,rtol=1e-4,atol=1e-4)

def test_find_pk_lin():
	
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	
	kout = scipy.copy(k)
	pk = pyregpt.find_pk_lin(kout,interpol='poly')
	testing.assert_allclose(pklin,pk,rtol=1e-7,atol=1e-7)
	
	kout = kout/1.2
	kout = kout[kout>k[0]]
	pk = pyregpt.find_pk_lin(kout,interpol='lin')
	ref = scipy.interp(kout,k,pklin,left=0.,right=0.)
	testing.assert_allclose(ref,pk,rtol=1e-7,atol=1e-7)

def test_interpol_poly():
	
	pyregpt = PyRegPT()
	tabx = scipy.linspace(2.,5.,40)
	taby = scipy.linspace(2.,5.,40)
	for i in range(200):
		for x,y in zip(tabx,taby):
			assert pyregpt.interpol_poly(x,tabx,taby)==y

def test_sigma_v2():
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	newk = scipy.concatenate([k,[k[-1]*2.,k[-1]*3.]],axis=-1)
	sigma_v2 = pyregpt.calc_running_sigma_v2(newk)
	assert sigma_v2[-2] == sigma_v2[-1]

	
def test_terms_1loop(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = Spectrum1Loop()
	pyregpt.set_precision(calculation='spectrum_1loop_q',min=-2,max=-1.)
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_2loop(a,b)[20:40]
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	for key in pyregpt.FIELDS:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} {} {} ok'.format(a,b,key))

def test_spectrum_1loop(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = Spectrum1Loop()
	pyregpt.set_precision(calculation='spectrum_1loop_q',min=-2,max=-1.)
	#pyregpt.set_precision(calculation='gamma2_tree_mu',n=15)
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_spectrum_1loop(a,b)[10:200]
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	pk_lin = pyregpt.pk_lin(Dgrowth=ref['Dgrowth'][0])
	pk = pyregpt.pk(Dgrowth=ref['Dgrowth'][0])
	testing.assert_allclose(pk_lin,ref['pk_lin'],rtol=1e-5,atol=1e-4)
	testing.assert_allclose(pk,ref['pk'],rtol=1e-5,atol=1e-5)
	

def test_all_1loop():
	for a in ['delta','theta']:
		for b in ['delta','theta']:
			test_terms_1loop(a,b)
			test_spectrum_1loop(a,b)

def test_terms_2loop(a='delta',b='delta'):

	k,pklin = load_pklin()
	pyregpt = Spectrum2Loop()
	pyregpt.set_precision(calculation='spectrum_2loop_q',min=-2,max=-1.)
	pyregpt.set_pk_lin(k,pklin)
	
	ref = load_reference_terms_2loop(a,b)[200:240]
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	for key in pyregpt.FIELDS:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} {} {} ok'.format(a,b,key))
	pyregpt.set_precision(calculation='all_q',n=700,interpol='poly')


def test_spectrum_2loop(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = Spectrum2Loop()
	#pyregpt.set_precision(calculation='spectrum_2loop_q',min=-2,max=-1.)
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_spectrum_2loop(a,b)[20:30]
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	pk_lin = pyregpt.pk_lin(Dgrowth=ref['Dgrowth'][0])
	pk = pyregpt.pk(Dgrowth=ref['Dgrowth'][0])
	testing.assert_allclose(pk_lin,ref['pk_lin'],rtol=1e-5,atol=1e-5)
	testing.assert_allclose(pk,ref['pk'],rtol=1e-5,atol=1e-5)

def test_all_2loop():
	for a in ['delta','theta']:
		for b in ['delta','theta']:
			test_terms_2loop(a,b)
			test_spectrum_2loop(a,b)

def test_precision(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = Spectrum2Loop()
	pyregpt.set_pk_lin(k,pklin)
	pyregpt.set_terms(pyregpt.spectrum_lin.k[10:20])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	pyregpt2 = Spectrum2Loop()
	pyregpt2.set_pk_lin(k,pklin)
	#pyregpt2.set_precision(calculation='gamma1_1loop',n=700,interpol='poly')
	pyregpt2.set_precision(calculation='all_q',n=700,interpol='poly')
	pyregpt2.set_terms(pyregpt.spectrum_lin.k[10:20])
	pyregpt2.run_terms(a,b,nthreads=nthreads)
	
	for key in pyregpt.FIELDS:
		if 'gamma1' in key: testing.assert_allclose(pyregpt[key],pyregpt2[key],rtol=1e-5,atol=1e-5)
		else: testing.assert_allclose(pyregpt[key],pyregpt2[key],rtol=1e-8,atol=1e-8)
		print('{} {} {} ok'.format(a,b,key))

def test_A_1loop():
	pyregpt = A1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_A_1loop()
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(nthreads=nthreads)
	pyregpt.pk()
	for key in pyregpt.FIELDS:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))

def test_A_2loop():
	pyregpt = A2Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_A_2loop()
	pyregpt.set_terms(ref['k'])
	pyregpt.set_precision(calculation='gamma1_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='gamma2_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='A_2loop_q',min=-2,max=-1.)
	pyregpt.run_terms(nthreads=nthreads)
	pyregpt.pk()
	for key in pyregpt.FIELDS:
		if key in ref.dtype.names:
			#print scipy.absolute(pyregpt[key]/ref[key]-1).max()
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))
	
def test_B_2loop():
	pyregpt = B2Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_B_2loop()
	pyregpt.set_terms(ref['k'])
	pyregpt.set_precision(calculation='gamma1_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='gamma2_tree_q',min=-2,max=-1.,n=400)
	pyregpt.set_precision(calculation='gamma2_tree_mu',min=-2,max=-1.,n=15)
	pyregpt.set_precision(calculation='B_q',min=-2,max=-1.)
	pyregpt.run_terms(nthreads)
	pyregpt.pk()
	for key in pyregpt.FIELDS:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))

def test_bias_1loop():
	pyregpt = Bias1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_bias_1loop()
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(nthreads=nthreads)
	for key in pyregpt.FIELDS:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))

def test_pad(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = Spectrum2Loop()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_2loop(a,b)[20:40]
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	bak = pyregpt.deepcopy()
	pyregpt.set_pk_lin(k,pklin)
	pyregpt.pad_k(k,pk_lin=pyregpt.find_pk_lin(k,interpol='poly'),sigma_v2=pyregpt.calc_running_sigma_v2(k))
	for key in pyregpt.FIELDS:
		#print key,pyregpt.terms_2loop[key][(pyregpt.terms_2loop.k>=bak.k[0]*0.8) & (pyregpt.terms_2loop.k<=bak.k[-1]*1.1)]
		assert len(pyregpt[key]) == len(k)
		mask = (pyregpt['k']>=bak['k'][0]) & (pyregpt['k']<=bak['k'][-1])
		testing.assert_allclose(pyregpt[key][mask],bak[key],rtol=1e-6,atol=1e-7)
		if key=='k': testing.assert_allclose(pyregpt[key][~mask],k[~mask],rtol=1e-6,atol=1e-7)
		if key=='pk_lin': testing.assert_allclose(pyregpt[key][~mask],pklin[~mask],rtol=1e-6,atol=1e-7)

def test_copy():
	pyregpt = PyRegPT()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	cp = pyregpt.copy()
	dcp = pyregpt.deepcopy()
	pyregpt.spectrum_lin['pk'][0] *= 2.
	assert cp.spectrum_lin['pk'][0] == pyregpt.spectrum_lin['pk'][0]
	assert dcp.spectrum_lin['pk'][0] == pyregpt.spectrum_lin['pk'][0]/2.

def plot_pk_lin():
	from matplotlib import pyplot
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	kout = scipy.logspace(scipy.log10(k[0])-2,scipy.log10(k[-1])+1,1000,base=10)
	for interpol in ['lin','poly']:
		pyplot.loglog(kout,pyregpt.find_pk_lin(kout,interpol=interpol),label=interpol)
	pyplot.legend()
	pyplot.axvline(x=k[0],ymin=0.,ymax=1.)
	pyplot.axvline(x=k[-1],ymin=0.,ymax=1.)
	pyplot.show()

def plot_spectrum_nowiggle():
	from matplotlib import pyplot
	k,pklin = load_pklin()
	spectrum_nowiggle = SpectrumNoWiggle(k=k)
	spectrum_nowiggle.run_terms(pk=pklin,**{'Omega_m':0.31,'omega_b':0.022,'h':0.676,'sigma8':0.8,'n_s':0.96,'N_ur':2.0328,'N_ncdm':1,'m_ncdm':0.06,'z_pk':0.86,'input_verbose':1})
	pyplot.loglog(k,pklin,label='$P_{\\rm lin}$')
	pyplot.loglog(k,spectrum_nowiggle.pk(),label='$P_{\\rm nowiggle}$')
	pyplot.legend()
	pyplot.show()

"""
save_reference_terms_A_1loop()
save_reference_terms_A_2loop()
save_reference_terms_bias_1loop()
"""
"""
test_gauss_legendre()
test_interpol_poly()
test_find_pk_lin()
test_sigma_v2()
#test_terms_1loop(a='delta',b='theta')
test_all_1loop()
#test_terms_2loop(a='delta',b='theta')
test_all_2loop()
test_precision()
test_A_1loop()
test_A_2loop()
test_B_2loop()
test_bias_1loop()
test_pad()
test_copy()
plot_pk_lin()
plot_spectrum_nowiggle()
"""
test_A_2loop()


"""
def debug():
	k,_,_,pk_ref_dt,pk_ref_tt = scipy.loadtxt('ref_spectra_B.dat',unpack=True)
	k,pk_debug_dt,pk_debug_tt = scipy.loadtxt('debug_spectra_B.dat',unpack=True)
	testing.assert_allclose(pk_debug_dt,pk_ref_dt,rtol=1e-6,atol=1e-7)
	testing.assert_allclose(pk_debug_tt,pk_ref_tt,rtol=1e-6,atol=1e-7)

debug()
"""
