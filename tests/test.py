# coding: utf8

import os
import scipy
from scipy import constants,integrate
from numpy import testing

from pyregpt import *

nthreads = 8

def load_pklin():

	return scipy.loadtxt('ref_pk_lin.dat',unpack=True)

def load_reference_gamma(a='delta',b='theta'):
	dtype = ['k','G1a_1loop','G1a_2loop','G1b_1loop','G1b_2loop','pkcorr_G2_tree_tree','pkcorr_G2_tree_1loop','pkcorr_G2_1loop_1loop','pkcorr_G3_tree','sigma_v2']
	dtype = [(key,'f8') for key in dtype]
	ref = scipy.loadtxt('ref_gamma_{}_{}.dat'.format(a,b),dtype=dtype)
	return ref

def load_reference_pk(a='delta',b='theta'):
	dtype = ['k','pk_nowiggle','pk_lin','pk_2loop','error','Dgrowth']
	dtype = [(key,'f8') for key in dtype]
	ref = scipy.loadtxt('ref_pk_{}_{}.dat'.format(a,b),dtype=dtype)
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
	
def test_interpol_pk_lin():
	
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	
	kout = scipy.copy(k)
	pk = pyregpt.interpol_pk_lin(kout)
	testing.assert_allclose(pklin,pk,rtol=1e-7,atol=1e-7)
	
	kout = kout/1.2
	pk = pyregpt.interpol_pk_lin(kout)
	ref = scipy.interp(kout,k,pklin,left=0.,right=0.)
	testing.assert_allclose(pk,ref,rtol=1e-7,atol=1e-7)

def test_find_pk_lin():
	
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	
	kout = scipy.copy(k)
	pk = pyregpt.find_pk_lin(kout,interpol='poly')
	testing.assert_allclose(pklin,pk,rtol=1e-7,atol=1e-7)
	
	kout = kout/1.2
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

def test_2loop(a='delta',b='delta'):

	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_gamma(a,b)[20:40]
	pyregpt.set_k_2loop(ref['k'])
	pyregpt.run_2loop(a,b,nthreads=nthreads)
	for key in pyregpt.terms_2loop.FIELDS:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt.terms_2loop[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} {} {} ok'.format(a,b,key))

def test_pad(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_gamma(a,b)[20:40]
	pyregpt.set_k_2loop(ref['k'])
	pyregpt.run_2loop(a,b,nthreads=nthreads)
	bak = pyregpt.terms_2loop.deepcopy()
	pyregpt.set_pk_lin(k,pklin)
	pyregpt.terms_2loop.pad_k(k,pk_lin=pyregpt.find_pk_lin(k,interpol='poly'),sigma_v2=pyregpt.calc_running_sigma_v2(k))
	for key in pyregpt.terms_2loop.FIELDS:
		#print key,pyregpt.terms_2loop[key][(pyregpt.terms_2loop.k>=bak.k[0]*0.8) & (pyregpt.terms_2loop.k<=bak.k[-1]*1.1)]
		assert len(pyregpt.terms_2loop[key]) == len(k)
		mask = (pyregpt.terms_2loop.k>=bak.k[0]) & (pyregpt.terms_2loop.k<=bak.k[-1])
		testing.assert_allclose(pyregpt.terms_2loop[key][mask],bak[key],rtol=1e-6,atol=1e-7)
		if key=='k': testing.assert_allclose(pyregpt.terms_2loop[key][~mask],k[~mask],rtol=1e-6,atol=1e-7)
		if key=='pk_lin': testing.assert_allclose(pyregpt.terms_2loop[key][~mask],pklin[~mask],rtol=1e-6,atol=1e-7)
			
"""
def test_2loop_lastk(a='delta',b='delta'):

	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	pyregpt.set_k_2loop(k[-1:])
	pyregpt.run_2loop(a,b,nthreads=nthreads)
	for key in pyregpt.terms_2loop.FIELDS:
		print key,scipy.isnan(pyregpt.terms_2loop[key]).sum(),pyregpt.terms_2loop[key]
"""

def test_precision(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	pyregpt.set_k_2loop(pyregpt.pk_lin.k[10:20])
	pyregpt.run_2loop(a,b,nthreads=nthreads)
	pyregpt2 = PyRegPT()
	pyregpt2.set_pk_lin(k,pklin)
	#pyregpt2.set_precision(calculation='gamma1_1loop',n=700,interpol='poly')
	pyregpt2.set_precision(calculation='allq',n=700,interpol='poly')
	pyregpt2.set_k_2loop(pyregpt.pk_lin.k[10:20])
	pyregpt2.run_2loop(a,b,nthreads=nthreads)
	
	for key in pyregpt.terms_2loop.FIELDS:
		if 'gamma1' in key: testing.assert_allclose(pyregpt.terms_2loop[key],pyregpt2.terms_2loop[key],rtol=1e-5,atol=1e-5)
		else: testing.assert_allclose(pyregpt.terms_2loop[key],pyregpt2.terms_2loop[key],rtol=1e-8,atol=1e-8)
		print('{} {} {} ok'.format(a,b,key))

def test_pk_2loop(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_pk(a,b)[20:30]
	pyregpt.set_k_2loop(ref['k'])
	pyregpt.run_2loop(a,b,nthreads=nthreads)
	pk_lin = pyregpt.terms_2loop.Plin(Dgrowth=ref['Dgrowth'][0])
	pk_2loop = pyregpt.terms_2loop.P2loop(Dgrowth=ref['Dgrowth'][0])
	testing.assert_allclose(pk_lin,ref['pk_lin'],rtol=1e-5,atol=1e-5)
	testing.assert_allclose(pk_2loop,ref['pk_2loop'],rtol=1e-5,atol=1e-5)

def test_all_2loop():
	for a in ['delta','theta']:
		for b in ['delta','theta']:
			test_2loop(a,b)
			test_pk_2loop(a,b)

def test_bias():
	pyregpt = PyRegPT()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	pyregpt.set_k_bias(pyregpt.pk_lin.k)
	pyregpt.run_bias(nthreads=nthreads)
	#for key in pyregpt.KEYS_BIAS: print key,pyregpt.terms_bias[key]

def test_A_B():
	pyregpt = PyRegPT()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	pyregpt.set_k_A_B(pyregpt.pk_lin.k)
	pyregpt.run_A_B(nthreads=nthreads)
	pyregpt.terms_A_B.PA()
	#for key in pyregpt.KEYS_A_B: print key,pyregpt.terms_A_B[key]

#test_gauss_legendre()
#test_interpol_pk_lin()
#test_interpol_poly()
#test_find_pk_lin()
#test_sigma_v2()
#test_2loop()
#test_all_2loop()
test_pad()
#test_precision()
#test_bias()
#test_A_B()
