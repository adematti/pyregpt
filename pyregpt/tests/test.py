import os

import numpy as np
from numpy import testing
from scipy import constants,integrate

from pyregpt import PyRegPT, SpectrumLin, SpectrumNoWiggle, Spectrum1Loop, Spectrum2Loop, A1Loop, A2Loop, B1Loop, B2Loop, Bias1Loop

nthreads = 8


def load_pklin():
	return np.loadtxt('ref_pk_lin.dat',unpack=True)


def save_reference_terms_spectrum_2loop():
	pyregpt = Spectrum2Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)][:50:5]
	pyregpt.set_terms(k)
	pyregpt.set_precision(calculation='spectrum_2loop_q',min=-2,max=-1.)
	for a in ['delta','theta']:
		for b in ['delta','theta']:
			pyregpt.run_terms(a,b,nthreads=nthreads)
			np.savetxt('self_terms_spectrum_2loop_{}_{}.dat'.format(a,b),np.concatenate([pyregpt[key][:,None] for key in pyregpt],axis=-1))
	pyregpt.clear()


def load_reference_terms_spectrum_2loop(a='delta',b='theta',self=True):
	if self:
		dtype = [(key,'f8') for key in Spectrum2Loop.FIELDS]
		ref = np.loadtxt('self_terms_spectrum_2loop_{}_{}.dat'.format(a,b),dtype=dtype)
	else:
		dtype = ['k','gamma1a_1loop','gamma1a_2loop','gamma1b_1loop','gamma1b_2loop','pk_gamma2_tree_tree','pk_gamma2_tree_1loop','pk_gamma2_1loop_1loop','pk_gamma3_tree','sigmad2']
		dtype = [(key,'f8') for key in dtype]
		ref = np.loadtxt('ref_terms_spectrum_2loop_{}_{}.dat'.format(a,b),dtype=dtype)[200:280:10] #gamma3 integration is routine has been updated since RegPT
	return ref


def save_reference_spectrum_2loop():
	pyregpt = Spectrum2Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)][:50:5]
	pyregpt.set_terms(k)
	pyregpt.set_precision(calculation='spectrum_2loop_q',min=-2,max=-1.)
	for a in ['delta','theta']:
		for b in ['delta','theta']:
			pyregpt.run_terms(a,b,nthreads=nthreads)
			Dgrowth = np.random.uniform(0.5,2.)*np.ones_like(pyregpt.k)
			pk = pyregpt.pk(Dgrowth)
			pk_lin = pyregpt.pk_lin(Dgrowth)
			np.savetxt('self_spectrum_2loop_{}_{}.dat'.format(a,b),np.asarray([pyregpt.k,pk_lin,pk,Dgrowth]).T)
	pyregpt.clear()


def load_reference_spectrum_2loop(a='delta',b='theta',self=True):
	if self:
		dtype = [(key,'f8') for key in ['k','pk_lin','pk','Dgrowth']]
		ref = np.loadtxt('self_spectrum_2loop_{}_{}.dat'.format(a,b),dtype=dtype)
	else:
		dtype = ['k','pk_nowiggle','pk_lin','pk','error','Dgrowth']
		dtype = [(key,'f8') for key in dtype]
		ref = np.loadtxt('ref_spectrum_2loop_{}_{}.dat'.format(a,b),dtype=dtype)[200:280:10] #gamma3 integration is routine has been updated since RegPT
	return ref


def load_reference_spectrum_1loop(a='delta',b='theta'):
	dtype = ['k','pk_nowiggle','pk_lin','pk','error','Dgrowth']
	dtype = [(key,'f8') for key in dtype]
	ref = np.loadtxt('ref_spectrum_1loop_{}_{}.dat'.format(a,b),dtype=dtype)
	return ref


def save_reference_terms_bias_1loop():
	pyregpt = Bias1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)]
	pyregpt.set_terms(k)
	pyregpt.run_terms(nthreads=nthreads)
	np.savetxt('self_terms_bias_1loop.dat',np.concatenate([pyregpt[key][:,None] for key in pyregpt],axis=-1))
	pyregpt.clear()


def load_reference_terms_bias_1loop():
	dtype = [(key,'f8') for key in Bias1Loop.FIELDS]
	ref = np.loadtxt('self_terms_bias_1loop.dat',dtype=dtype)
	return ref


def save_reference_terms_A_1loop():
	pyregpt = A1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)]
	pyregpt.set_terms(k)
	pyregpt.run_terms(nthreads=nthreads)
	np.savetxt('self_terms_A_1loop.dat',np.concatenate([pyregpt['k'][:,None],pyregpt['pk']],axis=-1))
	pyregpt.clear()


def load_reference_terms_A_1loop():
	dtype = [('k','f8'),('pk',('f8',5))]
	ref = np.loadtxt('self_terms_A_1loop.dat',dtype=dtype)
	return ref


def save_reference_terms_A_2loop():
	pyregpt = A2Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)]
	k = np.asarray([k[5],k[42]])
	pyregpt.set_terms(k)
	pyregpt.set_precision(calculation='gamma1_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='gamma2_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='A_2loop_q',min=-2,max=-1.)
	pyregpt.run_terms(nthreads=nthreads)
	np.savetxt('self_terms_A_2loop.dat',np.concatenate([pyregpt['k'][:,None],pyregpt['pk']],axis=-1))
	pyregpt.clear()


def load_reference_terms_A_2loop(self=True):
	dtype = [('k','f8'),('pk',('f8',5))]
	if self:
		ref = np.loadtxt('self_terms_A_2loop.dat',dtype=dtype)
		return ref
	else:
		ref_I = np.loadtxt('ref_terms_A_2loop_I_sd.dat',dtype=dtype) #the way we calculate sigmad2 is different (RegPT: interpolation, pyRegPT: recalculation)
		ref_II = np.loadtxt('ref_terms_A_2loop_II_sd.dat',dtype=dtype)
		ref_I['pk'] += ref_II['pk']
		return ref_I[200:220:10]


def save_reference_terms_B_1loop():
	pyregpt = B1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	k = pyregpt.spectrum_lin.k[(pyregpt.spectrum_lin.k > 0.1)]
	pyregpt.set_terms(k)
	pyregpt.run_terms(nthreads=nthreads)
	np.savetxt('self_terms_B_1loop.dat',np.concatenate([pyregpt['k'][:,None],pyregpt['pk']],axis=-1))
	pyregpt.clear()


def load_reference_terms_B_1loop():
	dtype = [('k','f8'),('pk',('f8',9))]
	ref = np.loadtxt('self_terms_B_1loop.dat',dtype=dtype)
	return ref


def load_reference_terms_B_2loop():
	dtype = [('k','f8'),('pk',('f8',9))]
	ref = np.loadtxt('ref_terms_B_2loop.dat',dtype=dtype)
	return ref


def test_gauss_legendre():

	pyregpt = PyRegPT()

	start,end,n = 0.1,10.,10
	xref,wref = pyregpt.nodes_weights_gauss_legendre(start,end,n)

	for a,b in [(2.,.0),(0.2,1.)]:
		x,w = pyregpt.nodes_weights_gauss_legendre(a*start+b,a*end+b,10)
		testing.assert_allclose(a*xref+b,x,rtol=1e-7,atol=1e-7)
		testing.assert_allclose(a*wref,w,rtol=1e-7,atol=1e-7)

	x = np.linspace(0.,2*constants.pi,100)
	y = np.sin(x)
	ref = integrate.trapz(y,x=x,axis=0)

	xref,wref = pyregpt.nodes_weights_gauss_legendre(x[0],x[-1],len(x))
	test = np.sum(np.sin(xref)*wref,axis=0)

	testing.assert_allclose(ref,test,rtol=1e-4,atol=1e-4)


def test_find_pk_lin():

	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)

	kout = np.copy(k)
	pk = pyregpt.find_pk_lin(kout,interpol='poly')
	testing.assert_allclose(pklin,pk,rtol=1e-7,atol=1e-7)

	kout = kout/1.2
	kout = kout[kout>k[0]]
	pk = pyregpt.find_pk_lin(kout,interpol='lin')
	ref = np.interp(kout,k,pklin,left=0.,right=0.)
	testing.assert_allclose(ref,pk,rtol=1e-7,atol=1e-7)


def test_interpol_poly():

	pyregpt = PyRegPT()
	tabx = np.linspace(2.,5.,40)
	taby = np.linspace(2.,5.,40)
	for i in range(200):
		for x,y in zip(tabx,taby):
			assert pyregpt.interpol_poly(x,tabx,taby)==y


def test_sigmad2():
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	newk = np.concatenate([k,[k[-1]*2.,k[-1]*3.]],axis=-1)
	sigmad2 = pyregpt.calc_running_sigmad2(newk)
	assert sigmad2[-2] == sigmad2[-1]


def test_terms_1loop(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = Spectrum1Loop()
	pyregpt.set_precision(calculation='spectrum_1loop_q',min=-2,max=-1.)
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_spectrum_2loop(a,b)[20:40]
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	for key in pyregpt:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} {} {} ok'.format(a,b,key))
	pyregpt.clear()


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
	pyregpt.clear()


def test_all_1loop():
	for a in ['delta','theta']:
		for b in ['delta','theta']:
			test_terms_1loop(a,b)
			test_spectrum_1loop(a,b)


def test_terms_2loop(a='delta',b='delta'):

	k,pklin = load_pklin()
	pyregpt = Spectrum2Loop()
	pyregpt.set_precision(calculation='spectrum_2loop_q',min=k[0],max=k[-1])
	pyregpt.set_pk_lin(k,pklin)

	ref = load_reference_terms_spectrum_2loop(a,b)
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	for key in pyregpt:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} {} {} ok'.format(a,b,key))
	pyregpt.set_precision(calculation='all_q',n=700,interpol='poly')
	pyregpt.clear()


def test_spectrum_2loop(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = Spectrum2Loop()
	pyregpt.set_precision(calculation='spectrum_2loop_q',min=-2,max=-1.)
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_spectrum_2loop(a,b)
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	pk_lin = pyregpt.pk_lin(Dgrowth=ref['Dgrowth'][0])
	pk = pyregpt.pk(Dgrowth=ref['Dgrowth'][0])
	testing.assert_allclose(pk_lin,ref['pk_lin'],rtol=1e-5,atol=1e-5)
	testing.assert_allclose(pk,ref['pk'],rtol=1e-5,atol=1e-5)
	pyregpt.clear()


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
	for key in pyregpt:
		if 'gamma1' in key: testing.assert_allclose(pyregpt[key],pyregpt2[key],rtol=1e-5,atol=1e-5)
		else: testing.assert_allclose(pyregpt[key],pyregpt2[key],rtol=1e-8,atol=1e-8)
		print('{} {} {} ok'.format(a,b,key))
	pyregpt.clear()


def test_A_1loop():
	pyregpt = A1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_A_1loop()
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(nthreads=nthreads)
	pyregpt.pk()
	for key in pyregpt:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))
	pyregpt.clear()


def test_A_2loop():
	pyregpt = A2Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_A_2loop()
	pyregpt.set_precision(calculation='gamma1_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='gamma2_1loop_q',min=-2,max=-1.,n=300)
	pyregpt.set_precision(calculation='A_2loop_q',min=-2,max=-1.)
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(nthreads=nthreads)
	pyregpt.pk()
	for key in pyregpt:
		if key in ref.dtype.names:
			#print np.absolute(pyregpt[key]/ref[key]-1).max()
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))
	pyregpt.clear()


def test_B_1loop():
	pyregpt = B1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_B_1loop()
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(nthreads)
	pyregpt.pk()
	for key in pyregpt:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))
	pyregpt.clear()


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
	for key in pyregpt:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))
	pyregpt.clear()


def test_bias_1loop():
	pyregpt = Bias1Loop()
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_bias_1loop()
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(nthreads=nthreads)
	for key in pyregpt:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))
	pyregpt.clear()


def test_verbosity():
	pyregpt = Bias1Loop()
	pyregpt.set_verbosity('quiet')
	print('No output in between <<')
	k,pklin = load_pklin()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_bias_1loop()
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(nthreads=nthreads)
	print('>>')
	for key in pyregpt:
		if key in ref.dtype.names:
			testing.assert_allclose(pyregpt[key],ref[key],rtol=1e-6,atol=1e-7)
			print('{} ok'.format(key))
	pyregpt.clear()


def test_pad(a='delta',b='delta'):
	k,pklin = load_pklin()
	pyregpt = Spectrum2Loop()
	pyregpt.set_pk_lin(k,pklin)
	ref = load_reference_terms_spectrum_2loop(a,b)
	pyregpt.set_terms(ref['k'])
	pyregpt.run_terms(a,b,nthreads=nthreads)
	bak = pyregpt.deepcopy()
	pyregpt.pad_k(k,interpol='poly')
	for key in pyregpt:
		#assert len(pyregpt[key]) == len(k) #false, as pyregpt.k is just extended, not replaced
		mask = (pyregpt['k']>=bak['k'][0]) & (pyregpt['k']<=bak['k'][-1])
		testing.assert_allclose(pyregpt[key][mask],bak[key],rtol=1e-6,atol=1e-7)


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
	from matplotlib import pyplot as plt
	k,pklin = load_pklin()
	pyregpt = PyRegPT()
	pyregpt.set_pk_lin(k,pklin)
	kout = np.logspace(np.log10(k[0])-2,np.log10(k[-1])+1,1000,base=10)
	for interpol in ['lin','poly']:
		plt.loglog(kout,pyregpt.find_pk_lin(kout,interpol=interpol),label=interpol)
	plt.axvline(x=k[0],ymin=0.,ymax=1.)
	plt.axvline(x=k[-1],ymin=0.,ymax=1.)
	plt.legend()
	plt.show()


def plot_regpt():
	kmin,kmax = 1e-3,5e-1
	k,pklin = np.loadtxt('matterpower_wmap5.dat',unpack=True)
	pyregpt = Spectrum2Loop()
	pyregpt.set_pk_lin(k,pklin)
	k,pk = np.loadtxt('pk_RegPT_spectrum.dat',usecols=[0,3],unpack=True)
	kmask = (k>kmin) & (k<kmax)
	k,pk = k[kmask],pk[kmask]
	pyregpt.set_terms(k)
	pyregpt.run_terms('delta','delta',nthreads=nthreads)
	from matplotlib import pyplot as plt
	plt.plot(k,pyregpt.pk()/pk,label='pyregpt/regpt')
	#plt.plot(k,k*pk,label='regpt')
	#plt.plot(pyregpt.k,pyregpt.k*pyregpt.pk(),label='pyregpt')
	plt.xlim(kmin,kmax)
	plt.show()


def plot_spectrum_nowiggle():
	from matplotlib import pyplot as plt
	k,pklin = load_pklin()
	spectrum_nowiggle = SpectrumNoWiggle(k=k)
	cosmo_kwargs = dict(Omega_m=0.31,omega_b=0.022,h=0.676,n_s=0.97)
	spectrum_nowiggle.run_terms(pk=pklin,**cosmo_kwargs)
	plt.loglog(k,pklin,label='$P_{\\rm lin}$')
	plt.loglog(k,spectrum_nowiggle.pk(),label='$P_{\\rm nowiggle}$')
	plt.legend()
	plt.show()


def plot_spectrum_nowiggle_comparison():

	from matplotlib import pyplot as plt
	k,pklin = load_pklin()
	spectrum_nowiggle = SpectrumNoWiggle(k=k)
	cosmo_kwargs = dict(Omega_m=0.31,omega_b=0.022,h=0.676,n_s=0.97)
	spectrum_nowiggle.run_terms(pk=pklin,**cosmo_kwargs)
	tk = spectrum_nowiggle.transfer()

	from nbodykit.cosmology.power.transfers import NoWiggleEisensteinHu
	from nbodykit import cosmology
	cosmo_kwargs = dict(Omega_m=0.31,omega_b=0.022,h=0.676,sigma8=0.8,n_s=0.97,N_ur=2.0328,m_ncdm=[0.06])
	cosmo_kwargs['Omega0_b'] = cosmo_kwargs.pop('omega_b')/cosmo_kwargs['h']**2
	Omega0_m = cosmo_kwargs.pop('Omega_m')
	sigma8 = cosmo_kwargs.pop('sigma8')
	cosmo = cosmology.Cosmology(**cosmo_kwargs).match(Omega0_m=Omega0_m).match(sigma8=sigma8)
	eh = NoWiggleEisensteinHu(cosmo,redshift=0.)(k)

	plt.loglog(k,pklin,label='$P_{\\rm lin}$')
	plt.loglog(k,tk,label='$P_{\\rm nowiggle}$')
	plt.loglog(k,eh,label='nbodykit')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	"""
	save_reference_terms_A_1loop()
	save_reference_terms_A_2loop()
	save_reference_terms_B_1loop()
	save_reference_terms_bias_1loop()
	save_reference_spectrum_2loop()
	save_reference_terms_spectrum_2loop()
	"""
	test_gauss_legendre()
	test_interpol_poly()
	test_find_pk_lin()
	test_sigmad2()
	#test_terms_1loop(a='delta',b='theta')
	test_all_1loop()
	#test_terms_2loop(a='delta',b='theta')
	test_all_2loop()
	test_precision()
	test_A_1loop()
	test_A_2loop()
	test_B_1loop()
	test_B_2loop()
	test_bias_1loop()
	test_verbosity()
	test_pad()
	test_copy()
	plot_pk_lin()
	plot_spectrum_nowiggle()
	#plot_spectrum_nowiggle_comparison()
	#plot_regpt()
