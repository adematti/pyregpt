# coding: utf8

import os
import ctypes
import scipy
import functools
from numpy import ctypeslib
"""
def transfer_nowiggle(k,h=0.676,omega_b=0.022,omega_m=0.21,T_cmb=2.7255,Omega_m=None,Omega_b=None,**kwargs):
	#Fitting formula for no-wiggle P(k) (Eq.[29] of Eisenstein and Hu 1998)
	if Omega_m is not None: omega_m = Omega_m * h**2
	if Omega_b is not None: omega_b = Omega_b * h**2
	frac_baryon  = omega_b / omega_m
	theta_cmb = T_cmb / 2.7
	sound_horizon = 44.5 * h * scipy.log(9.83/omega_m) / scipy.sqrt(1. + 10.*omega_b**0.75)
	alpha_gamma = 1. - 0.328 * scipy.log(431.*omega_m) * frac_baryon + 0.38 * scipy.log(22.3*omega_m) * frac_baryon**2
	k_eq = 0.0746 * omega_m * theta_cmb ** (-2)
	k = k * h
	ks = k * sound_horizon / h
	q = k / (13.41*k_eq)
	gamma_eff = omega_m * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
	q_eff = q * omega_m / gamma_eff
	L0 = scipy.log(2*scipy.e + 1.8 * q_eff)
	C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
	return L0 / (L0 + C0 * q_eff**2)
"""
def transfer_nowiggle(k,h=0.676,omega_b=0.022,omega_m=0.21,T_cmb=2.7255,Omega_m=None,Omega_b=None,**kwargs):
	#Fitting formula for no-wiggle P(k) (Eq.[29] of Eisenstein and Hu 1998)
	if Omega_m is not None: omega_m = Omega_m * h**2
	if Omega_b is not None: omega_b = Omega_b * h**2
	frac_baryon  = omega_b / omega_m
	theta_cmb = T_cmb / 2.7
	sound_horizon = 44.5 * h * scipy.log(9.83/omega_m) / scipy.sqrt(1. + 10.*omega_b**0.75)
	alpha_gamma = 1. - 0.328 * scipy.log(431.*omega_m) * frac_baryon + 0.38 * scipy.log(22.3*omega_m) * frac_baryon**2
	ks = k * sound_horizon
	gamma_eff = omega_m / h * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
	q = k * theta_cmb**2 / gamma_eff
	L0 = scipy.log(2*scipy.e + 1.8 * q)
	C0 = 14.2 + 731.0 / (1 + 62.5 * q)
	return L0 / (L0 + C0 * q**2)

def Pnowiggle(k,pk,n_s=1.,**kwargs):
	pknowiggle = k**n_s*transfer_nowiggle(k,**kwargs)**2
	return pknowiggle*pk[0]/pknowiggle[0]

def scale_factor(npk=1):
	def _decorate(func):
		@functools.wraps(func)				
		def wrapper(self,Dgrowth=1.,*args,**kwargs):
			return Dgrowth**(2*npk)*func(self,*args,**kwargs)
		return wrapper
	return _decorate

def damping_factor(npk=1):
	def _decorate(func):
		@functools.wraps(func)				
		def wrapper(self,Dgrowth=1.,sigmav2=None,*args,**kwargs):
			factor = 0.5 * (self['k']*Dgrowth)**2 * (self['sigma_v2'] if sigmav2 is None else sigmav2)
			return Dgrowth**(2*npk)*scipy.exp(-npk*factor)*(1.+factor)**npk*func(self,*args,**kwargs)
		return wrapper
	return _decorate

class Terms(object):

	SHAPE = {}
	
	def __init__(self,columns={},fields=None):
	
		super(Terms, self).__init__()
		self.columns = {}
		if fields is None:
			self.columns.update(columns)
		else:
			for key in fields:
				self.columns[key] = columns[key]
	
	@property
	def fields(self):
		return self.columns.keys()
		
	def __len__(self):
		return len(self['k'])

	@property
	def size(self):
		return len(self)
		
	def __str__(self):
		return str(self.__dict__)
	
	def __getitem__(self,name):
		if isinstance(name,(str,unicode)):
			if name in self.fields:
				return self.columns[name]
			else:
				raise KeyError('Term {} does not exist.'.format(name))
		else:
			return self.__class__({key:self.columns[key][name] for key in self.fields})	
	
	def __setitem__(self,name,item):
		if isinstance(name,(str,unicode)):
			self.columns[name] = item
		else:
			for key in self.fields:
				tmp = getattr(self,key)
				tmp[name] = item
				setattr(self,key,tmp)
			
	def	__contains__(self,name):
		if name in self.columns: return True
		return False
		
	def as_dict(self,fields=None):
		if fields is None: fields = self.fields
		return {field:self[field] for field in fields}
	
	def copy(self):
		new = self.__class__()
		new.__dict__.update(self.__dict__)
		return new
		
	def deepcopy(self):
		import copy
		new = self.__class__()
		#new.__dict__.update(copy.deepcopy(self.__dict__))
		for key in self.fields: new.columns[key] = self.columns[key].copy()
		return new
	
	def pad_k(self,k,mode='constant',constant_values=0.,**kwargs):
		pad_start = scipy.sum(k<self['k'][0])
		pad_end = scipy.sum(k>self['k'][-1])
		for key in self.FIELDS:
			pad_width = ((0,0))*(self[key].ndim-1) + ((pad_start,pad_end))
			self[key] = scipy.pad(self[key],pad_width=pad_width,mode=mode,constant_values=constant_values)
		self['k'][:pad_start] = k[:pad_start]
		self['k'][-pad_end:] = k[-pad_end:]
		for key in kwargs:
			self[key][:pad_start] = kwargs[key][:pad_start]
			self[key][-pad_end:] = kwargs[key][-pad_end:]
	
	def nan_to_zero(self):
		for key in self.FIELDS: self[key][scipy.isnan(self[key])] = 0.
		
	def rescale(self,scale=1.):
		for key in self.fields: self[key] *= scale**(2*self.SCALE[key])
		
	def zeros(self,k,dtype=ctypes.c_double):
	
		self['k'] = scipy.asarray(k,dtype=dtype).flatten()
		for key in self.FIELDS:
			if key == 'k': continue
			if key in self.SHAPE: self[key] = scipy.zeros((self.size,)+tuple(self.SHAPE[key]),dtype=self.C_TYPE).flatten()
			else: self[key] = scipy.zeros((self.size,),dtype=self.C_TYPE)

		
class TermsPkLin(Terms):

	FIELDS = ['k','pk']
	SCALE = {'k':0,'pk':1}
	
	@scale_factor(1)
	def Plin(self):
		return self['pk']

class PyRegPT(object):

	C_TYPE = ctypes.c_double
	#PATH_CUBA = os.path.join(os.getenv('CUBA'),'libcuba.so')
	PATH_REGPT = os.path.join(os.path.dirname(os.path.realpath(__file__)),'regpt.so')
	
	def __init__(self):

		super(PyRegPT, self).__init__()
		#cuba = ctypes.CDLL(self.PATH_CUBA,mode=ctypes.RTLD_GLOBAL)
		self.regpt = ctypes.CDLL(self.PATH_REGPT,mode=ctypes.RTLD_GLOBAL)
		self.pk_lin = TermsPkLin()

	def nodes_weights_gauss_legendre(self,xmin,xmax,nx):
		
		x = scipy.empty((nx),dtype=self.C_TYPE)
		w = scipy.empty((nx),dtype=self.C_TYPE)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(nx,))
		self.regpt.nodes_weights_gauss_legendre.argtypes = (self.C_TYPE,self.C_TYPE,pointer,pointer,ctypes.c_size_t)
		self.regpt.nodes_weights_gauss_legendre(xmin,xmax,x,w,nx)
		
		return x,w
		
	def interpol_poly(self,x,tabx,taby):
		
		nx = len(tabx)
		assert nx == len(taby)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(nx,))
		self.regpt.interpol_poly.argtypes = (self.C_TYPE,pointer,pointer,ctypes.c_size_t)
		self.regpt.interpol_poly.restype = self.C_TYPE
		return self.regpt.interpol_poly(x,tabx,taby,nx)
	
	def interpol_lin(self,x,tabx,taby):
		
		self.regpt.interpol_lin.argtypes = (self.C_TYPE,)*5
		self.regpt.interpol_lin.restype = self.C_TYPE
		return self.regpt.interpol_lin(x,tabx[0],tabx[-1],taby[0],taby[-1])
	
	def find_pk_lin(self,k,interpol='poly'):
	
		k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		pk = scipy.zeros_like(k)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(len(k),))
		self.regpt.find_pk_lin.argtypes = (pointer,pointer,ctypes.c_size_t,ctypes.c_size_t)
		self.regpt.find_pk_lin(k,pk,len(k),1 if interpol=='poly' else 0)
		
		return pk
	
	def calc_running_sigma_v2(self,k,uvcutoff=0.5):
		
		k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		sigma_v2 = scipy.zeros_like(k)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(len(k),))
		self.regpt.calc_running_sigma_v2.argtypes = (pointer,pointer,ctypes.c_size_t,self.C_TYPE)
		self.regpt.calc_running_sigma_v2(k,sigma_v2,len(k),uvcutoff)
		
		return sigma_v2
	
	def set_pk_lin(self,k,pk):
		
		self.pk_lin['k'] = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		self.pk_lin['pk'] = scipy.asarray(pk,dtype=self.C_TYPE).flatten()

		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.pk_lin.size,))
		self.regpt.set_pk_lin.argtypes = (pointer,pointer,ctypes.c_size_t)
		self.regpt.set_pk_lin(self.pk_lin['k'],self.pk_lin['pk'],self.pk_lin.size)

	def set_precision(self,calculation='allq',n=0,min=1.,max=-1.,interpol='test'):
	
		if calculation == 'all_q':
			for calculation in ['spectrum_2loop_q','bias_q','A_2loop_q','B_q']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
			return
		if calculation == 'all_mu':
			for calculation in ['gamma2_tree_mu','bias_mu','A_2loop_I_mu','B_mu']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
			return
		if calculation == 'spectrum_1loop_q':
			for calculation in ['gamma1_1loop_q','gamma2_tree_q']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
		if calculation == 'spectrum_2loop_q':
			for calculation in ['gamma1_1loop_q','gamma1_2loop_q','gamma2_tree_q','gamma2_1loop_q','gamma3_tree_q']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
			return
		if calculation == 'gamma2_1loop_q':
			for calculation in ['gamma2d_1loop_q','gamma2t_1loop_q']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
			return
		if calculation == 'A_2loop_q':
			for calculation in ['A_2loop_I_q','A_2loop_II_III_q']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
			return
		if calculation == 'all_pklin':
			for calculation in ['spectrum_1loop_pk_lin','spectrum_2loop_pk_lin','bias_pk_lin','bispectrum_1loop_pk_lin','spectrum_A_2loop_pklin','spectrum_B_2loop_pklin']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
			return
		func = getattr(self.regpt,'set_precision_'+calculation)
		if 'pk_lin' in calculation:
			func.argtypes = (ctypes.c_char_p,)
			func(interpol)
		elif calculation in ['gamma3_tree_q','A_2loop_II_III_q']:
			func.argtypes = (self.C_TYPE,self.C_TYPE,ctypes.c_char_p)
			func(min,max,interpol)
		elif 'mu' in calculation:
			func.argtypes = (ctypes.c_size_t,ctypes.c_char_p)
			func(n,interpol)
		else:
			func.argtypes = (ctypes.c_size_t,self.C_TYPE,self.C_TYPE,ctypes.c_char_p)
			func(n,min,max,interpol)
			
	def set_running_uvcutoff(self,calculation='all',uvcutoff=0.5):

		if calculation=='all':
			for calculation in ['spectrum_1loop','spectrum_2loop','bispectrum_1loop','bias']:
				self.set_running_uvcutoff(calculation,uvcutoff=uvcutoff)
			return
		func = getattr(self.regpt,'set_running_uvcutoff_'+calculation)
		func.argtypes = (self.C_TYPE,)
		func(uvcutoff)

class Spectrum1Loop(PyRegPT,Terms):

	FIELDS = ['k','pk_lin','sigma_v2','G1a_1loop','G1b_1loop','pkcorr_G2_tree_tree']
	SCALE = {'k':0,'pk_lin':1,'sigma_v2':1,'G1a_1loop':1,'G1b_1loop':1,'pkcorr_G2_tree_tree':2}
	
	def __init__(self):

		super(Spectrum1Loop,self).__init__()
	
	def set_terms(self,k):
		
		super(Spectrum1Loop,self).zeros(k=k,dtype=self.C_TYPE)
		
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.size,))
		self.regpt.set_terms_spectrum_1loop.argtypes = (ctypes.c_size_t,)+(pointer,)*len(self.FIELDS)
		self.regpt.set_terms_spectrum_1loop(self.size,*[self[key] for key in self.FIELDS])
	
	def run_terms(self,a='delta',b='delta',nthreads=8):
		
		self.regpt.run_terms_spectrum_1loop.argtypes = (ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t)
		self.regpt.run_terms_spectrum_1loop(a,b,nthreads)
	
	@scale_factor(1)
	def Plin(self):
		return self['pk_lin']
	
	def P1loop(self,Dgrowth=1.,sigmav2=None):
		#Dgrowth is Dgrowth or sigma8...
		#Taruya 2012 (arXiv 1208.1191v1) eq 24
		factor = 0.5 * (self['k']*Dgrowth)**2 * (self['sigma_v2'] if sigmav2 is None else sigmav2)
		G1a_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + Dgrowth**2*self['G1a_1loop'])
		G1b_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + Dgrowth**2*self['G1b_1loop'])
		#Taruya 2012 (arXiv 1208.1191v1) first term of eq 23
		pkcorr_G1 = G1a_reg * G1b_reg * self['pk_lin']
		#Taruya 2012 (arXiv 1208.1191v1) second term of eq 23
		pkcorr_G2 = Dgrowth**4 * scipy.exp(-2. * factor) * self['pkcorr_G2_tree_tree']
		return pkcorr_G1 + pkcorr_G2
		
class Spectrum2Loop(Spectrum1Loop):

	FIELDS = ['k','pk_lin','sigma_v2','G1a_1loop','G1a_2loop','G1b_1loop','G1b_2loop','pkcorr_G2_tree_tree','pkcorr_G2_tree_1loop','pkcorr_G2_1loop_1loop','pkcorr_G3_tree']
	SCALE = {'k':0,'pk_lin':1,'sigma_v2':1,'G1a_1loop':1,'G1a_2loop':2,'G1b_1loop':1,'G1b_2loop':2,'pkcorr_G2_tree_tree':2,'pkcorr_G2_tree_1loop':3,'pkcorr_G2_1loop_1loop':4,'pkcorr_G3_tree':3}
	
	def __init__(self):

		super(Spectrum2Loop,self).__init__()
	
	def set_terms(self,k):
		
		super(Spectrum2Loop,self).zeros(k=k,dtype=self.C_TYPE)
		
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.size,))
		self.regpt.set_terms_spectrum_2loop.argtypes = (ctypes.c_size_t,)+(pointer,)*len(self.FIELDS)
		self.regpt.set_terms_spectrum_2loop(self.size,*[self[key] for key in self.FIELDS])
	
	def run_terms(self,a='delta',b='delta',nthreads=8):
		
		self.regpt.run_terms_spectrum_2loop.argtypes = (ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t)
		self.regpt.run_terms_spectrum_2loop(a,b,nthreads)
	
	def P2loop(self,Dgrowth=1.,sigmav2=None):
		#Dgrowth is Dgrowth or sigma8...
		#Taruya 2012 (arXiv 1208.1191v1) eq 24
		factor = 0.5 * (self['k']*Dgrowth)**2 * (self['sigma_v2'] if sigmav2 is None else sigmav2)
		G1a_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + 0.5*factor**2 + Dgrowth**2*self['G1a_1loop']*(1. + factor) + Dgrowth**4*self['G1a_2loop'])
		G1b_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + 0.5*factor**2 + Dgrowth**2*self['G1b_1loop']*(1. + factor) + Dgrowth**4*self['G1b_2loop'])
		#Taruya 2012 (arXiv 1208.1191v1) first term of eq 23
		pkcorr_G1 = G1a_reg * G1b_reg * self['pk_lin']
		#Taruya 2012 (arXiv 1208.1191v1) second term of eq 23
		pkcorr_G2 = Dgrowth**4 * scipy.exp(-2. * factor) * (self['pkcorr_G2_tree_tree'] * (1. + factor)**2 + self['pkcorr_G2_tree_1loop'] * Dgrowth**2 * (1. + factor) + self['pkcorr_G2_1loop_1loop'] * Dgrowth**4)
		#Taruya 2012 (arXiv 1208.1191v1) third term of eq 23
		pkcorr_G3 = Dgrowth**6 * scipy.exp(-2. * factor) * self['pkcorr_G3_tree']
		return pkcorr_G1 + pkcorr_G2 + pkcorr_G3
		
class Bias1Loop(PyRegPT,Terms):

	FIELDS = ['k','pk_lin','sigma_v2','pkbias_b2d','pkbias_bs2d','pkbias_b2t','pkbias_bs2t','pkbias_b22','pkbias_b2s2','pkbias_bs22','sigma3sq']
	SCALE = {'k':0,'pk_lin':1,'sigma_v2':1,'pkbias_b2d':2,'pkbias_bs2d':2,'pkbias_b2t':2,'pkbias_bs2t':2,'pkbias_b22':2,'pkbias_b2s2':2,'pkbias_bs22':2,'sigma3sq':2}
	
	def __init__(self):

		super(Bias1Loop,self).__init__()
	
	@scale_factor(1)
	def Plin(self):
		return self['pk_lin']
		
	@scale_factor(2)
	def Pb2d(self):
		return self['pkbias_b2d']
	
	@scale_factor(2)
	def Pbs2d(self):
		return self['pkbias_bs2d']

	@scale_factor(2)
	def Pb2t(self):
		return self['pkbias_b2t']
	
	@scale_factor(2)
	def Pbs2t(self):
		return self['pkbias_bs2t']
	
	@scale_factor(2)
	def Pb22(self):
		return self['pkbias_b22']

	@scale_factor(2)
	def Pb2s2(self):
		return self['pkbias_b2s2']

	@scale_factor(2)
	def Pbs22(self):
		return self['pkbias_bs22']
	
	@scale_factor(2)
	def Psigma3sq(self):
		return self['pk_lin']*self['sigma3sq']
	
	@damping_factor(2)
	def Pb22damp(self):
		return self['pkbias_b22']

	@damping_factor(2)
	def Pb2s2damp(self):
		return self['pkbias_b2s2']

	@damping_factor(2)
	def Pbs22damp(self):
		return self['pkbias_bs22']
		
	def set_terms(self,k):
		
		super(Bias1Loop,self).zeros(k=k,dtype=self.C_TYPE)
		
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.size,))
		self.regpt.set_terms_bias_1loop.argtypes = (ctypes.c_size_t,)+(pointer,)*len(self.FIELDS)
		self.regpt.set_terms_bias_1loop(self.size,*[self[key] for key in self.FIELDS])
		
	def run_terms(self,nthreads=8):
		
		self.regpt.run_terms_bias_1loop.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_bias_1loop(nthreads)

class A1Loop(PyRegPT,Terms):
	
	FIELDS = ['k','pk_lin','sigma_v2','A']
	SCALE = {'k':0,'pk_lin':1,'sigma_v2':1,'A':2}
	SHAPE = {'A':(5,)}
	
	@scale_factor(1)
	def Plin(self):
		return self['pk_lin']
	
	@scale_factor(2)
	def PA(self):
		return self['A']

	def set_terms(self,k):
		
		super(A1Loop,self).zeros(k=k,dtype=self.C_TYPE)
		
		pointers = [ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self[key].size,)) for key in self.FIELDS]
		self.regpt.set_terms_A_1loop.argtypes = (ctypes.c_size_t,)+tuple(pointers)
		self.regpt.set_terms_A_1loop(self.size,*[self[key] for key in self.FIELDS])
		
		for key in ['A']:
			self[key].shape = (self.size,) + self.SHAPE[key]
			#self[key] = scipy.transpose(self[key],axes=(1,0))
		
	def run_terms(self,nthreads=8):
		
		self.regpt.run_terms_A_1loop.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_A_1loop(nthreads)

class A2Loop(A1Loop):

	def set_terms(self,k):
		
		super(A2Loop,self).zeros(k=k,dtype=self.C_TYPE)
		
		pointers = [ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self[key].size,)) for key in self.FIELDS]
		self.regpt.set_terms_A_2loop.argtypes = (ctypes.c_size_t,)+tuple(pointers)
		self.regpt.set_terms_A_2loop(self.size,*[self[key] for key in self.FIELDS])
		
		for key in ['A']:
			self[key].shape = (self.size,) + self.SHAPE[key]
			#self[key] = scipy.transpose(self[key],axes=(1,0))
		
	def run_terms(self,nthreads=8):
		
		self.regpt.run_terms_A_2loop.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_A_2loop(nthreads)
		

class B2Loop(PyRegPT,Terms):

	FIELDS = ['k','pk_lin','sigma_v2','B']
	SCALE = {'k':0,'pk_lin':1,'sigma_v2':1,'B':2}
	SHAPE = {'B':(9,)}
	
	@scale_factor(1)
	def Plin(self):
		return self['pk_lin']
	
	@scale_factor(2)
	def PB(self):
		return self['B']

	def set_terms(self,k):
		
		super(B2Loop,self).zeros(k=k,dtype=self.C_TYPE)
		
		pointers = [ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self[key].size,)) for key in self.FIELDS]
		self.regpt.set_terms_B.argtypes = (ctypes.c_size_t,)+tuple(pointers)
		self.regpt.set_terms_B(self.size,*[self[key] for key in self.FIELDS])
		
		for key in ['B']:
			self[key].shape = (self.size,) + self.SHAPE[key]
			#self[key] = scipy.transpose(self[key],axes=(1,0))
		
	def run_terms(self,nthreads=8):
		
		self.regpt.run_terms_B.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_B(nthreads)
