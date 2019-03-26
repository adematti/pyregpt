import os
import ctypes
import scipy
from scipy import constants,integrate
import functools
import copy
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
		def wrapper(self,Dgrowth=1.,sigmad2=None,*args,**kwargs):
			factor = 0.5 * (self['k']*Dgrowth)**2 * (self['sigmad2'] if sigmad2 is None else sigmad2)
			regfactor = scipy.exp(-npk*factor)*(1.+factor)**npk
			return Dgrowth**(2*npk)*regfactor*func(self,*args,**kwargs)
		return wrapper
	return _decorate

class Terms(object):

	FIELDS = []
	SHAPE = {}

	def __init__(self,**kwargs):
		super(Terms, self).__init__()
		self.columns = kwargs

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
			return self.__class__(**{key:self.columns[key][name] for key in self.fields})	
	
	def __setitem__(self,name,item):
		if isinstance(name,(str,unicode)):
			self.columns[name] = item
		else:
			for key in self.fields:
				self.columns[key][name] = item
	
	def	__contains__(self,name):
		return name in self.columns
	
	def __iter__(self):
		for field in self.FIELDS:
			if field in self.fields: yield field
	
	def tolist(self):
		return [self[key] for key in self]
	
	def pop(self,key):
		return self.columns.pop(key)

	def getstate(self,fields=None):
		if fields is None: fields = self.fields
		return {field:self[field] for field in fields}

	def copy(self):
		new = self.__class__()
		new.__dict__.update(self.__dict__)
		return new

	def deepcopy(self):
		new = self.__class__()
		#new.__dict__.update(copy.deepcopy(self.__dict__))
		for key in self.fields: new.columns[key] = self.columns[key].copy()
		return new

	def pad_k(self,k,mode='constant',constant_values=0.,**kwargs):
		pad_start = scipy.sum(k<self.k[0])
		pad_end = scipy.sum(k>self.k[-1])
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
			
	def reshape(self):
		for key in self.FIELDS:
			if key in self.SHAPE: self[key].shape = (self.size,) + self.SHAPE[key]
	
	@property
	def k(self):
		return self['k']
	
	def pk_interp(self,k,left=0.,right=0.,**kwargs):
		return scipy.interp(k,self.k,self.pk(**kwargs),left=left,right=right)
		
	def sigmav(self,**kwargs):
		return scipy.sqrt(1./6./constants.pi**2*integrate.trapz(self.pk(**kwargs),x=self.k,axis=-1))
	
	def sigmar(self,r,**kwargs):
		x = self.k*r
		w = 3.*(scipy.sin(x)-x*scipy.cos(x))/x**3
		sigmar2 = 1./2./constants.pi**2*integrate.trapz(self.pk(**kwargs)*(w*self.k)**2,x=self.k,axis=-1)
		return scipy.sqrt(sigmar2)

	def sigma8(self,**kwargs):
		return self.sigmar(8.,**kwargs)

class SpectrumLin(Terms):

	FIELDS = ['k','pk']
	SCALE = {'k':0,'pk':1}

	@scale_factor(1)
	def pk(self):
		return self['pk']

class SpectrumNoWiggle(SpectrumLin):

	FIELDS = ['k','pk']
	SCALE = {'k':0,'pk':1}
	
	def transfer(self,h=0.676,omega_b=0.022,omega_m=0.21,T_cmb=2.7255,Omega_m=None,Omega_b=None,**kwargs):
		# Fitting formula for no-wiggle P(k) (Eq.[29] of Eisenstein and Hu 1998)
		if Omega_m is not None: omega_m = Omega_m * h**2
		if Omega_b is not None: omega_b = Omega_b * h**2
		frac_baryon  = omega_b / omega_m
		theta_cmb = T_cmb / 2.7
		sound_horizon = 44.5 * h * scipy.log(9.83/omega_m) / scipy.sqrt(1. + 10.*omega_b**0.75)
		alpha_gamma = 1. - 0.328 * scipy.log(431.*omega_m) * frac_baryon + 0.38 * scipy.log(22.3*omega_m) * frac_baryon**2
		ks = self.k * sound_horizon
		gamma_eff = omega_m / h * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
		q = self.k * theta_cmb**2 / gamma_eff
		L0 = scipy.log(2*scipy.e + 1.8 * q)
		C0 = 14.2 + 731.0 / (1 + 62.5 * q)
		return L0 / (L0 + C0 * q**2)
	
	def set_terms(self,k):
		self.zeros(k=k)
	
	def run_terms(self,pk,n_s=1.,**kwargs):
		pknowiggle = self.k**n_s*self.transfer(**kwargs)**2
		self['pk'] = pknowiggle*pk[0]/pknowiggle[0]
	
	@scale_factor(1)
	def pk(self):
		return self['pk']
	

class PyRegPT(Terms):

	C_TYPE = ctypes.c_double
	#PATH_CUBA = os.path.join(os.getenv('CUBA'),'libcuba.so')
	PATH_REGPT = os.path.join(os.path.dirname(os.path.realpath(__file__)),'regpt.so')

	def __init__(self,*args,**kwargs):

		super(PyRegPT, self).__init__(*args,**kwargs)
		#cuba = ctypes.CDLL(self.pkTH_CUBA,mode=ctypes.RTLD_GLOBAL)
		self.regpt = ctypes.CDLL(self.PATH_REGPT,mode=ctypes.RTLD_GLOBAL)
		self.set_verbosity(self._verbose if hasattr(self,'_verbose') else 'info')

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

	def calc_running_sigmad2(self,k,uvcutoff=0.5):

		k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		sigmad2 = scipy.zeros_like(k)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(len(k),))
		self.regpt.calc_running_sigmad2.argtypes = (pointer,pointer,ctypes.c_size_t,self.C_TYPE)
		self.regpt.calc_running_sigmad2(k,sigmad2,len(k),uvcutoff)

		return sigmad2

	def set_verbosity(self,mode='info'):
		self.regpt.set_verbosity.argtypes = (ctypes.c_char_p,)
		self.regpt.set_verbosity(mode)
		self._verbose = mode
		
	def set_spectrum_lin(self,spectrum_lin):
	
		self.spectrum_lin = spectrum_lin
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.spectrum_lin.size,))
		self.regpt.set_pk_lin.argtypes = (pointer,pointer,ctypes.c_size_t)
		self.regpt.set_pk_lin(self.spectrum_lin.k,self.spectrum_lin['pk'],self.spectrum_lin.size)

	def set_pk_lin(self,k,pk):

		self.spectrum_lin = SpectrumLin()
		self.spectrum_lin['k'] = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		self.spectrum_lin['pk'] = scipy.asarray(pk,dtype=self.C_TYPE).flatten()

		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.spectrum_lin.size,))
		self.regpt.set_pk_lin.argtypes = (pointer,pointer,ctypes.c_size_t)
		self.regpt.set_pk_lin(self.spectrum_lin['k'],self.spectrum_lin['pk'],self.spectrum_lin.size)

	def set_precision(self,calculation='all_q',n=0,min=1.,max=-1.,interpol='test'):
	
		if calculation == 'all':
			for calculation in ['all_q','all_mu']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
			return
		if calculation == 'all_q':
			for calculation in ['spectrum_2loop_q','bias_1loop_q','A_2loop_q','B_q']:
				self.set_precision(calculation,n=n,min=min,max=max,interpol=interpol)
			return
		if calculation == 'all_mu':
			for calculation in ['gamma2_tree_mu','bias_1loop_mu','A_2loop_I_mu','B_mu']:
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
			for calculation in ['spectrum_1loop','spectrum_2loop','bispectrum_1loop','bias_1loop']:
				self.set_running_uvcutoff(calculation,uvcutoff=uvcutoff)
			return
		func = getattr(self.regpt,'set_running_uvcutoff_'+calculation)
		func.argtypes = (self.C_TYPE,)
		func(uvcutoff)

	def deepcopy(self):
		new = super(PyRegPT,self).deepcopy()
		new.spectrum_lin = self.spectrum_lin.deepcopy()
		return new
			
	def pad_k(self,k,mode='constant',constant_values=0.,interpol='poly'):
		super(PyRegPT,self).pad_k(k,mode=mode,constant_values=constant_values)
		if 'pk_lin' in self.FIELDS: self['pk_lin'] = self.find_pk_lin(self.k,interpol=interpol)
		if 'sigmad2' in self.FIELDS: self['sigmad2'] = self.calc_running_sigmad2(self.k)

	def clear(self):
		#To reset all precision settings
		self.set_precision(calculation='all',n=0,min=1.,max=-1.,interpol='test')
		self.set_running_uvcutoff(calculation='all')
		self.set_verbosity()

class Spectrum1Loop(PyRegPT):

	FIELDS = ['k','pk_lin','sigmad2','gamma1a_1loop','gamma1b_1loop','pk_gamma2_tree_tree']
	SCALE = {'k':0,'pk_lin':1,'sigmad2':1,'gamma1a_1loop':1,'gamma1b_1loop':1,'pk_gamma2_tree_tree':2}

	def set_terms(self,k):
		
		self.zeros(k=k,dtype=self.C_TYPE)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.size,))
		self.regpt.set_terms_spectrum_1loop.argtypes = (ctypes.c_size_t,)+(pointer,)*len(self.FIELDS)
		self.regpt.set_terms_spectrum_1loop(self.size,*self.tolist())

	def run_terms(self,a='delta',b='delta',nthreads=8):
		
		self.regpt.run_terms_spectrum_1loop.argtypes = (ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t)
		self.regpt.run_terms_spectrum_1loop(a,b,nthreads)

	@scale_factor(1)
	def pk_lin(self):
		return self['pk_lin']

	def pk(self,Dgrowth=1.,sigmad2=None):
		#Dgrowth is Dgrowth or sigma8...
		#Taruya 2012 (arXiv 1208.1191v1) eq 24
		factor = 0.5 * (self['k']*Dgrowth)**2 * (self['sigmad2'] if sigmad2 is None else sigmad2)
		gamma1a_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + Dgrowth**2*self['gamma1a_1loop'])
		gamma1b_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + Dgrowth**2*self['gamma1b_1loop'])
		#Taruya 2012 (arXiv 1208.1191v1) first term of eq 23
		pk_gamma1 = gamma1a_reg * gamma1b_reg * self['pk_lin']
		#Taruya 2012 (arXiv 1208.1191v1) second term of eq 23
		pk_gamma2 = Dgrowth**4 * scipy.exp(-2. * factor) * self['pk_gamma2_tree_tree']
		return pk_gamma1 + pk_gamma2
		
class Spectrum2Loop(Spectrum1Loop):

	FIELDS = ['k','pk_lin','sigmad2','gamma1a_1loop','gamma1a_2loop','gamma1b_1loop','gamma1b_2loop','pk_gamma2_tree_tree','pk_gamma2_tree_1loop','pk_gamma2_1loop_1loop','pk_gamma3_tree']
	SCALE = {'k':0,'pk_lin':1,'sigmad2':1,'gamma1a_1loop':1,'gamma1a_2loop':2,'gamma1b_1loop':1,'gamma1b_2loop':2,'pk_gamma2_tree_tree':2,'pk_gamma2_tree_1loop':3,'pk_gamma2_1loop_1loop':4,'pk_gamma3_tree':3}
	
	def set_terms(self,k):
		
		self.zeros(k=k,dtype=self.C_TYPE)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.size,))
		self.regpt.set_terms_spectrum_2loop.argtypes = (ctypes.c_size_t,)+(pointer,)*len(self.FIELDS)
		self.regpt.set_terms_spectrum_2loop(self.size,*self.tolist())
	
	def run_terms(self,a='delta',b='delta',nthreads=8):
		
		self.regpt.run_terms_spectrum_2loop.argtypes = (ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t)
		self.regpt.run_terms_spectrum_2loop(a,b,nthreads)
	
	def pk(self,Dgrowth=1.,sigmad2=None):
		#Dgrowth is Dgrowth or sigma8...
		#Taruya 2012 (arXiv 1208.1191v1) eq 24
		factor = 0.5 * (self['k']*Dgrowth)**2 * (self['sigmad2'] if sigmad2 is None else sigmad2)
		gamma1a_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + 0.5*factor**2 + Dgrowth**2*self['gamma1a_1loop']*(1. + factor) + Dgrowth**4*self['gamma1a_2loop'])
		gamma1b_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + 0.5*factor**2 + Dgrowth**2*self['gamma1b_1loop']*(1. + factor) + Dgrowth**4*self['gamma1b_2loop'])
		#Taruya 2012 (arXiv 1208.1191v1) first term of eq 23
		pk_gamma1 = gamma1a_reg * gamma1b_reg * self['pk_lin']
		#Taruya 2012 (arXiv 1208.1191v1) second term of eq 23
		pk_gamma2 = Dgrowth**4 * scipy.exp(-2. * factor) * (self['pk_gamma2_tree_tree'] * (1. + factor)**2 + self['pk_gamma2_tree_1loop'] * Dgrowth**2 * (1. + factor) + self['pk_gamma2_1loop_1loop'] * Dgrowth**4)
		#Taruya 2012 (arXiv 1208.1191v1) third term of eq 23
		pk_gamma3 = Dgrowth**6 * scipy.exp(-2. * factor) * self['pk_gamma3_tree']
		return pk_gamma1 + pk_gamma2 + pk_gamma3
		#return pk_gamma1 + pk_gamma2 + pk_gamma3
		
class Bias1Loop(PyRegPT):

	FIELDS = ['k','pk_lin','sigmad2','pk_b2d','pk_bs2d','pk_b2t','pk_bs2t','pk_b22','pk_b2s2','pk_bs22','sigma3sq']
	SCALE = {'k':0,'pk_lin':1,'sigmad2':1,'pk_b2d':2,'pk_bs2d':2,'pk_b2t':2,'pk_bs2t':2,'pk_b22':2,'pk_b2s2':2,'pk_bs22':2,'sigma3sq':2}
	
	@scale_factor(1)
	def pk_lin(self):
		return self['pk_lin']
		
	@scale_factor(2)
	def pk_b2d(self):
		return self['pk_b2d']
	
	@scale_factor(2)
	def pk_bs2d(self):
		return self['pk_bs2d']

	@scale_factor(2)
	def pk_b2t(self):
		return self['pk_b2t']
	
	@scale_factor(2)
	def pk_bs2t(self):
		return self['pk_bs2t']
	
	@scale_factor(2)
	def pk_b22(self):
		return self['pk_b22']

	@scale_factor(2)
	def pk_b2s2(self):
		return self['pk_b2s2']

	@scale_factor(2)
	def pk_bs22(self):
		return self['pk_bs22']
	
	@scale_factor(2)
	def pk_sigma3sq(self):
		return self['pk_lin']*self['sigma3sq']
	
	@damping_factor(2)
	def pk_b22_damp(self):
		return self['pk_b22']

	@damping_factor(2)
	def pk_b2s2_damp(self):
		return self['pk_b2s2']

	@damping_factor(2)
	def pk_bs22_damp(self):
		return self['pk_bs22']
		
	def set_terms(self,k):
		
		self.zeros(k=k,dtype=self.C_TYPE)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.size,))
		self.regpt.set_terms_bias_1loop.argtypes = (ctypes.c_size_t,)+(pointer,)*len(self.FIELDS)
		self.regpt.set_terms_bias_1loop(self.size,*self.tolist())
		
	def run_terms(self,nthreads=8):
		
		self.regpt.run_terms_bias_1loop.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_bias_1loop(nthreads)

	def shotnoise(self,kmin=5e-4,kmax=10.):
		ones = scipy.ones_like(self.k)
		mask = self.k>kmax
		ones[mask] *= scipy.exp(-(self.k[mask]/kmax-1)**2)
		mask = self.k<kmin
		ones[mask] *= scipy.exp(-(kmin/self.k[mask]-1)**2)
		return ones


class A1Loop(PyRegPT):
	
	FIELDS = ['k','pk_lin','sigmad2','pk']
	SCALE = {'k':0,'pk_lin':1,'sigmad2':1,'pk':2}
	SHAPE = {'pk':(5,)}
	
	@scale_factor(1)
	def pk_lin(self):
		return self['pk_lin']
	
	@scale_factor(2)
	def pk(self):
		return self['pk']

	def set_terms(self,k):
		
		self.zeros(k=k,dtype=self.C_TYPE)
		pointers = [ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self[key].size,)) for key in self.FIELDS]
		self.regpt.set_terms_A_1loop.argtypes = (ctypes.c_size_t,)+tuple(pointers)
		self.regpt.set_terms_A_1loop(self.size,*self.tolist())
		
	def run_terms(self,nthreads=8):
		
		self.regpt.run_terms_A_1loop.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_A_1loop(nthreads)
		self.reshape()
	
	def pk_interp(self,k,mu2,beta=1.,Dgrowth=1.,left=0.,right=0.):
		pk = self.pk(Dgrowth).T
		mu4 = mu2*mu2
		mu6 = mu2*mu4
		beta2 = beta**2
		beta3 = beta2*beta
		beta4 = beta2*beta2
		return mu2*beta*scipy.interp(k,self.k,pk[0],left=left,right=right) \
			+mu2*beta2*scipy.interp(k,self.k,pk[1],left=left,right=right) \
			+mu4*beta2*scipy.interp(k,self.k,pk[2],left=left,right=right) \
			+mu4*beta3*scipy.interp(k,self.k,pk[3],left=left,right=right) \
			+mu6*beta3*scipy.interp(k,self.k,pk[4],left=left,right=right)

class A2Loop(A1Loop):

	def set_terms(self,k):
		
		self.zeros(k=k,dtype=self.C_TYPE)
		pointers = [ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self[key].size,)) for key in self.FIELDS]
		self.regpt.set_terms_A_2loop.argtypes = (ctypes.c_size_t,)+tuple(pointers)
		self.regpt.set_terms_A_2loop(self.size,*self.tolist())
		self.reshape()
		
	def run_terms(self,nthreads=8):
		
		self.regpt.run_terms_A_2loop.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_A_2loop(nthreads)

class B1Loop(PyRegPT):

	FIELDS = ['k','pk_lin','sigmad2','pk']
	SCALE = {'k':0,'pk_lin':1,'sigmad2':1,'pk':2}
	SHAPE = {'pk':(9,)}
	
	@scale_factor(1)
	def pk_lin(self):
		return self['pk_lin']
	
	@scale_factor(2)
	def pk(self):
		return self['pk']

	def set_terms(self,k):
		
		self.zeros(k=k,dtype=self.C_TYPE)
		pointers = [ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self[key].size,)) for key in self.FIELDS]
		self.regpt.set_terms_B.argtypes = (ctypes.c_size_t,)+tuple(pointers)
		self.regpt.set_terms_B(self.size,*self.tolist())
		self.reshape()
		
	def run_terms(self,nthreads=8):
		
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.spectrum_lin.size,))
		self.regpt.set_spectra_B.argtypes = (ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t,pointer,pointer)
		self.regpt.set_spectra_B('delta','theta',self.spectrum_lin.size,self.spectrum_lin.k,self.spectrum_lin['pk'])
		self.regpt.set_spectra_B('theta','theta',self.spectrum_lin.size,self.spectrum_lin.k,self.spectrum_lin['pk'])
		self.regpt.run_terms_B.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_B(nthreads)
	
	def pk_interp(self,k,mu2,beta=1.,Dgrowth=1.,left=0.,right=0.):
		pk = self.pk(Dgrowth).T
		mu4 = mu2*mu2
		mu6 = mu2*mu4
		mu8 = mu2*mu6
		beta2 = beta**2
		beta3 = beta2*beta
		beta4 = beta2*beta2
		return mu2*beta2*scipy.interp(k,self['k'],pk[0],left=0.,right=0.) \
			+mu2*beta3*scipy.interp(k,self['k'],pk[1],left=0.,right=0.) \
			+mu2*beta4*scipy.interp(k,self['k'],pk[2],left=0.,right=0.) \
			+mu4*beta2*scipy.interp(k,self['k'],pk[3],left=0.,right=0.) \
			+mu4*beta3*scipy.interp(k,self['k'],pk[4],left=0.,right=0.) \
			+mu4*beta4*scipy.interp(k,self['k'],pk[5],left=0.,right=0.) \
			+mu6*beta3*scipy.interp(k,self['k'],pk[6],left=0.,right=0.) \
			+mu6*beta4*scipy.interp(k,self['k'],pk[7],left=0.,right=0.) \
			+mu8*beta4*scipy.interp(k,self['k'],pk[8],left=0.,right=0.)

class B2Loop(B1Loop):
	
	
	def run_terms(self,nthreads=8):
		
		self.spectrum_1loop_dt = Spectrum1Loop()
		self.spectrum_1loop_dt.set_terms(self.spectrum_lin.k)
		self.spectrum_1loop_dt.run_terms(a='delta',b='theta',nthreads=nthreads)

		self.spectrum_1loop_tt = Spectrum1Loop()
		self.spectrum_1loop_tt.set_terms(self.spectrum_lin.k)
		self.spectrum_1loop_tt.run_terms(a='theta',b='theta',nthreads=nthreads)
		
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.spectrum_lin.size,))
		self.regpt.set_spectra_B.argtypes = (ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t,pointer,pointer)
		
		pk_dt = self.spectrum_1loop_dt.pk() # why do I have to do that?
		pk_tt = self.spectrum_1loop_tt.pk()
		
		self.regpt.set_spectra_B('delta','theta',self.spectrum_lin.size,self.spectrum_lin.k,pk_dt)
		self.regpt.set_spectra_B('theta','theta',self.spectrum_lin.size,self.spectrum_lin.k,pk_tt)
			
		self.regpt.run_terms_B.argtypes = (ctypes.c_size_t,)
		self.regpt.run_terms_B(nthreads)
