# coding: utf8

import os
import ctypes
import scipy
from numpy import ctypeslib

def transfer_nowiggle(k,z=0.,h=0.7,Omegab0=0.05,Omegam0=0.3,Tcmb0=2.7):
	#Fitting formula for no-wiggle P(k) (Eq.[29] of Eisenstein and Hu 1998)
	omegab0 = Omegab0 * h ** 2
	omegam0 = Omegam0 * h ** 2
	theta_cmb = cosmo.Tcmb0 / 2.7
	sound_horizon = 44.5 * h * scipy.log(9.83/omegam0) / scipy.sqrt(1. + 10.*omegab0**0.75)
	alpha_gamma = 1. - 0.328 * scipy.log(431.*omegam0) * frac_baryon + 0.38 * scipy.log(22.3*omegam0) * frac_baryon**2
	k_eq = 0.0746 * omegam0 * theta_cmb ** (-2)
	k = k * h
	ks = k * sound_horizon / h
	q = k / (13.41*k_eq)
	gamma_eff = omegam0 * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
	q_eff = q * omegam0 / gamma_eff
	L0 = scipy.log(2*scipy.e + 1.8 * q_eff)
	C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
	return L0 / (L0 + C0 * q_eff**2)

def pk_nowiggle(k,pk,ns=1.,**kwargs):
	pknowiggle = k**ns*transfer_nowiggle(k,**kwargs)
	return pknowiggle*pk[0]/pknowiggle[0]
	

class Terms(object):
	
	def __init__(self,columns={},fields=None):
	
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
		return len(self[self.fields[0]])

	@property
	def size(self):
		return len(self)
		
	def __str__(self):
		return str(self.__dict__)
	
	def __setattr__(self,name,item):
		if name not in ['columns']:
			self.columns[name] = item
		else:
			object.__setattr__(self,name,item)
	
	def __getattribute__(self,name):
		columns = object.__getattribute__(self,'columns')
		if name in columns: return columns[name]
		return object.__getattribute__(self,name)
	
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

class PyRegPT(object):

	C_TYPE = ctypes.c_double
	PATH_CUBA = os.path.join(os.getenv('CUBA'),'libcuba.so')
	PATH_REGPT = os.path.join(os.path.dirname(os.path.realpath(__file__)),'regpt.so')
	KEYS_PK_LIN = ['k','pk_lin']
	KEYS_2LOOP = ['k','pk_lin','sigma_v2','G1a_1loop','G1a_2loop','G1b_1loop','G1b_2loop','pkcorr_G2_tree_tree','pkcorr_G2_tree_1loop','pkcorr_G2_1loop_1loop','pkcorr_G3_tree']
	KEYS_BIAS = ['k','pk_lin','pkbias_b2d','pkbias_bs2d','pkbias_b2t','pkbias_bs2t','pkbias_b22','pkbias_b2s2','pkbias_bs22','sigma3sq']
	KEYS_A_B = ['k','pk_lin','A','B']
	
	def __init__(self):

		cuba = ctypes.CDLL(self.PATH_CUBA,mode=ctypes.RTLD_GLOBAL)
		self.regpt = ctypes.CDLL(self.PATH_REGPT,mode=ctypes.RTLD_GLOBAL)
		self.pk_lin = Terms()
		self.terms_2loop = Terms()
		self.terms_bias = Terms()
		self.terms_A_B = Terms()

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
		
	def interpol_pk_lin(self,k):
	
		k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		pk = scipy.zeros_like(k)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(len(k),))
		self.regpt.interpol_pk_lin.argtypes = (pointer,pointer,ctypes.c_size_t)
		self.regpt.interpol_pk_lin(k,pk,len(k))
		
		return pk
		
	def find_pk_lin(self,k,mode=1):
	
		k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		pk = scipy.zeros_like(k)
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(len(k),))
		self.regpt.find_pk_lin.argtypes = (pointer,pointer,ctypes.c_size_t,ctypes.c_size_t)
		self.regpt.find_pk_lin(k,pk,len(k),mode)
		
		return pk
	
	def set_pk_lin(self,k,pk):
		
		self.pk_lin.k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		self.pk_lin.pk_lin = scipy.asarray(pk,dtype=self.C_TYPE).flatten()

		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.pk_lin.size,))
		self.regpt.set_pk_lin.argtypes = (pointer,pointer,ctypes.c_size_t)
		self.regpt.set_pk_lin(self.pk_lin.k,self.pk_lin.pk_lin,self.pk_lin.size)
		#self.regpt.set_pk_lin(k,pk,self.pk_lin.size)

	def set_precision(self,calculation,n=100,interpol='poly'):
	
		func = getattr(self.regpt,'set_precision_'+calculation)
		if calculation == 'pk_lin':
			func.argtypes = (ctypes.c_char_p,)
			func(interpol)
		else:
			func.argtypes = (ctypes.c_size_t,ctypes.c_char_p)
			func(n,interpol)
	
	def set_k_2loop(self,k):
		
		k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		for key in self.KEYS_2LOOP: self.terms_2loop[key] = scipy.zeros_like(k)
		self.terms_2loop.k = k
		
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.terms_2loop.size,))
		self.regpt.set_k_2loop.argtypes = (ctypes.c_size_t,)+(pointer,)*len(self.KEYS_2LOOP)
		self.regpt.set_k_2loop(self.terms_2loop.size,*[self.terms_2loop[key] for key in self.KEYS_2LOOP])
	
	def run_2loop(self,a='delta',b='delta',nthreads=8):
		
		self.regpt.run_2loop.argtypes = (ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t)
		self.regpt.run_2loop(a,b,nthreads)
	
	def set_k_bias(self,k):
		
		k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		for key in self.KEYS_BIAS: self.terms_bias[key] = scipy.zeros_like(k)
		self.terms_bias.k = k
		
		pointer = ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.terms_bias.size,))
		self.regpt.set_k_bias.argtypes = (ctypes.c_size_t,)+(pointer,)*len(self.KEYS_BIAS)
		self.regpt.set_k_bias(self.terms_bias.size,*[self.terms_bias[key] for key in self.KEYS_BIAS])
		
	def run_bias(self,nthreads=8):
		
		self.regpt.run_bias.argtypes = (ctypes.c_size_t,)
		self.regpt.run_bias(nthreads)
		
	def set_k_A_B(self,k):
		
		self.terms_A_B.k = scipy.asarray(k,dtype=self.C_TYPE).flatten()
		self.terms_A_B.pk_lin = scipy.zeros_like(k)
		self.terms_A_B.A = scipy.zeros((self.terms_A_B.k.size,3,3),dtype=self.C_TYPE).flatten()
		self.terms_A_B.B = scipy.zeros((self.terms_A_B.k.size,4,3),dtype=self.C_TYPE).flatten()
		
		pointers = [ctypeslib.ndpointer(dtype=self.C_TYPE,shape=(self.terms_A_B[key].size,)) for key in self.KEYS_A_B]
		self.regpt.set_k_A_B.argtypes = (ctypes.c_size_t,)+tuple(pointers)
		self.regpt.set_k_A_B(self.terms_A_B.k.size,*[self.terms_A_B[key] for key in self.KEYS_A_B])
		
		self.terms_A_B.A.shape = (self.terms_A_B.k.size,3,3)
		self.terms_A_B.A = scipy.transpose(self.terms_A_B.A,axes=(1,2,0))
		self.terms_A_B.B.shape = (self.terms_A_B.k.size,4,3)
		self.terms_A_B.B = scipy.transpose(self.terms_A_B.B,axes=(1,2,0))
		
	def run_A_B(self,nthreads=8):
		
		self.regpt.run_A_B.argtypes = (ctypes.c_size_t,)
		self.regpt.run_A_B(nthreads)
	
	def pk_2loop(self,Dgrowth=1.):
		#Dgrowth is Dgrowth or sigma8...
		#Taruya 2012 (arXiv 1208.1191v1) eq 24
		factor = 0.5 * (self.terms_2loop['k']*Dgrowth)**2 * self.terms_2loop['sigma_v2']
		G1a_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + 0.5*factor**2 + Dgrowth**2*self.terms_2loop['G1a_1loop']*(1. + factor) + Dgrowth**4*self.terms_2loop['G1a_2loop'])
		G1b_reg = Dgrowth * scipy.exp(-factor) * (1. + factor + 0.5*factor**2 + Dgrowth**2*self.terms_2loop['G1b_1loop']*(1. + factor) + Dgrowth**4*self.terms_2loop['G1b_2loop'])
		#Taruya 2012 (arXiv 1208.1191v1) first term of eq 23
		pkcorr_G1 = G1a_reg * G1b_reg * self.terms_2loop['pk_lin']
		#Taruya 2012 (arXiv 1208.1191v1) second term of eq 23
		pkcorr_G2 = Dgrowth**4 * scipy.exp(-2. * factor) * (self.terms_2loop['pkcorr_G2_tree_tree'] * (1. + factor)**2 + self.terms_2loop['pkcorr_G2_tree_1loop'] * Dgrowth**2 * (1. + factor) + self.terms_2loop['pkcorr_G2_1loop_1loop'] * Dgrowth**4)
		#Taruya 2012 (arXiv 1208.1191v1) third term of eq 23
		pkcorr_G3 = Dgrowth**6 * scipy.exp(-2. * factor) * self.terms_2loop['pkcorr_G3_tree']
		return pkcorr_G1 + pkcorr_G2 + pkcorr_G3
