#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuba.h"
#include "define.h"
#include "common.h"
#include "kernels.h"

static histo_t k,logqmin,logqmax;
static const Precision precision_q_default = {.n=0,.min=5e-4,.max=10.,.interpol=POLY};
static Precision precision_q = {.n=0,.min=5e-4,.max=10.,.interpol=POLY};
static FLAG a,b;
#pragma omp threadprivate(k)

void set_precision_gamma3_tree_q(histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,0,min_,max_,interpol_,&precision_q_default);
}

void init_gamma3_tree()
{
	if ((precision_q.min<=0.)||(precision_q.max<=0.)) {
		precision_q.min = pk_lin.k[0];
		precision_q.max = pk_lin.k[pk_lin.nk-1];
	}
	logqmin = my_log(precision_q.min);
	logqmax = my_log(precision_q.max);
}

void free_gamma3_tree() {}

#ifdef _CUBA15

static void kernel_pk_gamma3_tree(const int *ndim, const double xx[],const int *ncomp, double ff[])
{

	histo_t f;
	histo_t pp[3],qq[3],kpq[3],kp[3];
	histo_t p = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[0]));
	histo_t theta1 = ((histo_t) xx[1]) * M_PI;
	histo_t phi1 = ((histo_t) xx[2]) * 2. * M_PI;
	histo_t q = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[3]));
	histo_t theta2 = ((histo_t) xx[4]) * M_PI;

	histo_t sintheta1 = my_sin(theta1);
	histo_t sintheta2 = my_sin(theta2);
	pp[0] = p * sintheta1 * my_cos(phi1);
	pp[1] = p * sintheta1 * my_sin(phi1);
	pp[2] = p * my_cos(theta1);
	qq[0] = q * sintheta2;
	qq[1] = 0.;
	qq[2] = q * my_cos(theta2); 
	kpq[0] = -pp[0]-qq[0];
	kpq[1] = -pp[1]-qq[1];
	kpq[2] = k-pp[2]-qq[2];

	histo_t k_p_q = my_sqrt(kpq[0]*kpq[0] + kpq[1]*kpq[1] + kpq[2]*kpq[2]);
	
	kp[0] = -pp[0];
	kp[1] = -pp[1];
	kp[2] = k-pp[2];
	histo_t k_p = my_sqrt(kp[0]*kp[0] + kp[1]*kp[1] + kp[2]*kp[2]);
	histo_t prod_q_kp = qq[0]*kp[0] + qq[1]*kp[1] + qq[2]*kp[2];
	
	ff[0] = 0.;
	if ((k_p_q>=precision_q.min)&&(k_p_q<=precision_q.max)&&(prod_q_kp<=0.5*k_p*k_p)) {
		histo_t pk_p,pk_q,pk_kpq;
		find_pk_lin(&p,&pk_p,1,precision_q.interpol);
		find_pk_lin(&q,&pk_q,1,precision_q.interpol);
		find_pk_lin(&k_p_q,&pk_kpq,1,precision_q.interpol);
		f = F3_sym(a,pp,qq,kpq) * F3_sym(b,pp,qq,kpq) * pk_p * pk_q * pk_kpq;
		histo_t jacobian = (logqmax-logqmin) * M_PI * 2. * M_PI;
		jacobian *= jacobian * sintheta1 * sintheta2;
		f *= p*p*p*q*q*q * jacobian;
		ff[0] = (double) (2.*f);
	}
}

#define NDIM 5
#define NCOMP 1
#define EPSREL 0.005
#define EPSABS 1e-12
#define VERBOSE 0
#define MINEVAL 0
#define MAXEVAL 110000000
#define NSTART 4000
#define NINCREASE 700

void calc_pk_gamma3_tree(FLAG a_, FLAG b_, histo_t k_, histo_t *pk_gamma3_tree)
{
	a = a_;
	b = b_;
	k = k_;
	
	int neval, fail;
	double integral[NCOMP]={0.}, error[NCOMP]={0.}, prob[NCOMP]={0.};

	Vegas(NDIM, NCOMP, kernel_pk_gamma3_tree, EPSREL, EPSABS, VERBOSE, MINEVAL, MAXEVAL, NSTART, NINCREASE, &neval, &fail, integral, error, prob);
	
	*pk_gamma3_tree = 6.*((histo_t) integral[0])/power(2.*M_PI,6);

}

#else //_CUBA15

static int kernel_pk_gamma3_tree(const int *ndim, const double xx[],const int *ncomp, double ff[], void *userdata)
{
	histo_t f;
	histo_t pp[3],qq[3],kpq[3],kp[3];
	histo_t p = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[0]));
	histo_t theta1 = ((histo_t) xx[1]) * M_PI;
	histo_t phi1 = ((histo_t) xx[2]) * 2. * M_PI;
	histo_t q = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[3]));
	histo_t theta2 = ((histo_t) xx[4]) * M_PI;

	histo_t sintheta1 = my_sin(theta1);
	histo_t sintheta2 = my_sin(theta2);
	pp[0] = p * sintheta1 * my_cos(phi1);
	pp[1] = p * sintheta1 * my_sin(phi1);
	pp[2] = p * my_cos(theta1);
	qq[0] = q * sintheta2;
	qq[1] = 0.;
	qq[2] = q * my_cos(theta2); 
	kpq[0] = -pp[0]-qq[0];
	kpq[1] = -pp[1]-qq[1];
	kpq[2] = k-pp[2]-qq[2];

	histo_t k_p_q = my_sqrt(kpq[0]*kpq[0] + kpq[1]*kpq[1] + kpq[2]*kpq[2]);
	
	kp[0] = -pp[0];
	kp[1] = -pp[1];
	kp[2] = k-pp[2];
	histo_t k_p = my_sqrt(kp[0]*kp[0] + kp[1]*kp[1] + kp[2]*kp[2]);
	histo_t prod_q_kp = qq[0]*kp[0] + qq[1]*kp[1] + qq[2]*kp[2];
	
	ff[0] = 0.;
	if ((k_p_q>=precision_q.min)&&(k_p_q<=precision_q.max)&&(prod_q_kp<=0.5*k_p*k_p)) {
		histo_t pk_p,pk_q,pk_kpq;
		find_pk_lin(&p,&pk_p,1,precision_q.interpol);
		find_pk_lin(&q,&pk_q,1,precision_q.interpol);
		find_pk_lin(&k_p_q,&pk_kpq,1,precision_q.interpol);
		f = F3_sym(a,pp,qq,kpq) * F3_sym(b,pp,qq,kpq) * pk_p * pk_q * pk_kpq;
		histo_t jacobian = (logqmax-logqmin) * M_PI * 2. * M_PI;
		jacobian *= jacobian * sintheta1 * sintheta2;
		f *= p*p*p*q*q*q * jacobian;
		ff[0] = (double) (2.*f);
	}
	
	return 0;
}

/*
#define NDIM 5
#define NCOMP 1
#define USERDATA NULL
#define NVEC 1
//#define EPSREL 0.005
#define EPSREL 1e-4
#define EPSABS 1e-12
#define VERBOSE 0
#define SEED 0
#define MINEVAL 0
#define MAXEVAL 110000000
#define NSTART 4000
#define NINCREASE 700
#define NBATCH 1000
#define gammaRIDNO 0
#define STATEFILE NULL
#define SM_PIN NULL


void calc_pk_gamma3_tree(FLAG a_, FLAG b_, histo_t k_, histo_t *pk_gamma3_tree)
{
	a = a_;
	b = b_;
	k = k_;
	
	int neval, fail;
	double integral[NCOMP]={0.}, error[NCOMP]={0.}, prob[NCOMP]={0.};

	Vegas(NDIM, NCOMP, kernel_pk_gamma3_tree, USERDATA, NVEC, EPSREL, EPSABS, VERBOSE, SEED, MINEVAL, MAXEVAL, NSTART, NINCREASE, NBATCH, gammaRIDNO, STATEFILE, SM_PIN, &neval, &fail, integral, error, prob);
	
	*pk_gamma3_tree = 6.*((histo_t) integral[0])/power(2.*M_PI,6);

}
*/

#define NCOMP 1
#define NDIM 5
#define USERDATA NULL
#define NVEC 1
#define EPSREL 1e-6
#define EPSABS 1e-13
#define VERBOSE 0
#define LAST 4
#define SEED 0
#define MINEVAL 0
#define MAXEVAL 3000000
#define NNEW 2000000
#define NMIN 2
#define FLATNESS 5.
#define STATEFILE NULL
#define SPIN NULL

void calc_pk_gamma3_tree(FLAG a_, FLAG b_, histo_t k_, histo_t *pk_gamma3_tree)
{
	a = a_;
	b = b_;
	k = k_;
	
	int nregions,neval,fail;
	double integral[NCOMP]={0.}, error[NCOMP]={0.}, prob[NCOMP]={0.};

	Suave(NDIM, NCOMP, kernel_pk_gamma3_tree, USERDATA, NVEC, EPSREL, EPSABS, VERBOSE | LAST, SEED, MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS, STATEFILE, SPIN, &nregions, &neval, &fail, integral, error, prob);
	
	*pk_gamma3_tree = 6.*((histo_t) integral[0])/power(2.*M_PI,6);

}

#endif //_CUBA15
