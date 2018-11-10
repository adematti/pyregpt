#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "cuba.h"
#include "define.h"
#include "common.h"
#include "kernels.h"

#define NDIM 5
#define NCOMP 12
#define EPSREL 1e-6
#define EPSABS 1e-13
#define VERBOSE 0
#define LAST 4
#define MINEVAL 0
#define MAXEVAL 110000000
#define NNEW 2000000
#define FLATNESS 5.

static histo_t k,logqmin,logqmax;
static Precision precision_q = {.n=0,.min=5e-4,.max=10.,.interpol=POLY};
static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;

static void kernel_A_tA_2loop_II_III(const int *ndim, const double xx[],const int *ncomp, double ff[])
{
	size_t ii;
	histo_t okk[3],pp[3],qq[3],kkpp[3];
	histo_t p = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[0]));
	histo_t x = p/k;
	histo_t mu = -1. + 2.*xx[1];
	if (mu>0.5) {
		for (ii=0;ii<NCOMP;ii++) ff[ii] = 0.;
		return;
	}
	pp[0] = p * my_sqrt(1. -mu * mu);
	pp[1] = 0.;
	pp[2] = p * mu;
	histo_t q = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[2]));
	histo_t theta = ((histo_t) xx[3]) * M_PI;
	histo_t sintheta = my_sin(theta);
	histo_t phi = ((histo_t) xx[4]) * 2.* M_PI;
	qq[0] = q * sintheta * my_cos(phi);
	qq[1] = q * sintheta * my_sin(phi);
	qq[2] = q * my_cos(theta);
	okk[0] = 0.;
	okk[1] = 0.;
	okk[2] = -k;
	for (ii=0;ii<3;ii++) kkpp[ii] = -okk[ii]-pp[ii];
	histo_t kp = k * my_sqrt(1. + x*x - 2.*x*mu);
	
	histo_t sigmav2k,sigmav2p,sigmav2kp;
	calc_running_sigma_v2(&k,&sigmav2k,1,uvcutoff);
	calc_running_sigma_v2(&p,&sigmav2p,1,uvcutoff);
	calc_running_sigma_v2(&kp,&sigmav2kp,1,uvcutoff);
	
	histo_t exp_factor = (k*k*sigmav2k + p*p*sigmav2p + kp*kp*sigmav2kp) / 2.;
	if(exp_factor>1e2) {
		for (ii=0;ii<NCOMP;ii++) ff[ii] = 0.;
		return;
	}
	
	histo_t b211A_II, b211A_III, b221A_II, b221A_III, b212A_II, b212A_III, b222A_II, b222A_III;
	histo_t b211tA_II, b211tA_III, b221tA_II, b221tA_III, b212tA_II, b212tA_III, b222tA_II, b222tA_III;
	
	bispectrum_1loop_II_III(pp, kkpp, okk, qq, THETA, DELTA, DELTA, &b211A_II, &b211A_III);
	bispectrum_1loop_II_III(pp, kkpp, okk, qq, THETA, THETA, DELTA, &b221A_II, &b221A_III);
    bispectrum_1loop_II_III(pp, kkpp, okk, qq, THETA, DELTA, THETA, &b212A_II, &b212A_III);
    bispectrum_1loop_II_III(pp, kkpp, okk, qq, THETA, THETA, THETA, &b222A_II, &b222A_III);
    bispectrum_1loop_II_III(kkpp, pp, okk, qq, THETA, DELTA, DELTA, &b211tA_II, &b211tA_III);
    bispectrum_1loop_II_III(kkpp, pp, okk, qq, THETA, THETA, DELTA, &b221tA_II, &b221tA_III);
	bispectrum_1loop_II_III(kkpp, pp, okk, qq, THETA, DELTA, THETA, &b212tA_II, &b212tA_III);
	bispectrum_1loop_II_III(kkpp, pp, okk, qq, THETA, THETA, THETA, &b222tA_II, &b222tA_III);
	
	histo_t jacobian = (logqmax-logqmin) * M_PI * 2. * M_PI;
	exp_factor = my_exp(-exp_factor)*x*q*q*q*sintheta*jacobian;
	histo_t kernel_fp_A[6],kernel_fp_tA[6];
	fp_A(x,mu,kernel_fp_A,kernel_fp_tA);
	
	for (ii=0;ii<6;ii++) {
		ff[ii] = kernel_fp_A[ii] * exp_factor;
		ff[ii+6] = kernel_fp_tA[ii] * exp_factor;
	}
	
	ff[0] *= b211A_II + b211A_III;
	ff[1] *= b221A_II + b221A_III;
	ff[2] *= b221A_II + b221A_III;
	ff[3] *= b212A_II + b212A_III;
    ff[4] *= b222A_II + b222A_III;
    ff[5] *= b222A_II + b222A_III;
    
    ff[6] *= b211tA_II + b211tA_III;
    ff[7] *= b221tA_II + b221tA_III;
	ff[8] *= b221tA_II + b221tA_III;
	ff[9] *= b212tA_II + b212tA_III;
	ff[10] *= b222tA_II + b222tA_III;
	ff[11] *= b222tA_II + b222tA_III;
}


void set_precision_A_2loop_II_III_q(histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,0,min_,max_,interpol_);
}

void init_A_2loop_II_III()
{
	if ((precision_q.min<=0.)||(precision_q.max<=0.)) {
		precision_q.min = pk_lin.k[0];
		precision_q.max = pk_lin.k[pk_lin.nk-1];
	}
	logqmin = my_log(precision_q.min);
	logqmax = my_log(precision_q.max);
	get_precision_bispectrum_1loop_pk_lin(&interpol_pk_lin);
	get_running_uvcutoff_bispectrum_1loop(&uvcutoff);
}

void free_A_2loop_II_III() {}


void calc_pkcorr_from_A_2loop_II_III(histo_t k_,histo_t* pkcorr_A)
{
	k = k_;
	
	int nregions,neval,fail;
	double integral[NCOMP]={0.}, error[NCOMP]={0.}, prob[NCOMP]={0.};
	
	Suave(NDIM, NCOMP, kernel_A_tA_2loop_II_III, EPSREL, EPSABS, VERBOSE + LAST, MINEVAL, MAXEVAL, NNEW, FLATNESS, &nregions, &neval, &fail, integral, error, prob);
	
	histo_t factor = k*k*k / (2.*M_PI*M_PI);
	size_t ii;
	for (ii=0;ii<NCOMP;ii++) pkcorr_A[ii] = integral[ii]*factor;

}


