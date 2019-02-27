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

#define NCOMP 12

/*
#define NSTART 4000
#define NINCREASE 700
*/

static histo_t k,logqmin,logqmax;
static const Precision precision_q_default = {.n=0,.min=5e-4,.max=10.,.interpol=POLY};
static Precision precision_q = {.n=0,.min=5e-4,.max=10.,.interpol=POLY};
static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;
#pragma omp threadprivate(k)

void set_precision_A_2loop_II_III_q(histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,0,min_,max_,interpol_,&precision_q_default);
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

#ifdef _CUBA15

static void kernel_A_tA_2loop_II_III(const int *ndim, const double xx[],const int *ncomp, double ff[])
{
	size_t ii;
	histo_t okk[3],pp[3],qq[3],kkpp[3];
	
	histo_t p = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[0]));
	histo_t x = p/k;
	histo_t mu = -1. + 2.*((histo_t) xx[1]);
	if (mu>0.5/x) {
		for (ii=0;ii<NCOMP;ii++) ff[ii] = 0.;
		return;
	}
	pp[0] = p * my_sqrt(1. - mu*mu);
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
	
	histo_t sigmad2k,sigmad2p,sigmad2kp;
	calc_running_sigmad2(&k,&sigmad2k,1,uvcutoff);
	calc_running_sigmad2(&p,&sigmad2p,1,uvcutoff);
	calc_running_sigmad2(&kp,&sigmad2kp,1,uvcutoff);
	/*
	sigmad2k = 1.;
	sigmad2p = 1.;
	sigmad2kp = 1.;
	*/
	histo_t exp_factor = (k*k*sigmad2k + p*p*sigmad2p + kp*kp*sigmad2kp) / 2.;
	if (exp_factor>=1e2) {
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
	
	histo_t jacobian = (logqmax-logqmin) * (logqmax-logqmin) * 4 * M_PI * M_PI;
	exp_factor = my_exp(-exp_factor)*x*q*q*q*sintheta*jacobian;
	
	histo_t kernel_projection_A[6],kernel_projection_tA[6];
	kernel_projection_A_tA(x,mu,kernel_projection_A,kernel_projection_tA);
	
	for (ii=0;ii<6;ii++) {
		ff[ii] = ((double) kernel_projection_A[ii] * exp_factor);
		ff[ii+6] = ((double) kernel_projection_tA[ii] * exp_factor);
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

#define NDIM 5
#define EPSREL 1e-6
#define EPSABS 1e-13
#define VERBOSE 0
#define LAST 4
#define MINEVAL 0
#define MAXEVAL 3000000
#define NNEW 2000000
#define FLATNESS 5.

void calc_pk_A_2loop_II_III(histo_t k_,histo_t* pk_A)
{
	k = k_;
	
	int nregions,neval,fail;
	double integral[NCOMP]={0.}, error[NCOMP]={0.}, prob[NCOMP]={0.};
	/*
	double xx[5] = {3.1999999285e-01, 2.3000000417e-01,  2.0999999344e-01,  3.3000001311e-01,  1.0000000149e-01};
	const int ndim = 5;
	const int ncomp = 12;
	kernel_A_tA_2loop_II_III(&ndim,xx,&ncomp,integral);
	size_t ii;
	for (ii=0;ii<NCOMP;ii++) printf("%6f ",integral[ii]);
	*/
	Suave(NDIM, NCOMP, kernel_A_tA_2loop_II_III, EPSREL, EPSABS, VERBOSE | LAST, MINEVAL, MAXEVAL, NNEW, FLATNESS, &nregions, &neval, &fail, integral, error, prob);
	
	histo_t factor = k*k*k / (2.*M_PI*M_PI);
	size_t ii;
	for (ii=0;ii<NCOMP;ii++) pk_A[ii] = ((histo_t) integral[ii])*factor;

}

#else //_CUBA15

static int kernel_A_tA_2loop_II_III(const int *ndim, const double xx[],const int *ncomp, double ff[], void *userdata)
{
	size_t ii;
	histo_t okk[3],pp[3],qq[3],kkpp[3];
	
	histo_t p = my_exp(logqmin + (logqmax - logqmin) * ((histo_t) xx[0]));
	histo_t x = p/k;
	histo_t mu = -1. + 2.*((histo_t) xx[1]);
	if (mu>0.5/x) {
		for (ii=0;ii<NCOMP;ii++) ff[ii] = 0.;
		return 0;
	}
	pp[0] = p * my_sqrt(1. - mu*mu);
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
	
	histo_t sigmad2k,sigmad2p,sigmad2kp;
	calc_running_sigmad2(&k,&sigmad2k,1,uvcutoff);
	calc_running_sigmad2(&p,&sigmad2p,1,uvcutoff);
	calc_running_sigmad2(&kp,&sigmad2kp,1,uvcutoff);
	/*
	sigmad2k = 1.;
	sigmad2p = 1.;
	sigmad2kp = 1.;
	*/
	histo_t exp_factor = (k*k*sigmad2k + p*p*sigmad2p + kp*kp*sigmad2kp) / 2.;
	if (exp_factor>=1e2) {
		for (ii=0;ii<NCOMP;ii++) ff[ii] = 0.;
		return 0;
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
	
	histo_t jacobian = (logqmax-logqmin) * (logqmax-logqmin) * 4 * M_PI * M_PI;
	exp_factor = my_exp(-exp_factor)*x*q*q*q*sintheta*jacobian;
	
	histo_t kernel_projection_A[6],kernel_projection_tA[6];
	kernel_projection_A_tA(x,mu,kernel_projection_A,kernel_projection_tA);
	
	for (ii=0;ii<6;ii++) {
		ff[ii] = kernel_projection_A[ii] * exp_factor;
		ff[ii+6] = kernel_projection_tA[ii] * exp_factor;
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
	
	return 0;
}

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

void calc_pk_A_2loop_II_III(histo_t k_,histo_t* pk_A)
{
	k = k_;
	
	int nregions,neval,fail;
	double integral[NCOMP]={0.}, error[NCOMP]={0.}, prob[NCOMP]={0.};
	
	Suave(NDIM, NCOMP, kernel_A_tA_2loop_II_III, USERDATA, NVEC, EPSREL, EPSABS, VERBOSE | LAST, SEED, MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS, STATEFILE, SPIN, &nregions, &neval, &fail, integral, error, prob);
	
	histo_t factor = k*k*k / (2.*M_PI*M_PI);
	size_t ii;
	for (ii=0;ii<NCOMP;ii++) pk_A[ii] = ((histo_t) integral[ii])*factor;

}

#endif //_CUBA15
