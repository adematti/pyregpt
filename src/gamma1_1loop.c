#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"

static GaussLegendreQ gauss_legendre_q;
#pragma omp threadprivate(gauss_legendre_q)
static const Precision precision_q_default = {.n=600,.min=5e-4,.max=10.,.interpol=POLY};
static Precision precision_q = {.n=600,.min=5e-4,.max=10.,.interpol=POLY};

histo_t kernel_ff(histo_t x)
{
	histo_t kernel_ff,x2;
	histo_t diffx = x-1.;
	if (my_abs(diffx)<0.01) {
	 	kernel_ff = - 11./126. + diffx/126. - 29./252. * diffx*diffx;
	}
	else if (x>10.) {
		x2 = x*x;
		kernel_ff = - 61./630. + 2./105./x2 - 10./1323./x2/x2;
	}
	else {
		x2 = x*x;
		kernel_ff = 6./x2 - 79. + 50.*x2 - 21.*x2*x2 + 0.75*power(1./x-x,3) * (2. + 7.*x2) * 2 * my_log(my_abs(diffx/(1.+x)));
		kernel_ff /= 504.;
#ifdef _DEBUgamma
		//printf("%.10g %.10g %.10g\n",x,6./x2 - 79. + 50.*x2 - 21.*x2*x2,0.75*power(1./x-x,3) * (2. + 7.*x2) * 2 * my_log(my_abs(diffx/(1.+x))));
#endif //_DEBUgamma
	}
	return kernel_ff * x;
}


histo_t kernel_gg(histo_t x)
{
	histo_t kernel_gg,x2;
	histo_t diff = x-1.;
	if (my_abs(diff)<0.01) {
	 	kernel_gg = - 3./14. - 5./42.*diff + diff*diff/84.;
	}
	else if (x>10.) {
		x2 = x*x;
		kernel_gg = - 3./10. + 26./245./x2 - 38./2205./x2/x2;
	}
	else {
		x2 = x*x;
		kernel_gg = 6./x2 - 41. + 2.*x2 - 3.*x2*x2 + 0.75*power(1./x-x,3) * (2. + x2) * 2 * my_log(my_abs(diff/(1.+x)));
		kernel_gg /= 168.;
	}
	return kernel_gg * x;
}


histo_t gamma1_1loop(FLAG a,histo_t k)
{
	update_gauss_legendre_q(&gauss_legendre_q,k);

	size_t iq,nq = gauss_legendre_q.nq;
	histo_t integ_gamma = 0.;
	
	if (a==DELTA) {
		for (iq=0;iq<nq;iq++) integ_gamma += kernel_ff(gauss_legendre_q.x[iq])*gauss_legendre_q.pk[iq]*gauss_legendre_q.w[iq];
	}
	else { //THETA
		for (iq=0;iq<nq;iq++) integ_gamma += kernel_gg(gauss_legendre_q.x[iq])*gauss_legendre_q.pk[iq]*gauss_legendre_q.w[iq];
	}
	
	return integ_gamma * k*k*k/(2.*M_PI*M_PI);
}

void set_precision_gamma1_1loop_q(size_t n_,histo_t min_,histo_t max_,char* interpol_)
{
	set_precision(&precision_q,n_,min_,max_,interpol_,&precision_q_default);
}

void init_gamma1_1loop()
{
	init_gauss_legendre_q(&gauss_legendre_q,&precision_q);
}

void free_gamma1_1loop()
{
	free_gauss_legendre_q(&gauss_legendre_q);
}
