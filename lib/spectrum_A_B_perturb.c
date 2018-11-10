#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"
#include "kernels.h"

#define STEP_VERBOSE 10

static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;
static TermsAB terms_A_B;

void set_terms_A_B(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* A,histo_t* B)
{
	terms_A_B.nk = nk;
	terms_A_B.k = k;
	terms_A_B.pk_lin = pk_lin;
	terms_A_B.sigma_v2 = sigma_v2;
	terms_A_B.A = A;
	terms_A_B.B = B;
	
#ifdef _VERBOSE
	print_k(terms_A_B.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void run_A_B(size_t num_threads)
{

#ifdef _VERBOSE
	printf("*** Calculation of A and B terms\n");
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_A_B.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_A_B.k,terms_A_B.pk_lin,terms_A_B.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_A_B.k,terms_A_B.sigma_v2,terms_A_B.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_A_B,step_verbose) private(ik)
	{
		init_A_B();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_A_B.nk;ik++) {
			histo_t k = terms_A_B.k[ik];
#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			calc_pkcorr_from_A(k,terms_A_B.pk_lin[ik],&(terms_A_B.A[ik*9]));
			calc_pkcorr_from_B(k,&(terms_A_B.B[ik*12]));
		}
#pragma omp critical
		{
			free_A_B();
		}
	}
	
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
}

