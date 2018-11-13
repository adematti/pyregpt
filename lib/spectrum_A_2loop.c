#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"

#define NCOMP 5
#define STEP_VERBOSE 10

static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;
static TermsA terms_A;

void set_precision_A_2loop_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

void set_precision_A_2loop_sigma_v2(histo_t uvcutoff_)
{
	uvcutoff = uvcutoff_;
}

void set_terms_A_2loop(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* A)
{
	terms_A.nk = nk;
	terms_A.k = k;
	terms_A.pk_lin = pk_lin;
	terms_A.sigma_v2 = sigma_v2;
	terms_A.A = A;
	
#ifdef _VERBOSE
	print_k(terms_A.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void run_terms_A_2loop(size_t num_threads)
{

#ifdef _VERBOSE
	printf("*** Calculation of A term at 2 loop\n");
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_A.nk*STEP_VERBOSE/100,1);
	histo_t pkcorr_A[12];
		
	timer(0);
	find_pk_lin(terms_A.k,terms_A.pk_lin,terms_A.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_A.k,terms_A.sigma_v2,terms_A.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_A,step_verbose) private(ik,pkcorr_A)
	{
		init_A_2loop_I();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_A.nk;ik++) {
			histo_t k = terms_A.k[ik];
#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			calc_pkcorr_from_A_2loop_I(k,pkcorr_A);
			terms_A.A[ik*NCOMP] = pkcorr_A[0] + pkcorr_A[6];								//pk_A111 + pk_tA111, mu^2 * f
			terms_A.A[ik*NCOMP+1] = pkcorr_A[1] + pkcorr_A[7];								//pk_A121 + pk_tA121, mu^2 * f^2
			terms_A.A[ik*NCOMP+2] = pkcorr_A[2] + pkcorr_A[8] + pkcorr_A[3] + pkcorr_A[9];	//pk_A221 + pk_A212 + pk_tA221 + pk_tA212, mu^4 * f^3
			terms_A.A[ik*NCOMP+3] = pkcorr_A[4] + pkcorr_A[10];								//pk_A222 + pk_tA222, mu^4 * f^3
			terms_A.A[ik*NCOMP+4] = pkcorr_A[5] + pkcorr_A[11];								//pk_A322 + pk_tA322, mu^6 * f^3
			
			//size_t ii;
			//for (ii=0;ii<NCOMP;ii++) terms_A.A[ik*NCOMP+ii] = 0.;
		}
#pragma omp critical
		{
			free_A_2loop_I();
		}
	}
	init_A_2loop_II_III();
	for (ik=0;ik<terms_A.nk;ik++) {
		histo_t k = terms_A.k[ik];
		calc_pkcorr_from_A_2loop_II_III(k,pkcorr_A);
		terms_A.A[ik*NCOMP] += pkcorr_A[0] + pkcorr_A[6];									//pk_A111 + pk_tA111, mu^2 * f
		terms_A.A[ik*NCOMP+1] += pkcorr_A[1] + pkcorr_A[7];									//pk_A121 + pk_tA121, mu^2 * f^2
		terms_A.A[ik*NCOMP+2] += pkcorr_A[2] + pkcorr_A[8] + pkcorr_A[3] + pkcorr_A[9];		//pk_A221 + pk_A212 + pk_tA221 + pk_tA212, mu^4 * f^3
		terms_A.A[ik*NCOMP+3] += pkcorr_A[4] + pkcorr_A[10];								//pk_A222 + pk_tA222, mu^4 * f^3
		terms_A.A[ik*NCOMP+4] += pkcorr_A[5] + pkcorr_A[11];								//pk_A322 + pk_tA322, mu^6 * f^3
	}
	free_A_2loop_II_III();
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
}

/*
void run_terms_A_2loop(size_t num_threads)
{

#ifdef _VERBOSE
	printf("*** Calculation of A term at 2 loop\n");
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_A.nk*STEP_VERBOSE/100,1);
	histo_t pkcorr_A[12];
		
	timer(0);
	find_pk_lin(terms_A.k,terms_A.pk_lin,terms_A.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_A.k,terms_A.sigma_v2,terms_A.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_A,step_verbose) private(ik,pkcorr_A)
	{
		init_A_2loop_I();
		init_A_2loop_II_III();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_A.nk;ik++) {
			histo_t k = terms_A.k[ik];
#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			calc_pkcorr_from_A_2loop_I(k,pkcorr_A);
			terms_A.A[ik*NCOMP] = pkcorr_A[0] + pkcorr_A[6];								//pk_A111 + pk_tA111, mu^2 * f
			terms_A.A[ik*NCOMP+1] = pkcorr_A[1] + pkcorr_A[7];								//pk_A121 + pk_tA121, mu^2 * f^2
			terms_A.A[ik*NCOMP+2] = pkcorr_A[2] + pkcorr_A[8] + pkcorr_A[3] + pkcorr_A[9];	//pk_A221 + pk_A212 + pk_tA221 + pk_tA212, mu^4 * f^3
			terms_A.A[ik*NCOMP+3] = pkcorr_A[4] + pkcorr_A[10];								//pk_A222 + pk_tA222, mu^4 * f^3
			terms_A.A[ik*NCOMP+4] = pkcorr_A[5] + pkcorr_A[11];								//pk_A322 + pk_tA322, mu^6 * f^3
			calc_pkcorr_from_A_2loop_II_III(k,pkcorr_A);
			terms_A.A[ik*NCOMP] += pkcorr_A[0] + pkcorr_A[6];									//pk_A111 + pk_tA111, mu^2 * f
			terms_A.A[ik*NCOMP+1] += pkcorr_A[1] + pkcorr_A[7];									//pk_A121 + pk_tA121, mu^2 * f^2
			terms_A.A[ik*NCOMP+2] += pkcorr_A[2] + pkcorr_A[8] + pkcorr_A[3] + pkcorr_A[9];		//pk_A221 + pk_A212 + pk_tA221 + pk_tA212, mu^4 * f^3
			terms_A.A[ik*NCOMP+3] += pkcorr_A[4] + pkcorr_A[10];								//pk_A222 + pk_tA222, mu^4 * f^3
			terms_A.A[ik*NCOMP+4] += pkcorr_A[5] + pkcorr_A[11];								//pk_A322 + pk_tA322, mu^6 * f^3
		}
#pragma omp critical
		{
			free_A_2loop_I();
			free_A_2loop_II_III();
		}
	}
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
}
*/

