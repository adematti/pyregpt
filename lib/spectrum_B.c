#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"

#define NCOMP 9
#define STEP_VERBOSE 10

static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;
static TermsB terms_B;
static Pk pk_dt,pk_tt;

void set_precision_B_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

void set_running_uvcutoff_B(histo_t uvcutoff_)
{
	uvcutoff = uvcutoff_;
}


void write_spectra_B(char *fn)
{
	// Only used for debugging
	FILE *fr;
	size_t ik;
	fr=fopen(fn,"w");
	if (fr==NULL) error_open_file(fn);
	for (ik=0;ik<pk_dt.nk;ik++) {
		fprintf(fr,"%f %f %f\n",pk_dt.k[ik],pk_dt.pk[ik],pk_tt.pk[ik]);
	}
	fclose(fr);
}

void set_terms_B(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* B)
{
	terms_B.nk = nk;
	terms_B.k = k;
	terms_B.pk_lin = pk_lin;
	terms_B.sigma_v2 = sigma_v2;
	terms_B.B = B;
	
#ifdef _VERBOSE
	print_k(terms_B.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void set_spectra_B(char *flag, size_t nk, histo_t *k, histo_t *pk)
{
	Pk pk_ab;
	pk_ab.nk = nk;
	pk_ab.k = k;
	pk_ab.pk = pk;
	if (!strcmp(flag,"deltatheta")) pk_dt = pk_ab;
	if (!strcmp(flag,"thetatheta")) pk_tt = pk_ab;
}

static void run_spectra_B_1loop(size_t num_threads)
{
	size_t size = sizeof(histo_t);

	pk_dt.nk = pk_lin.nk;
	pk_dt.k = pk_lin.k;
	pk_dt.pk = (histo_t *) calloc(pk_dt.nk,size);
	
	set_spectrum_1loop(pk_dt.nk,pk_dt.k,pk_dt.pk);
	run_spectrum_1loop("delta","theta",num_threads);
	
	pk_tt.nk = pk_lin.nk;
	pk_tt.k = pk_lin.k;
	pk_tt.pk = (histo_t *) calloc(pk_tt.nk,size);
	
	set_spectrum_1loop(pk_tt.nk,pk_tt.k,pk_tt.pk);
	run_spectrum_1loop("theta","theta",num_threads);
#ifdef _DEBUG
	write_spectra_B("debug_spectra_B.dat");
#endif //_DEBUG
}

void free_spectra_B()
{
	free(pk_dt.pk);
	free(pk_tt.pk);
}

void run_terms_B(size_t num_threads)
{

#ifdef _VERBOSE
	printf("*** Calculation of B term\n");
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_B.nk*STEP_VERBOSE/100,1);
	histo_t pkcorr_B[12];
		
	timer(0);
	find_pk_lin(terms_B.k,terms_B.pk_lin,terms_B.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_B.k,terms_B.sigma_v2,terms_B.nk,uvcutoff);
	
	_Bool free_spectra = 0;
	if ((pk_dt.nk==0)||(pk_tt.nk==0)) {
		free_spectra = 1; 
		run_spectra_B_1loop(num_threads);
	}
#pragma omp parallel default(none) shared(terms_B,step_verbose,pk_dt,pk_tt) private(ik,pkcorr_B)
	{
		init_B(pk_dt,pk_tt);
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_B.nk;ik++) {
			histo_t k = terms_B.k[ik];
#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			calc_pkcorr_from_B(k,pkcorr_B);
			terms_B.B[ik*NCOMP] = pkcorr_B[0];					//pk_B111, mu^2 * f^2
			terms_B.B[ik*NCOMP+1] = - (pkcorr_B[1] + pkcorr_B[2]);	//pk_B112 + pk_B121, mu^2 * f^3 
			terms_B.B[ik*NCOMP+2] = pkcorr_B[3];				//pk_B122, mu^2 * f^4
			terms_B.B[ik*NCOMP+3] = pkcorr_B[4];				//pk_B211, mu^4 * f^2
			terms_B.B[ik*NCOMP+4] = - (pkcorr_B[5] + pkcorr_B[6]);	//pk_B212 + pk_B221, mu^4 * f^3
			terms_B.B[ik*NCOMP+5] = pkcorr_B[7];				//pk_B222, mu^4 * f^4
			terms_B.B[ik*NCOMP+6] = - (pkcorr_B[8] + pkcorr_B[9]);	//pk_B312 + pk_B321, mu^6 * f^3
			terms_B.B[ik*NCOMP+7] = pkcorr_B[10];				//pk_B322, mu^6 * f^4
			terms_B.B[ik*NCOMP+8] = pkcorr_B[11];				//pk_B422, mu^8 * f^4
		}
#pragma omp critical
		{
			free_B();
		}
	}
	if (free_spectra) free_spectra_B();
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
}
