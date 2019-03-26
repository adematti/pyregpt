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
static Pk pk_dt = {.nk=0};
static Pk pk_tt = {.nk=0};

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

void set_terms_B(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigmad2,histo_t* B)
{
	terms_B.nk = nk;
	terms_B.k = k;
	terms_B.pk_lin = pk_lin;
	terms_B.sigmad2 = sigmad2;
	terms_B.B = B;

	if (verbose == INFO) {
		print_k(terms_B.k,nk);
		printf("\n");
	}
}

void set_spectra_B(char* a_,char* b_, size_t nk, histo_t *k, histo_t *pk)
{
	Pk pk_ab;
	pk_ab.nk = nk;
	pk_ab.k = k;
	pk_ab.pk = pk;
	FLAG a = set_flag(a_);
	FLAG b = set_flag(b_);
	if (((a==DELTA)&&(b=THETA))||((a==THETA)&&(b==DELTA))) {
		if (verbose == INFO) printf(" - setting P_delta_theta\n");
		pk_dt = pk_ab;
	}
	if ((a==THETA)&&(b==THETA)) {
		if (verbose == INFO) printf(" - setting P_theta_theta\n");
		pk_tt = pk_ab;
	}
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
}

void free_spectra_B()
{
	free(pk_dt.pk);
	free(pk_tt.pk);
}

void run_terms_B(size_t num_threads)
{
	if (verbose == INFO) printf("*** Calculation of B term\n");
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_B.nk*STEP_VERBOSE/100,1);
	histo_t pk_B[12];
		
	timer(0);
	find_pk_lin(terms_B.k,terms_B.pk_lin,terms_B.nk,interpol_pk_lin);
	calc_running_sigmad2(terms_B.k,terms_B.sigmad2,terms_B.nk,uvcutoff);
	
	_Bool free_spectra = 0;
	if ((pk_dt.nk==0)||(pk_tt.nk==0)) {
		free_spectra = 1; 
		run_spectra_B_1loop(num_threads);
	}
	if (verbose == DEBUG) write_spectra_B("debug_spectra_B.dat");
#pragma omp parallel default(none) shared(terms_B,pk_dt,pk_tt,step_verbose,verbose) private(ik,pk_B)
	{
		init_B(pk_dt,pk_tt);
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_B.nk;ik++) {
			histo_t k = terms_B.k[ik];
			if ((verbose == INFO) && (ik % step_verbose == 0)) printf(" - computation done at %zu percent\n",ik*STEP_VERBOSE/step_verbose);
			calc_pk_B(k,pk_B);
			terms_B.B[ik*NCOMP] = pk_B[0];					//pk_B111, mu^2 * f^2
			terms_B.B[ik*NCOMP+1] = - (pk_B[1] + pk_B[2]);	//pk_B112 + pk_B121, mu^2 * f^3 
			terms_B.B[ik*NCOMP+2] = pk_B[3];				//pk_B122, mu^2 * f^4
			terms_B.B[ik*NCOMP+3] = pk_B[4];				//pk_B211, mu^4 * f^2
			terms_B.B[ik*NCOMP+4] = - (pk_B[5] + pk_B[6]);	//pk_B212 + pk_B221, mu^4 * f^3
			terms_B.B[ik*NCOMP+5] = pk_B[7];				//pk_B222, mu^4 * f^4
			terms_B.B[ik*NCOMP+6] = - (pk_B[8] + pk_B[9]);	//pk_B312 + pk_B321, mu^6 * f^3
			terms_B.B[ik*NCOMP+7] = pk_B[10];				//pk_B322, mu^6 * f^4
			terms_B.B[ik*NCOMP+8] = pk_B[11];				//pk_B422, mu^8 * f^4
		}
#pragma omp critical
		{
			free_B();
		}
	}
	if (free_spectra) free_spectra_B();
	if (verbose == INFO) timer(1);
}
