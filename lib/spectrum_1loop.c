#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "define.h"
#include "common.h"

#define STEP_VERBOSE 10

static INTERPOL interpol_pk_lin = POLY;
static histo_t uvcutoff = 0.5;
static Terms1Loop terms_1loop;
static Pk pk_1loop;

void set_precision_spectrum_1loop_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

void set_running_uvcutoff_spectrum_1loop(histo_t uvcutoff_)
{
	uvcutoff = uvcutoff_;
}

void write_terms_spectrum_1loop(char *fn)
{
	// Writes 1-loop terms into file fn, only used for debugging
	FILE *fr;
	size_t ik;
	fr=fopen(fn,"w");
	if(fr==NULL) error_open_file(fn);
	for(ik=0;ik<terms_1loop.nk;ik++) {
		fprintf(fr,"%f %f %f %f\n",terms_1loop.k[ik],terms_1loop.G1a_1loop[ik],terms_1loop.G1b_1loop[ik],terms_1loop.pkcorr_G2_tree_tree[ik]);
	}
	fclose(fr);
}

void set_terms_spectrum_1loop(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* G1a_1loop,histo_t* G1b_1loop,histo_t* pkcorr_G2_tree_tree)
{
	terms_1loop.nk = nk;
	terms_1loop.k = k;
	terms_1loop.pk_lin = pk_lin;
	terms_1loop.sigma_v2 = sigma_v2;
	terms_1loop.G1a_1loop = G1a_1loop;
	terms_1loop.G1b_1loop = G1b_1loop;
	terms_1loop.pkcorr_G2_tree_tree = pkcorr_G2_tree_tree;

#ifdef _VERBOSE
	print_k(terms_1loop.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void free_terms_spectrum_1loop()
{
	free(terms_1loop.pk_lin);
	free(terms_1loop.sigma_v2);
	free(terms_1loop.G1a_1loop);
	free(terms_1loop.G1b_1loop);
	free(terms_1loop.pkcorr_G2_tree_tree);
}


void run_terms_spectrum_1loop(char* a_,char* b_,size_t num_threads)
{

	FLAG a = set_flag(a_);
	FLAG b = set_flag(b_);
#ifdef _VERBOSE
	printf("*** Calculation at one-loop order\n");
	print_flags(a,b);
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_1loop.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_1loop.k,terms_1loop.pk_lin,terms_1loop.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_1loop.k,terms_1loop.sigma_v2,terms_1loop.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_1loop,a,b,step_verbose) private(ik)
	{
		init_gamma1_1loop();
		init_gamma2_tree();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_1loop.nk;ik++) {

#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			histo_t k = terms_1loop.k[ik];
			terms_1loop.G1a_1loop[ik] = gamma1_1loop(a,k);
			terms_1loop.G1b_1loop[ik] = gamma1_1loop(b,k);
			calc_pkcorr_from_gamma2_tree(a,b,k,&(terms_1loop.pkcorr_G2_tree_tree[ik]));
		}
#pragma omp critical
		{
			free_gamma1_1loop();
			free_gamma2_tree();
		}
	}
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
#ifdef _DEBUG
	write_1loop("debug_1loop.dat");
#endif //_DEBUG
}

void spectrum_1loop(histo_t *pk_1loop)
{
	size_t ik;
	for (ik=0;ik<terms_1loop.nk;ik++) {
		histo_t factor = 0.5 * terms_1loop.k[ik]*terms_1loop.k[ik] * terms_1loop.sigma_v2[ik];
		histo_t exp_factor = my_exp(-factor);
		histo_t G1a_reg = exp_factor * (1. + factor + terms_1loop.G1a_1loop[ik]);
		histo_t G1b_reg = exp_factor * (1. + factor + terms_1loop.G1b_1loop[ik]);
		histo_t pkcorr_G1 = G1a_reg * G1b_reg * terms_1loop.pk_lin[ik];
		//Taruya 2012 (arXiv 1208.1191v1) second term of eq 23
		histo_t pkcorr_G2 = exp_factor*exp_factor * terms_1loop.pkcorr_G2_tree_tree[ik];
		pk_1loop[ik] = pkcorr_G1 + pkcorr_G2;
	}
}

void set_spectrum_1loop(size_t nk,histo_t* k,histo_t* pk)
{
	pk_1loop.nk = nk;
	pk_1loop.k = k;
	pk_1loop.pk = pk;	
}

void run_spectrum_1loop(char* a_,char* b_,size_t num_threads)
{
	size_t nk = pk_1loop.nk;
	histo_t *k = pk_1loop.k;
	size_t size = sizeof(histo_t);
	set_terms_spectrum_1loop(nk,k,(histo_t *) calloc(nk,size),(histo_t *) calloc(nk,size),(histo_t *) calloc(nk,size),(histo_t *) calloc(nk,size),(histo_t *) calloc(nk,size));
	run_terms_spectrum_1loop(a_,b_,num_threads);
	spectrum_1loop(pk_1loop.pk);
	free_terms_spectrum_1loop();
}
