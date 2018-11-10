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
static Terms2Loop terms_2loop;

static void set_precision_spectrum_2loop_pk_lin(char* interpol_)
{
	set_interpol(&interpol_pk_lin,interpol_);
}

static void set_running_uvcutoff_spectrum_2loop(histo_t uvcutoff_)
{
	uvcutoff = uvcutoff_;
}

void write_terms_spectrum_2loop(char *fn)
{
	// Writes 2-loop terms into file fn, only used for debugging
	FILE *fr;
	size_t ik;
	fr=fopen(fn,"w");
	if(fr==NULL) error_open_file(fn);
	for(ik=0;ik<terms_2loop.nk;ik++) {
		fprintf(fr,"%f %f %f %f %f %f %f %f %f\n",terms_2loop.k[ik],terms_2loop.G1a_1loop[ik],terms_2loop.G1a_2loop[ik],terms_2loop.G1b_1loop[ik],terms_2loop.G1b_2loop[ik],terms_2loop.pkcorr_G2_tree_tree[ik],terms_2loop.pkcorr_G2_tree_1loop[ik],terms_2loop.pkcorr_G2_1loop_1loop[ik],terms_2loop.pkcorr_G3_tree[ik]);
	}
	fclose(fr);
}

void set_terms_spectrum_2loop(size_t nk,histo_t* k,histo_t* pk_lin,histo_t* sigma_v2,histo_t* G1a_1loop,histo_t* G1a_2loop,histo_t* G1b_1loop,histo_t* G1b_2loop,histo_t* pkcorr_G2_tree_tree,histo_t *pkcorr_G2_tree_1loop,histo_t *pkcorr_G2_1loop_1loop,histo_t* pkcorr_G3_tree)
{
	terms_2loop.nk = nk;
	terms_2loop.k = k;
	terms_2loop.pk_lin = pk_lin;
	terms_2loop.sigma_v2 = sigma_v2;
	terms_2loop.G1a_1loop = G1a_1loop;
	terms_2loop.G1a_2loop = G1a_2loop;
	terms_2loop.G1b_1loop = G1b_1loop;
	terms_2loop.G1b_2loop = G1b_2loop;
	terms_2loop.pkcorr_G2_tree_tree = pkcorr_G2_tree_tree;
	terms_2loop.pkcorr_G2_tree_1loop = pkcorr_G2_tree_1loop;
	terms_2loop.pkcorr_G2_1loop_1loop = pkcorr_G2_1loop_1loop;
	terms_2loop.pkcorr_G3_tree = pkcorr_G3_tree;

#ifdef _VERBOSE
	print_k(terms_2loop.k,nk);
	printf("\n");
#endif //_VERBOSE
}

void run_terms_spectrum_2loop(char* a_,char* b_,size_t num_threads)
{

	FLAG a = set_flag(a_);
	FLAG b = set_flag(b_);
#ifdef _VERBOSE
	printf("*** Calculation at two-loop order\n");
	print_flags(a,b);
#endif //_VERBOSE
	set_num_threads(num_threads);
	size_t ik=0;
	size_t step_verbose = MAX(terms_2loop.nk*STEP_VERBOSE/100,1);

	timer(0);
	find_pk_lin(terms_2loop.k,terms_2loop.pk_lin,terms_2loop.nk,interpol_pk_lin);
	calc_running_sigma_v2(terms_2loop.k,terms_2loop.sigma_v2,terms_2loop.nk,uvcutoff);
#pragma omp parallel default(none) shared(terms_2loop,a,b,step_verbose) private(ik)
	{
		init_gamma1_1loop();
		init_gamma1_2loop();
		init_gamma2_tree();
		init_gamma2d_1loop();
		init_gamma2t_1loop();
#pragma omp for nowait schedule(dynamic)
		for (ik=0;ik<terms_2loop.nk;ik++) {

#ifdef _VERBOSE
			if (ik % step_verbose == 0) printf(" - Computation done at %zu percent.\n",ik*STEP_VERBOSE/step_verbose);
#endif //_VERBOSE
			histo_t k = terms_2loop.k[ik];
			calc_pkcorr_from_gamma1_2loop(a,b,k,&(terms_2loop.G1a_1loop[ik]),&(terms_2loop.G1a_2loop[ik]),&(terms_2loop.G1b_1loop[ik]),&(terms_2loop.G1b_2loop[ik]));
			calc_pkcorr_from_gamma2_1loop(a,b,k,&(terms_2loop.pkcorr_G2_tree_tree[ik]),&(terms_2loop.pkcorr_G2_tree_1loop[ik]),&(terms_2loop.pkcorr_G2_1loop_1loop[ik]));
		}
#pragma omp critical
		{
			free_gamma1_1loop();
			free_gamma1_2loop();
			free_gamma2_tree();
			free_gamma2d_1loop();
			free_gamma2t_1loop();
		}
	}
	init_gamma3_tree();
	for (ik=0;ik<terms_2loop.nk;ik++) calc_pkcorr_from_gamma3_tree(a,b,terms_2loop.k[ik],&(terms_2loop.pkcorr_G3_tree[ik]));
	free_gamma3_tree();
	
#ifdef _VERBOSE
	timer(1);
#endif //_VERBOSE
#ifdef _DEBUG
	write_2loop("debug_2loop.dat");
#endif //_DEBUG
}

void spectrum_2loop(histo_t *pk_2loop)
{
	size_t ik;
	for (ik=0;ik<terms_2loop.nk;ik++) {
		histo_t factor = 0.5 * terms_2loop.k[ik]*terms_2loop.k[ik] * terms_2loop.sigma_v2[ik];
		histo_t exp_factor = my_exp(-factor);
		histo_t G1a_reg = exp_factor * (1. + factor + 0.5*factor*factor + terms_2loop.G1a_1loop[ik]*(1. + factor) + terms_2loop.G1a_2loop[ik]);
		histo_t G1b_reg = exp_factor * (1. + factor + 0.5*factor*factor + terms_2loop.G1b_1loop[ik]*(1. + factor) + terms_2loop.G1b_2loop[ik]);
		histo_t pkcorr_G1 = G1a_reg * G1b_reg * terms_2loop.pk_lin[ik];
		//Taruya 2012 (arXiv 1208.1191v1) second term of eq 23
		histo_t pkcorr_G2 = exp_factor*exp_factor * (terms_2loop.pkcorr_G2_tree_tree[ik] * (1. + factor)*(1. + factor) + terms_2loop.pkcorr_G2_tree_1loop[ik] * (1. + factor) + terms_2loop.pkcorr_G2_1loop_1loop[ik]);
		//Taruya 2012 (arXiv 1208.1191v1) third term of eq 23
		histo_t pkcorr_G3 = exp_factor*exp_factor * terms_2loop.pkcorr_G3_tree[ik];
		pk_2loop[ik] = pkcorr_G1 + pkcorr_G2 + pkcorr_G3;
	}
}