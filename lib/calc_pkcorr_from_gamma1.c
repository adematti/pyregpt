#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "define.h"
#include "common.h"


void init_gamma1()
{
	init_gamma1_1loop();
	init_gamma1_2loop();
}

void free_gamma1()
{
	free_gamma1_1loop();
	free_gamma1_2loop();
}

void calc_pkcorr_from_gamma1_2loop(FLAG a,FLAG b,histo_t k,histo_t *G1a_1loop,histo_t *G1a_2loop,histo_t *G1b_1loop,histo_t *G1b_2loop)
{
	if (a==b) {
		*G1a_1loop = gamma1_1loop(a,k);
		*G1a_2loop = gamma1_2loop(a,k);
		*G1a_2loop = (*G1a_2loop) + (*G1a_1loop)*(*G1a_1loop)/2.;
		*G1b_1loop = *G1a_1loop;
		*G1b_2loop = *G1a_2loop;
	}
	else {
		*G1a_1loop = gamma1_1loop(a,k);
		*G1a_2loop = gamma1_2loop(a,k);
		*G1b_1loop = gamma1_1loop(b,k);
		*G1b_2loop = gamma1_2loop(b,k);
		*G1a_2loop = (*G1a_2loop) + (*G1a_1loop)*(*G1a_1loop)/2.;
        *G1b_2loop = (*G1b_2loop) + (*G1b_1loop)*(*G1b_1loop)/2.;
	}

}
