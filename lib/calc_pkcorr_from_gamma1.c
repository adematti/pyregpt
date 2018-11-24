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

void calc_pkcorr_gamma1_2loop(FLAG a,FLAG b,histo_t k,histo_t *gamma1a_1loop,histo_t *gamma1a_2loop,histo_t *gamma1b_1loop,histo_t *gamma1b_2loop)
{
	if (a==b) {
		*gamma1a_1loop = gamma1_1loop(a,k);
		*gamma1a_2loop = gamma1_2loop(a,k);
		*gamma1a_2loop = (*gamma1a_2loop) + (*gamma1a_1loop)*(*gamma1a_1loop)/2.;
		*gamma1b_1loop = *gamma1a_1loop;
		*gamma1b_2loop = *gamma1a_2loop;
	}
	else {
		*gamma1a_1loop = gamma1_1loop(a,k);
		*gamma1a_2loop = gamma1_2loop(a,k);
		*gamma1b_1loop = gamma1_1loop(b,k);
		*gamma1b_2loop = gamma1_2loop(b,k);
		*gamma1a_2loop = (*gamma1a_2loop) + (*gamma1a_1loop)*(*gamma1a_1loop)/2.;
        *gamma1b_2loop = (*gamma1b_2loop) + (*gamma1b_1loop)*(*gamma1b_1loop)/2.;
	}

}
