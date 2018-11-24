#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "define.h"
#include "common.h"

//  Timing variables
#ifdef _HAVE_OMP
#include <omp.h>
static double relbeg,relend,absbeg,absend;
#else //_HAVE_OMP
#include <time.h>
static time_t relbeg,relend,absbeg,absend;
#endif //_HAVE_OMP

histo_t my_abs(histo_t x)
{
#ifdef _FLOAT32
	return fabsf(x);
#else
	return fabs(x);
#endif //_FLOAT32
}

histo_t my_log(histo_t x)
{
#ifdef _FLOAT32
	return logf(x);
#else
	return log(x);
#endif //_FLOAT32
}

histo_t my_exp(histo_t x)
{
#ifdef _FLOAT32
	return expf(x);
#else
	return exp(x);
#endif //_FLOAT32
}

histo_t my_pow(histo_t x,histo_t exp)
{
#ifdef _FLOAT32
	return powf(x,exp);
#else
	return pow(x,exp);
#endif //_FLOAT32
}

histo_t my_sqrt(histo_t x)
{
#ifdef _FLOAT32
	return sqrtf(x);
#else
	return sqrt(x);
#endif //_FLOAT32
}

histo_t my_sin(histo_t x)
{
#ifdef _FLOAT32
	return sinf(x);
#else
	return sin(x);
#endif //_FLOAT32
}

histo_t my_cos(histo_t x)
{
#ifdef _FLOAT32
	return cosf(x);
#else
	return cos(x);
#endif //_FLOAT32
}

histo_t power(histo_t x,const size_t n)
{
	if (n==1) return x;
	if (n%2==0) return power(x*x,n/2);
	return x*power(x*x,(n-1)/2);
} 

void powers(histo_t x,histo_t *tab, const size_t npowers)
{
	tab[0] = 1.;
	tab[1] = x;
	size_t i;
	for (i=2;i<npowers;i++) tab[i] = tab[i-1]*x;
}

size_t get_dichotomy_index(histo_t x,histo_t *tab,size_t min,size_t max)
{
	if (min==max-1) return min;
	else {
		size_t ind=(min+max)/2;
		if (x<tab[ind]) return get_dichotomy_index(x,tab,min,ind);
		else if (x>tab[ind]) return get_dichotomy_index(x,tab,ind,max);
		return ind;
	}
}

histo_t interpol_lin(histo_t x,histo_t xmin,histo_t xmax,histo_t ymin,histo_t ymax) 
{
	return (x-xmin)/(xmax-xmin)*(ymax-ymin) + ymin;
}

histo_t interpol_poly(histo_t x,histo_t *tabx,histo_t *taby,size_t nx) 
{
	size_t ix,jx,ixclose = 0;
	histo_t* c = (histo_t*) malloc(nx*sizeof(histo_t));
	histo_t* d = (histo_t*) malloc(nx*sizeof(histo_t));
	histo_t diff = my_abs(x-tabx[ixclose]);
	for (ix=0;ix<nx;ix++) {
		histo_t tmpdiff = my_abs(x-tabx[ix]);
        if (tmpdiff<diff) {
        	ixclose = ix;
        	diff = tmpdiff;
        }
        c[ix] = taby[ix];
        d[ix] = taby[ix];
	}
	//printf("%d %.3f\n",(int) ixclose,taby[ixclose]);
	histo_t y = taby[ixclose];
	ixclose -= 1;
	for (jx=1;jx<nx;jx++) {
		for (ix=0;ix<nx-jx;ix++) {
			histo_t den = (c[ix+1]-d[ix])/(tabx[ix]-tabx[ix+jx]);
			c[ix] = (tabx[ix]-x)*den;
			d[ix] = (tabx[ix+jx]-x)*den;
			//printf("%.3f %.3f %.3f %.3f\n",c[ix+1]-d[ix],tabx[ix]-tabx[ix+jx],c[ix],d[ix]);
		}
		histo_t dy;
		if (2*(ixclose+1)<nx-jx) dy = c[ixclose+1];
		else {
			dy = d[ixclose];
			ixclose -= 1;			
		}
		y += dy;
		//printf("%d %.3f\n",(int) ixclose,y);
	}
	free(c);
	free(d);
	return y;
}

histo_t extrapol_pk_lowk(Pk pkin,histo_t k)
{
	size_t ik,start,end;
	end = MIN(15,((long) pkin.nk-1));
	start = 1;
	histo_t dlnp_dlnk = 0.;
	for (ik=start;ik<end;ik++) {
		if ((pkin.pk[ik+1]<=0.)||(pkin.pk[ik-1]<=0.)) return 0.;
		dlnp_dlnk += my_log(pkin.pk[ik+1]/pkin.pk[ik-1])/my_log(pkin.k[ik+1]/pkin.k[ik-1]);
	}
	histo_t n_eff = dlnp_dlnk/((histo_t) (end-start));
	return pkin.pk[start]*my_pow((k/pkin.k[start]),n_eff);
}

histo_t extrapol_pk_highk(Pk pkin,histo_t k)
{
	size_t ik,start,end;
	end = pkin.nk-1;
	start = MAX(1,((long) end)-15);
	histo_t dlnp_dlnk = 0.;
	for (ik=start;ik<end;ik++) {
		if ((pkin.pk[ik+1]<=0.)||(pkin.pk[ik-1]<=0.)) return 0.;
		dlnp_dlnk += my_log(pkin.pk[ik+1]/pkin.pk[ik-1])/my_log(pkin.k[ik+1]/pkin.k[ik-1]);
	}
	histo_t n_eff = dlnp_dlnk/((histo_t) (end-start));
	return pkin.pk[end]*my_pow((k/pkin.k[end]),n_eff);
}

void find_pk(Pk pkin,histo_t* k,histo_t* pk,size_t nk,INTERPOL interpol)
{
	size_t ik;
	size_t start=0;
	size_t end=pkin.nk-1;
	histo_t kstart = pkin.k[start];
	histo_t kend = pkin.k[end];
	for (ik=0;ik<nk;ik++) {
		if (k[ik]>kend) pk[ik] = extrapol_pk_highk(pkin,k[ik]);
		else if (k[ik]<kstart) pk[ik] = extrapol_pk_lowk(pkin,k[ik]);
		else {
			size_t ind = get_dichotomy_index(k[ik],pkin.k,start,end+1);
			start = ind;
			if (interpol==POLY) {
				size_t ikmin = MAX(0,((long) ind)-2);
				size_t ikmax = MIN(end,ind+2);
				pk[ik] = interpol_poly(k[ik],&(pkin.k[ikmin]),&(pkin.pk[ikmin]),ikmax-ikmin+1);
			}
			else pk[ik] = interpol_lin(k[ik],pkin.k[ind],pkin.k[ind+1],pkin.pk[ind],pkin.pk[ind+1]);
		}
	}
}

void find_pk_lin(histo_t* k,histo_t* pk,size_t nk,INTERPOL interpol)
{
	find_pk(pk_lin,k,pk,nk,interpol);
}

void calc_running_sigma_d2(histo_t *k,histo_t *sigmad2,size_t nk,histo_t uvcutoff)
{
	//sigmav^2 = int_0^{k/2} dq P0(q)/(6*pi^2)
	size_t ik,iklin = 0;
	histo_t sigmad2_ = 0.;
	for (ik=0;ik<nk;ik++) {
		histo_t kmax = k[ik]*uvcutoff;
		if (kmax < pk_lin.k[0]) {
			sigmad2[ik] = 0.;
			continue;
		}
		for (iklin=iklin;iklin<pk_lin.nk-1;iklin++) {
			if (kmax < pk_lin.k[iklin+1]) {
				histo_t pk_kmax = interpol_lin(kmax,pk_lin.k[iklin],pk_lin.k[iklin+1],pk_lin.pk[iklin],pk_lin.pk[iklin+1]);
				sigmad2[ik] = (sigmad2_ + (pk_kmax + pk_lin.pk[iklin]) * (kmax - pk_lin.k[iklin]))/(12.*M_PI*M_PI);
				break;
			}
			else sigmad2_ += (pk_lin.pk[iklin+1] + pk_lin.pk[iklin]) * (pk_lin.k[iklin+1] - pk_lin.k[iklin]);
		}
		if (iklin==pk_lin.nk-1) sigmad2[ik] = sigmad2_/(12.*M_PI*M_PI);
	}
}


_Bool set_precision(Precision *precision,size_t n,histo_t min,histo_t max,char *interpol,const Precision *precision_default)
{
	_Bool change = 0;
	if (n>0) {
		precision->n = n;
		change = 1;
	}
	if (max>=min) {
		precision->min = min;
		precision->max = max;
		change = 1;
	}
	if (set_interpol(&(precision->interpol),interpol)) {
		change = 1;
	}
	if (!change) *precision = *precision_default;
	return change;
}


_Bool set_interpol(INTERPOL *current,char *new)
{
	if (!strcmp(new,"poly")) {
		*current = POLY;
		return 1;
	}
	else if (!strcmp(new,"lin")) {
		*current=LIN;
		return 1;
	}
	return 0;
}

void nodes_weights_gauss_legendre(histo_t xmin,histo_t xmax,histo_t *x,histo_t *w,size_t n) {

	/*
	xmin: minimum x
	xmax: maximum x
	x: array of nodes
	w: array of weights
	n: size of x,w
	*/

	size_t ii,jj,mid = (n+1)/2;
	histo_t xmid = (xmax+xmin)/2.;
	histo_t xl = (xmax-xmin)/2.;
	histo_t p1=0.,p2=0.,p3=0.,pp=0.,z=0.,z1=0.;
	
	for (ii=0;ii<mid;ii++) {
	
		z = cos(M_PI*(ii+.75)/(n+.5));
		z1 = z + EPS + 1.;
		while (my_abs(z-z1)>EPS) {
			p1 = 1.;
			p2 = 0.;
			for (jj=0;jj<n;jj++) {
				p3 = p2;
				p2 = p1;
				p1 = ((2.*jj+1.)*z*p2-jj*p3)/(jj+1.);
			}
			pp=n*(z*p1-p2)/(z*z-1.);
			z1=z;
			z=z1-p1/pp;
		}
		x[ii] = xmid-xl*z;
		x[n-ii-1] = xmid+xl*z;
		w[ii] = 2.*xl/((1.-z*z)*pp*pp);
 		w[n-ii-1] = w[ii];

	}
}

void init_gauss_legendre_q(gammaaussLegendreQ *gauss_legendre,Precision *precision)
{
	histo_t qmin=precision->min,qmax=precision->max;
	if ((qmin<=0.)||(qmax<=0.)) {
		qmin = pk_lin.k[0];
		qmax = pk_lin.k[pk_lin.nk-1];
	}
	gauss_legendre->k = 1.;
	size_t size = sizeof(histo_t);
	size_t nq = precision->n;
	gauss_legendre->nq = nq;
	gauss_legendre->x = (histo_t *) calloc(nq,size);
	gauss_legendre->q = (histo_t *) calloc(nq,size);
	gauss_legendre->w = (histo_t *) calloc(nq,size);
	gauss_legendre->pk = (histo_t *) calloc(nq,size);
	
	nodes_weights_gauss_legendre(my_log(qmin),my_log(qmax),gauss_legendre->x,gauss_legendre->w,gauss_legendre->nq); 
	
	size_t iq;
	for (iq=0;iq<nq;iq++) {
		gauss_legendre->q[iq] = my_exp(gauss_legendre->x[iq]); //independent of k
		gauss_legendre->x[iq] = gauss_legendre->q[iq];
	}
	find_pk_lin(gauss_legendre->q,gauss_legendre->pk,nq,precision->interpol);
}

void update_gauss_legendre_q(gammaaussLegendreQ *gauss_legendre,histo_t k)
{
	gauss_legendre->k = k;
	size_t iq;
	for (iq=0;iq<gauss_legendre->nq;iq++) gauss_legendre->x[iq] = gauss_legendre->q[iq]/k;
}


void free_gauss_legendre_q(gammaaussLegendreQ *gauss_legendre)
{
	free(gauss_legendre->x);
	free(gauss_legendre->q);
	free(gauss_legendre->w);
	free(gauss_legendre->pk);
}


void init_gauss_legendre_mu(gammaaussLegendreMu *gauss_legendre,Precision *precision)
{
	histo_t mumin=-1.,mumax=1.;
	size_t size = sizeof(histo_t);
	size_t nmu = precision->n;
	gauss_legendre->nmu = nmu;
	gauss_legendre->muref = (histo_t *) calloc(nmu,size);
	gauss_legendre->wref = (histo_t *) calloc(nmu,size);
	gauss_legendre->mu = (histo_t *) calloc(nmu,size);
	gauss_legendre->w = (histo_t *) calloc(nmu,size);
	
	nodes_weights_gauss_legendre(mumin,mumax,gauss_legendre->muref,gauss_legendre->wref,gauss_legendre->nmu); 
	
	size_t imu;
	for (imu=0;imu<nmu;imu++) {
		gauss_legendre->mu[imu] = gauss_legendre->muref[imu];
		gauss_legendre->w[imu] = gauss_legendre->wref[imu];
	}
}

void update_gauss_legendre_mu(gammaaussLegendreMu *gauss_legendre,histo_t mumin,histo_t mumax)
{
	//gauss_legendre->mumin = mumin;
	//gauss_legendre->mumin = mumin;
	size_t imu;
	histo_t jacobian = (mumax-mumin)/2.;
	for (imu=0;imu<gauss_legendre->nmu;imu++) {
		gauss_legendre->mu[imu] = (gauss_legendre->muref[imu]+1.)*jacobian+mumin;
		gauss_legendre->w[imu] = gauss_legendre->wref[imu]*jacobian;
	}
}

void free_gauss_legendre_mu(gammaaussLegendreMu *gauss_legendre)
{
	free(gauss_legendre->muref);
	free(gauss_legendre->wref);
	free(gauss_legendre->mu);
	free(gauss_legendre->w);
}

void timer(size_t i)
{
	/////
	// Timing routine
	// timer(0) -> initialize relative clock
	// timer(1) -> read relative clock
	// timer(2) -> read relative clock and initialize it afterwards
	// timer(4) -> initialize absolute clock
	// timer(5) -> read absolute clock
#ifdef _HAVE_OMP
	if(i==0)
		relbeg=omp_get_wtime();
	else if(i==1) {
		relend=omp_get_wtime();
		printf(" - relative time ellapsed %.1f ms\n",1000*(relend-relbeg));
	}    
	else if(i==2) {
		relend=omp_get_wtime();
		printf(" - relative time ellapsed %.1f ms\n",1000*(relend-relbeg));
		relbeg=omp_get_wtime();
	}
	else if(i==4)
		absbeg=omp_get_wtime();
	else if(i==5) {
		absend=omp_get_wtime();
		printf(" - total time ellapsed %.1f ms \n",1000*(absend-absbeg));
	}
#else //_HAVE_OMP
	int diff;
	
	if(i==0)
		relbeg=time(NULL);
	else if(i==1) {
		relend=time(NULL);
		diff=(int)(difftime(relend,relbeg));
		printf(" - relative time ellapsed %02d:%02d:%02d \n",
		 diff/3600,(diff/60)%60,diff%60);
	}    
	else if(i==2) {
		relend=time(NULL);
		diff=(size_t)(difftime(relend,relbeg));
		printf(" - relative time ellapsed %02d:%02d:%02d \n",
		 diff/3600,(diff/60)%60,diff%60);
		relbeg=time(NULL);
	}
	else if(i==4)
		absbeg=time(NULL);
	else if(i==5) {
		absend=time(NULL);
		diff=(size_t)(difftime(absend,absbeg));
		printf(" - total time ellapsed %02d:%02d:%02d \n",
		 diff/3600,(diff/60)%60,diff%60);
	}
#endif //_HAVE_OMP
}

void error_open_file(char *fname)
{
	//////
	// Open error handler
	fprintf(stderr,"REgammaPT: Could not open file %s \n",fname);
	exit(1);
}

void write_gauss_legendre_q(gammaaussLegendreQ gauss_legendre,char *fn)
{
	//////
	// Writes gauss legendre terms into file fn, only used for debugging
	FILE *fr;
	size_t iq,nq=gauss_legendre.nq;
	fr=fopen(fn,"w");
	if (fr==NULL) error_open_file(fn);
	for (iq=0;iq<nq;iq++) fprintf(fr,"%f %f %f\n",gauss_legendre.q[iq],gauss_legendre.w[iq],gauss_legendre.pk[iq]);
	fclose(fr);
}

void write_gauss_legendre_mu(gammaaussLegendreMu gauss_legendre,char *fn)
{
	//////
	// Writes gauss legendre terms into file fn, only used for debugging
	FILE *fr;
	size_t imu,nmu=gauss_legendre.nmu;
	fr=fopen(fn,"w");
	if (fr==NULL) error_open_file(fn);
	for (imu=0;imu<nmu;imu++) fprintf(fr,"%f %f\n",gauss_legendre.mu[imu],gauss_legendre.w[imu]);
	fclose(fr);
}

/*
void interpol_pk_lin(histo_t* k,histo_t* pk,size_t nk)
{
	size_t ik;
	size_t start=0;
	size_t end=pk_lin.nk;
	histo_t kstart = pk_lin.k[0];
	histo_t kend = pk_lin.k[end-1];
	for (ik=0;ik<nk;ik++) {
		if ((k[ik]>=kstart)&&(k[ik]<=kend)) {
			size_t ind = get_dichotomy_index(k[ik],pk_lin.k,start,end);
			start = ind;
			pk[ik] = interpol_lin(k[ik],pk_lin.k[ind],pk_lin.k[ind+1],pk_lin.pk[ind],pk_lin.pk[ind+1]);
		}
		else pk[ik] = 0.;
	}
}
*/
