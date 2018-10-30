####################################################
###          User definable stuff

DEFINEOPTIONS = -D_VERBOSE
#DEFINEOPTIONS += -D_DEBUG
#DEFINEOPTIONS += -D_FLOAT32
#GSL options
GSL_INC = /opt/rhel-6.x86_64/gnu4.6.2/gsl/1.15/include
GSL_LIB = /opt/rhel-6.x86_64/gnu4.6.2/gsl/1.15/lib
CUBA_INC = $(CUBA)
CUBA_LIB = $(CUBA)
REGPT_SRC = ./lib
REGPT_LIB = ./lib

### End of user-definable stuff
####################################################

LGSL = -L$(GSL_LIB) -lgsl -lgslcblas
#LCUBA = -L$(CUBA_LIB) -Wl,-whole-archive -lcuba
#LCUBA = -L$(CUBA_LIB)/libcuba.a
LCUBA = -L$(CUBA_LIB)

# DEFINES for the OpenMP version
DEFINEFLAGSCPU = $(DEFINEOPTIONS)

# COMPILER AND OPTIONS
COMPCPU = gcc
#OPTCPU = -Wall -O3 -fopenmp $(DEFINEFLAGSCPU)
OPTCPU = -Wextra -O3 -fopenmp $(DEFINEFLAGSCPU)

#INCLUDES AND LIBRARIES
INCLUDECOM = -I./lib -I$(GSL_INC) -I$(CUBA_INC)
#LIBCPU = $(LGSL) $(LCUBA) -lm
LIBCPU = $(LGSL) $(LCUBA) -lm

#.c FILES
CCOM = $(REGPT_LIB)/common.c
CKERN = $(REGPT_LIB)/kernels.c
CGAM1 = $(REGPT_LIB)/calc_pkcorr_from_gamma1.c
CGAM2 = $(REGPT_LIB)/calc_pkcorr_from_gamma2.c
CGAM2d = $(REGPT_LIB)/gamma2d.c
CGAM2v = $(REGPT_LIB)/gamma2v.c
CGAM3 = $(REGPT_LIB)/calc_pkcorr_from_gamma3.c
CBIAS = $(REGPT_LIB)/calc_pkcorr_from_bias.c
CAB = $(REGPT_LIB)/calc_pkcorr_from_A_B.c
CREGPT = $(REGPT_LIB)/regpt.c

#.o FILES
OCOM = $(REGPT_LIB)/common.o
OKERN = $(REGPT_LIB)/kernels.o
OGAM1 = $(REGPT_LIB)/calc_pkcorr_from_gamma1.o
OGAM2 = $(REGPT_LIB)/calc_pkcorr_from_gamma2.o
OGAM2d = $(REGPT_LIB)/gamma2d.o
OGAM2v = $(REGPT_LIB)/gamma2v.o
OGAM3 = $(REGPT_LIB)/calc_pkcorr_from_gamma3.o
OBIAS = $(REGPT_LIB)/calc_pkcorr_from_bias.o
OAB = $(REGPT_LIB)/calc_pkcorr_from_A_B.o
OREGPT = $(REGPT_LIB)/regpt.o
OFILES = $(OCOM) $(OKERN) $(OGAM1) $(OGAM2) $(OGAM2d) $(OGAM2v) $(OGAM3) $(OBIAS) $(OAB) $(OREGPT)

#FINAL GOAL
EXE = REGPT

#RULES
default : $(EXE)
#RULE TO MAKE .o's FROM .c's
$(OCOM) : $(CCOM) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OKERN) : $(CKERN) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OGAM1) : $(CGAM1) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OGAM2) : $(CGAM2) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OGAM2d) : $(CGAM2d) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OGAM2v) : $(CGAM2v) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OGAM3) : $(CGAM3) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OBIAS) : $(CBIAS) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OAB) : $(CAB) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)
$(OREGPT) : $(CREGPT) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -c $< -o $@ $(INCLUDECOM) $(LIBCPU)

#RULES TO MAKE THE FINAL EXECUTABLES
$(EXE) : $(OFILES) Makefile
	$(COMPCPU) $(OPTCPU) -fPIC -shared $(OFILES) -o regpt.so

#CLEANING RULES
clean :
	rm -f $(REGPT_LIB)/*.o
	rm -f $(REGPT_LIB)/*.so
