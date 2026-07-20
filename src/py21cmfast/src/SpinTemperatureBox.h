#ifndef _SPINTEMP_H
#define _SPINTEMP_H

#include <complex.h>
#include <fftw3.h>

#include "InputParameters.h"
#include "OutputStructs.h"
#include "scaling_relations.h"

int ComputeTsBox(float redshift, float prev_redshift, float perturbed_field_redshift, short cleanup,
                 PerturbedField *perturbed_field, XraySourceBox *source_box,
                 TsBox *previous_spin_temp, InitialConditions *ini_boxes, TsBox *this_spin_temp);

typedef struct RadiationFieldsSetup {
    double *ave_log10_MturnLW;
    double inverse_growth_factor_z;
    double x_e_ave_p;
    double Q_HI_zp;
    int NO_LIGHT;
    // TODO: remove these fields below when https://github.com/21cmfast/21cmFAST/issues/668 is fixed
    // (they are needed only for the Eulerian source model)
    ScalingConstants sc;
    fftwf_complex *delta_unfiltered;
    fftwf_complex *log10_Mcrit_LW_unfiltered;
    double *ave_dens;
    double *min_log10_MturnLW;
    double *max_log10_MturnLW;
    double *mean_sfr_zpp;
    double *mean_sfr_zpp_mini;
} RadiationFieldsSetup;

void setup_radiation_fields(float redshift, float perturbed_field_redshift,
                            XraySourceBox *source_box, RadiationFieldsSetup *rad_setup,
                            PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                            InitialConditions *ini_boxes);

void accumulate_radiation_shell(float redshift, RadiationFieldsSetup *rad_setup,
                                XraySourceBox *source_box, int R_ct);

void free_rad_setup(RadiationFieldsSetup *rad_setup, short cleanup);

#endif
