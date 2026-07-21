#ifndef _XRAYSOURCEBOX_H
#define _XRAYSOURCEBOX_H

#include <complex.h>
#include <fftw3.h>

#include "OutputStructs.h"
#include "scaling_relations.h"

int UpdateXraySourceBox(float redshift, HaloBox *halobox, double R_inner, double R_outer, int R_ct,
                        double R_star, short cleanup, float perturbed_field_redshift,
                        PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                        InitialConditions *ini_boxes, XraySourceBox *source_box);

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
