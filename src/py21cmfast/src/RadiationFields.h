#ifndef _RADIATIONFIELDS_H
#define _RADIATIONFIELDS_H

#include "OutputStructs.h"

int UpdateRadiationFields(float redshift, HaloBox *halobox, double R_inner, double R_outer,
                          int R_ct, double R_star, short cleanup, float perturbed_field_redshift,
                          PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                          RadiationFieldsSetup *rad_setup, RadiationFields *radiation_fields);

#endif
