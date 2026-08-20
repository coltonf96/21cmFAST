#ifndef _RADIATIONFIELDS_H
#define _RADIATIONFIELDS_H

#include "OutputStructs.h"

#define UPDATE_RADIATION_FIELDS_EVAL 1
#define UPDATE_RADIATION_FIELDS_CLEANUP 2

int UpdateRadiationFields(float redshift, HaloBox *halobox, double R_inner, double R_outer,
                          int R_ct, double R_star, short mode, short cleanup,
                          float perturbed_field_redshift, PerturbedField *perturbed_field,
                          TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup,
                          RadiationFields *radiation_fields);

#endif
