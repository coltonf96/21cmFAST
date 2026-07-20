#ifndef _XRAYSOURCEBOX_H
#define _XRAYSOURCEBOX_H

#include "OutputStructs.h"

int UpdateXraySourceBox(float redshift, HaloBox *halobox, double R_inner, double R_outer, int R_ct,
                        double R_star, short cleanup, float perturbed_field_redshift,
                        PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                        InitialConditions *ini_boxes, XraySourceBox *source_box);

#endif
