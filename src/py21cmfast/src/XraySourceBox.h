#ifndef _XRAYSOURCEBOX_H
#define _XRAYSOURCEBOX_H

#include "OutputStructs.h"

int UpdateXraySourceBox(HaloBox *halobox, double R_inner, double R_outer, int R_ct, double R_star,
                        XraySourceBox *source_box);

#endif
