#include "XraySourceBox.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "cexcept.h"
#include "debugging.h"
#include "dft.h"
#include "exceptions.h"
#include "filtering.h"
#include "indexing.h"
#include "logger.h"

// NOTE: I've moved this to a function to help in simplicity, it is not clear whether it is faster
//   to do all of one radii at once (more clustered FFT and larger thread blocks) or all of one box
//   (better memory locality)
// TODO: filter speed tests
void one_annular_filter(float *input_box, float *output_box, double R_inner, double R_outer,
                        double R_star, int filter_type, double *u_avg, double *f_avg) {
    int i, j, k;
    index_huge ct;
    double unfiltered_avg = 0;
    double filtered_avg = 0;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};

    fftwf_complex *unfiltered_box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    fftwf_complex *filtered_box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS) \
    reduction(+ : unfiltered_avg)
    {
        float curr_val;
        index_huge index_r, index_f;
#pragma omp for
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    index_r = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    curr_val = input_box[index_r];
                    *((float *)unfiltered_box + index_f) = curr_val;
                    unfiltered_avg += curr_val;
                }
            }
        }
    }
    // No need to filter the box if we only have one cell!
    if (simulation_options_global->HII_DIM > 1) {
        // Transform unfiltered box to k-space to prepare for filtering
        // this would normally only be done once but we're using a different redshift for each R now
        dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, unfiltered_box);

// remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
// Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
                unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
            }
        }

        // Smooth the density field, at the same time store the minimum and maximum densities for
        // their usage in the interpolation tables copy over unfiltered box
        memcpy(filtered_box, unfiltered_box, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        // Don't filter on the cell scale
        if (R_inner > 0) {
            filter_box(filtered_box, box_dim, filter_type, R_inner, R_outer, R_star);
        }

        // now fft back to real space
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, filtered_box);
    }
// copy over the values
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS) \
    reduction(+ : filtered_avg)
    {
        float curr_val;
        index_huge index_f, index_r;
#pragma omp for
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    index_r = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    if (simulation_options_global->HII_DIM > 1) {
                        curr_val = *((float *)filtered_box + index_f);
                    } else {  // Just take the unfiltered box/cell if HII_DIM = 1
                        curr_val = *((float *)unfiltered_box + index_f);
                    }
                    // correct for aliasing in the filtering step
                    if (curr_val < 0.) curr_val = 0.;

                    output_box[index_r] = curr_val;
                    filtered_avg += curr_val;
                }
            }
        }
    }

    unfiltered_avg /= HII_TOT_NUM_PIXELS;
    filtered_avg /= HII_TOT_NUM_PIXELS;

    *u_avg = unfiltered_avg;
    *f_avg = filtered_avg;

    fftwf_free(filtered_box);
    fftwf_free(unfiltered_box);
}

int UpdateXraySourceBox(HaloBox *halobox, double R_inner, double R_outer, int R_ct, double R_star,
                        XraySourceBox *source_box) {
    int status, filter_type;
    Try {
        // the indexing needs these
        filter_type = astro_options_global->LYA_MULTIPLE_SCATTERING ? 5 : 4;

        // only print once, since this is called for every R
        if (R_ct == 0) LOG_DEBUG("starting XraySourceBox");

        double sfr_avg, fsfr_avg, sfr_avg_mini = 0., fsfr_avg_mini = 0.;
        double xray_avg, fxray_avg;
        one_annular_filter(halobox->halo_sfr,
                           &(source_box->filtered_sfr[R_ct * HII_TOT_NUM_PIXELS]), R_inner, R_outer,
                           R_star, filter_type, &sfr_avg, &fsfr_avg);
        one_annular_filter(halobox->halo_xray,
                           &(source_box->filtered_xray[R_ct * HII_TOT_NUM_PIXELS]), R_inner,
                           R_outer, R_star, 4, &xray_avg, &fxray_avg);
        source_box->mean_sfr[R_ct] = fsfr_avg;
        if (astro_options_global->USE_MINI_HALOS) {
            one_annular_filter(halobox->halo_sfr_mini,
                               &(source_box->filtered_sfr_mini[R_ct * HII_TOT_NUM_PIXELS]), R_inner,
                               R_outer, R_star, filter_type, &sfr_avg_mini, &fsfr_avg_mini);
            source_box->mean_sfr_mini[R_ct] = fsfr_avg_mini;
            source_box->mean_log10_Mcrit_LW[R_ct] = halobox->log10_Mcrit_MCG_ave;
            // In case of multiple scattering and mini-halos, we need to filter the SFRD fields
            // again for the the LW feedback, as these photons travel in straight lines
            if (astro_options_global->LYA_MULTIPLE_SCATTERING) {
                one_annular_filter(halobox->halo_sfr,
                                   &(source_box->filtered_sfr_lw[R_ct * HII_TOT_NUM_PIXELS]),
                                   R_inner, R_outer, R_star, 4, &sfr_avg, &fsfr_avg);
                one_annular_filter(halobox->halo_sfr_mini,
                                   &(source_box->filtered_sfr_mini_lw[R_ct * HII_TOT_NUM_PIXELS]),
                                   R_inner, R_outer, R_star, 4, &sfr_avg_mini, &fsfr_avg_mini);
            }
        }

        if (R_ct == astro_params_global->N_STEP_TS - 1) LOG_DEBUG("finished XraySourceBox");

        LOG_SUPER_DEBUG("R = [%8.3f - %8.3f] | mean filtered sfr  = %10.3e unfiltered %10.3e",
                        R_inner, R_outer, fsfr_avg, sfr_avg);
        LOG_ULTRA_DEBUG("mean filtered xray = %10.3e unfiltered %10.3e", fxray_avg, xray_avg);
        if (astro_options_global->USE_MINI_HALOS) {
            LOG_SUPER_DEBUG("MINI: filtered sfr %10.3e unfiltered %10.3e log10_Mcrit_LW = %10.3e",
                            fsfr_avg_mini, sfr_avg_mini, source_box->mean_log10_Mcrit_LW[R_ct]);
        }

        // free fftwf only if we have a full box (with more than one cell)
        if (simulation_options_global->HII_DIM > 1) {
            fftwf_forget_wisdom();
            fftwf_cleanup_threads();
            fftwf_cleanup();
        }
    }  // End of try
    Catch(status) { return (status); }
    return (0);
}
