/*
Module for computing the radiation fields in 21cmFAST.
This includes X-ray heating rate, photoionization rate, and Lyman-alpha flux.
*/
#include "RadiationFields.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "dft.h"
#include "elec_interp.h"
#include "exceptions.h"
#include "filtering.h"
#include "heating_helper_progs.h"
#include "hmf.h"
#include "indexing.h"
#include "interp_tables.h"
#include "logger.h"

// This function should construct all the tables which depend on R
void setup_z_edges(double zp, RadiationFieldsSetup *rad_setup) {
    double R, R_factor;
    double zpp, prev_zpp, prev_R;
    double dzpp_for_evolve;
    int R_ct;

    if (simulation_options_global->HII_DIM == 1) {
        // If HII_DIM=1 (happens when we run_global_evolution), we take a typical cell size
        // of 1.5Mpc, just to for setting the z'' array (note that filtering won't be done on a box
        // with a single cell)
        R = physconst.l_factor * 1.5;
    } else {
        R = physconst.l_factor * simulation_options_global->BOX_LEN /
            (float)simulation_options_global->HII_DIM;
    }
    R_factor = pow(astro_params_global->R_MAX_TS / R, 1 / ((float)astro_params_global->N_STEP_TS));

    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        rad_setup->R_values[R_ct] = R;
        if (R_ct == 0) {
            prev_zpp = zp;
            prev_R = 0;
        } else {
            prev_zpp = rad_setup->zpp_edge[R_ct - 1];
            prev_R = rad_setup->R_values[R_ct - 1];
        }

        // cell size
        rad_setup->zpp_edge[R_ct] =
            prev_zpp - (rad_setup->R_values[R_ct] - prev_R) * physconst.cm_per_Mpc / drdz(prev_zpp);
        // average redshift value of shell: z'' + 0.5 * dz''
        zpp = (rad_setup->zpp_edge[R_ct] + prev_zpp) * 0.5;

        rad_setup->zpp_for_evolve_list[R_ct] = zpp;
        if (R_ct == 0) {
            dzpp_for_evolve = zp - rad_setup->zpp_edge[0];
        } else {
            dzpp_for_evolve = rad_setup->zpp_edge[R_ct - 1] - rad_setup->zpp_edge[R_ct];
        }
        rad_setup->dzpp_list[R_ct] = dzpp_for_evolve;  // z bin width
        rad_setup->dtdz_list[R_ct] = dtdz(zpp);        // dt/dz''

        R *= R_factor;
    }
    LOG_DEBUG("%d steps R range [%.2e,%.2e] z range [%.2f,%.2f]", R_ct, rad_setup->R_values[0],
              rad_setup->R_values[R_ct - 1], zp, rad_setup->zpp_edge[R_ct - 1]);
}

void calculate_spectral_factors(double zp, RadiationFieldsSetup *rad_setup) {
    double nuprime;
    bool first_radii = true, first_zero = true;
    double trial_zpp;
    int counter, ii;
    int n_pts_radii = 1000;
    double weight = 0.;
    int R_ct, n_ct;
    double zpp, zpp_integrand;

    double sum_lyn_val, sum_lyn_val_MINI;
    double sum_lyLW_val, sum_lyLW_val_MINI;
    double sum_lynto2_val, sum_lynto2_val_MINI;
    double sum_ly2_val, sum_ly2_val_MINI;
    // technically don't need to initialise since it is only used in R_ct > 1 conditional
    double sum_lyn_prev = 0., sum_lyn_prev_MINI = 0.;
    double sum_ly2_prev = 0., sum_ly2_prev_MINI = 0.;
    double sum_lynto2_prev = 0., sum_lynto2_prev_MINI = 0.;
    double prev_zpp = 0;
    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        zpp = rad_setup->zpp_for_evolve_list[R_ct];
        // We need to set up prefactors for how much of Lyman-N radiation is recycled to Lyman-alpha
        sum_lyLW_val = 0.;
        sum_lyLW_val_MINI = 0.;
        sum_lynto2_val = 0.;
        sum_lynto2_val_MINI = 0.;
        sum_ly2_val = 0.;
        sum_ly2_val_MINI = 0.;

        // in case we use LYA_HEATING, we separate the ==2 and >2 cases
        nuprime = nu_n(2) * (1. + zpp) / (1. + zp);
        if (zpp < zmax(zp, 2)) {
            sum_ly2_val = frecycle(2) * spectral_emissivity(nuprime, 0, 2);
            if (astro_options_global->USE_MINI_HALOS) {
                sum_ly2_val_MINI = frecycle(2) * spectral_emissivity(nuprime, 0, 3);

                if (nuprime < physconst.nu_LW_thresh / physconst.nu_ion_HI)
                    nuprime = physconst.nu_LW_thresh / physconst.nu_ion_HI;
                // NOTE: are we comparing nuprime at z' and z'' correctly here?
                //   currently: emitted frequency >= received frequency of next n
                if (nuprime >= nu_n(2 + 1)) continue;

                sum_lyLW_val +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                sum_lyLW_val_MINI +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
            }
        }

        for (n_ct = NSPEC_MAX; n_ct >= 3; n_ct--) {
            if (zpp > zmax(zp, n_ct)) continue;

            nuprime = nu_n(n_ct) * (1 + zpp) / (1.0 + zp);
            sum_lynto2_val += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 2);
            if (astro_options_global->USE_MINI_HALOS) {
                sum_lynto2_val_MINI += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 3);

                if (nuprime < physconst.nu_LW_thresh / physconst.nu_ion_HI)
                    nuprime = physconst.nu_LW_thresh / physconst.nu_ion_HI;
                if (nuprime >= nu_n(n_ct + 1)) continue;
                sum_lyLW_val +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                sum_lyLW_val_MINI +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
            }
        }
        sum_lyn_val = sum_ly2_val + sum_lynto2_val;
        sum_lyn_val_MINI = sum_ly2_val_MINI + sum_lynto2_val_MINI;

        // At the edge of the redshift limit, part of the shell will still contain a contribution
        //   This loop approximates the volume which contains the contribution
        //   and multiplies this by the previous shell's value.
        // This should probably be done separately for each line, since they all face the same issue
        //   It could also be avoided by approximating the integral within each bin better
        //   than taking the value at the bin centre
        if (R_ct > 1 && sum_lyn_val == 0.0 && sum_lyn_prev > 0. && first_radii) {
            for (ii = 0; ii < n_pts_radii; ii++) {
                trial_zpp = prev_zpp + (zpp - prev_zpp) * (float)ii / ((float)n_pts_radii - 1.);
                counter = 0;
                for (n_ct = NSPEC_MAX; n_ct >= 2; n_ct--) {
                    if (trial_zpp > zmax(zp, n_ct)) continue;
                    counter += 1;
                }
                // This is the first sub-radius which has no contribution
                // Use this distance to weigh contribution at previous R
                if (counter == 0 && first_zero) {
                    first_zero = false;
                    weight = (float)ii / (float)n_pts_radii;
                }
            }
            sum_lyn_val = weight * sum_lyn_prev;
            sum_ly2_val = weight * sum_ly2_prev;
            sum_lynto2_val = weight * sum_lynto2_prev;
            if (astro_options_global->USE_MINI_HALOS) {
                sum_lyn_val_MINI = weight * sum_lyn_prev_MINI;
                sum_ly2_val_MINI = weight * sum_ly2_prev_MINI;
                sum_lynto2_val_MINI = weight * sum_lynto2_prev_MINI;
            }
            first_radii = false;
        }
        zpp_integrand = (pow(1 + zp, 2) * (1 + zpp));

        if (astro_options_global->USE_LYA_HEATING) {
            rad_setup->lya_flux_continuum_prefactor[R_ct] = zpp_integrand * sum_ly2_val;
            rad_setup->lya_flux_injected_prefactor[R_ct] = zpp_integrand * sum_lynto2_val;
            LOG_ULTRA_DEBUG("cont %.2e inj %.2e", rad_setup->lya_flux_continuum_prefactor[R_ct],
                            rad_setup->lya_flux_injected_prefactor[R_ct]);
        } else {
            rad_setup->lya_flux_continuum_injected_prefactor[R_ct] = zpp_integrand * sum_lyn_val;
            LOG_ULTRA_DEBUG("z: %.2e R: %.2e int %.2e starlya: %.4e", zpp,
                            rad_setup->R_values[R_ct], zpp_integrand,
                            rad_setup->lya_flux_continuum_injected_prefactor[R_ct]);
        }
        if (astro_options_global->USE_MINI_HALOS) {
            rad_setup->lyw_flux_prefactor[R_ct] = zpp_integrand * sum_lyLW_val;
            rad_setup->lyw_flux_prefactor_MINI[R_ct] = zpp_integrand * sum_lyLW_val_MINI;
            LOG_ULTRA_DEBUG("LW: %.2e LWmini: %.2e", rad_setup->lyw_flux_prefactor[R_ct],
                            rad_setup->lyw_flux_prefactor_MINI[R_ct]);
            if (astro_options_global->USE_LYA_HEATING) {
                rad_setup->lya_flux_continuum_prefactor_MINI[R_ct] =
                    zpp_integrand * sum_ly2_val_MINI;
                rad_setup->lya_flux_injected_prefactor_MINI[R_ct] =
                    zpp_integrand * sum_lynto2_val_MINI;
                LOG_ULTRA_DEBUG("cont mini %.2e inj mini %.2e",
                                rad_setup->lya_flux_continuum_prefactor_MINI[R_ct],
                                rad_setup->lya_flux_injected_prefactor_MINI[R_ct]);
            } else {
                rad_setup->lya_flux_continuum_injected_prefactor_MINI[R_ct] =
                    zpp_integrand * sum_lyn_val_MINI;
                LOG_ULTRA_DEBUG("starmini: %.2e",
                                rad_setup->lya_flux_continuum_injected_prefactor_MINI[R_ct]);
            }
        }

        sum_lyn_prev = sum_lyn_val;
        sum_lyn_prev_MINI = sum_lyn_val_MINI;
        sum_ly2_prev = sum_ly2_val;
        sum_ly2_prev_MINI = sum_ly2_val_MINI;
        sum_lynto2_prev = sum_lynto2_val;
        sum_lynto2_prev_MINI = sum_lynto2_val_MINI;
        prev_zpp = zpp;
    }
}

// construct the [x_e][R_ct] tables
// NOTE: Frequency integrals are based on PREVIOUS x_e_ave
//   The x_e tables are not regular, hence the precomputation of indices/interp points
void fill_freqint_tables(double zp, RadiationFieldsSetup *rad_setup, ScalingConstants *sc) {
    double lower_int_limit;
    int x_e_ct, R_ct;
    int num_R = astro_params_global->N_STEP_TS;

#pragma omp parallel private(R_ct, x_e_ct, lower_int_limit) \
    num_threads(simulation_options_global -> N_THREADS)
    {
#pragma omp for
        // In TauX we integrate Nion from zpp to zp using the LW turnover mass at zp (predending its
        // at zpp)
        //   Calculated from the average smoothed zp grid (from previous LW field) at radius R
        // NOTE: The one difference currently between the halobox and density field options is the
        // weighting of the average
        //   density -> volume weighted cell average || halo -> halo weighted average
        for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
            // TODO: At the moment, inhomogeneous reionization feedback cannot be accounted in
            // SpinTemperatureBox.c,
            //      see https://github.com/21cmfast/21cmFAST/issues/470. Thus, we use the
            //      homogeneous (feedback-free) ACG turnover mass. It is important to remember to
            //      fix this when issue #470 is fixed!
            if (astro_options_global->USE_MINI_HALOS) {
                lower_int_limit = fmax(
                    nu_tau_one_MINI(zp, rad_setup->zpp_for_evolve_list[R_ct], rad_setup->x_e_ave_zp,
                                    rad_setup->Q_HI_zp, log10(sc->mturn_acg_homogeneous),
                                    rad_setup->ave_log10_MturnLW[R_ct], sc),
                    (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz);
            } else {
                lower_int_limit =
                    fmax(nu_tau_one(zp, rad_setup->zpp_for_evolve_list[R_ct], rad_setup->x_e_ave_zp,
                                    rad_setup->Q_HI_zp, log10(sc->mturn_acg_homogeneous), sc),
                         (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz);
            }
            // set up frequency integral table for later interpolation for the cell's x_e value
            for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
                rad_setup->freq_int_heat_tbl[freq_index(x_e_ct, R_ct, num_R)] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                rad_setup->freq_int_ion_tbl[freq_index(x_e_ct, R_ct, num_R)] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                rad_setup->freq_int_lya_tbl[freq_index(x_e_ct, R_ct, num_R)] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);

                // we store these to avoid calculating them in the box_ct loop
                if (x_e_ct > 0) {
                    rad_setup->freq_int_heat_tbl_diff[freq_index(x_e_ct - 1, R_ct, num_R)] =
                        rad_setup->freq_int_heat_tbl[freq_index(x_e_ct, R_ct, num_R)] -
                        rad_setup->freq_int_heat_tbl[freq_index(x_e_ct - 1, R_ct, num_R)];
                    rad_setup->freq_int_ion_tbl_diff[freq_index(x_e_ct - 1, R_ct, num_R)] =
                        rad_setup->freq_int_ion_tbl[freq_index(x_e_ct, R_ct, num_R)] -
                        rad_setup->freq_int_ion_tbl[freq_index(x_e_ct - 1, R_ct, num_R)];
                    rad_setup->freq_int_lya_tbl_diff[freq_index(x_e_ct - 1, R_ct, num_R)] =
                        rad_setup->freq_int_lya_tbl[freq_index(x_e_ct, R_ct, num_R)] -
                        rad_setup->freq_int_lya_tbl[freq_index(x_e_ct - 1, R_ct, num_R)];
                }
            }
            LOG_ULTRA_DEBUG("Nu Integrals || R_ct %d R %.2e zpp %.2f nu_min %.2e", R_ct,
                            rad_setup->R_values[R_ct], rad_setup->zpp_for_evolve_list[R_ct],
                            lower_int_limit);
            LOG_ULTRA_DEBUG("heat[x_e=0] %.2e ion[x_e=0] %.2e lya[x_e=0] %.2e",
                            rad_setup->freq_int_heat_tbl[freq_index(0, R_ct, num_R)],
                            rad_setup->freq_int_ion_tbl[freq_index(0, R_ct, num_R)],
                            rad_setup->freq_int_lya_tbl[freq_index(0, R_ct, num_R)]);
        }
// separating the inverse diff loop to prevent a race on different R_ct (shouldn't matter)
#pragma omp for
        for (x_e_ct = 0; x_e_ct < x_int_NXHII - 1; x_e_ct++) {
            rad_setup->inverse_diff[x_e_ct] = 1. / (x_int_XHII[x_e_ct + 1] - x_int_XHII[x_e_ct]);
        }
    }

    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
            if (isfinite(rad_setup->freq_int_heat_tbl[freq_index(x_e_ct, R_ct, num_R)]) == 0 ||
                isfinite(rad_setup->freq_int_ion_tbl[freq_index(x_e_ct, R_ct, num_R)]) == 0 ||
                isfinite(rad_setup->freq_int_lya_tbl[freq_index(x_e_ct, R_ct, num_R)]) == 0) {
                LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
                Throw(TableGenerationError);
            }
        }
    }
}

// calculate the global properties used for making the frequency integrals,
//   used for filling factor and NO_LIGHT
int global_reion_properties(double zp, RadiationFieldsSetup *rad_setup) {
    double sum_nion = 0, sum_nion_mini = 0;

    // For a lot of global evolution, this code uses Nion_general. We can replace this with the halo
    // field at the same snapshot, but the nu integrals go from zp to zpp to find the tau = 1
    // barrier so it needs the QHII in a range [zp,zpp]. I want to replace this whole thing with a
    // global history struct but I will need to change the Tau function chain.
    double determine_zpp_max, determine_zpp_min;

    // at z', we need a differenc constant struct
    ScalingConstants sc;
    set_scaling_constants(zp, &sc, false);

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        determine_zpp_min = zp * 0.999;
        // NOTE: must be called after setup_z_edges for this line
        determine_zpp_max =
            rad_setup->zpp_for_evolve_list[astro_params_global->N_STEP_TS - 1] * 1.001;

        // We need the tables for the frequency integrals & mean fixing
        // NOTE: These global tables confuse me, we do ~400 (x50 for mini) integrals to build the
        // table, despite only having
        //   ~100 redshifts. The benefit of interpolating here would only matter if we keep the same
        //   table over subsequent snapshots, which we don't seem to do. The Nion table is used in
        //   nu_tau_one a lot but I think there's a better way to do that
        if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
            /* initialise interpolation of the mean collapse fraction for global reionization.*/
            initialise_Nion_Ts_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                      &sc);
        } else {
            init_FcollTable(determine_zpp_min, determine_zpp_max, true);
        }
    }

    // For consistency between halo and non-halo based, the NO_LIGHT and filling_factor_zp
    //   are based on the expected global Nion. as mentioned above it would be nice to
    //   change this to a saved reionisation/sfrd history from previous snapshots
    // TODO: At the moment, inhomogeneous reionization feedback cannot be accounted in
    // SpinTemperatureBox.c,
    //      see https://github.com/21cmfast/21cmFAST/issues/470. Thus, we use the homogeneous
    //      (feedback-free) ACG turnover mass. It is important to remember to fix this when issue
    //      #470 is fixed!
    sum_nion = EvaluateNionTs(zp, log10(sc.mturn_acg_homogeneous), &sc);
    if (astro_options_global->USE_MINI_HALOS) {
        sum_nion_mini = EvaluateNionTs_MINI(zp, log10(sc.mturn_acg_homogeneous),
                                            rad_setup->ave_log10_MturnLW[0], &sc);
    }

    LOG_DEBUG("nion zp = %.3e (%.3e MINI)", sum_nion, sum_nion_mini);

    double ION_EFF_FACTOR, ION_EFF_FACTOR_MINI = 0.;
    if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
        ION_EFF_FACTOR = astro_params_global->F_STAR10 * astro_params_global->F_ESC10 *
                         astro_params_global->POP2_ION;
        ION_EFF_FACTOR_MINI = astro_params_global->F_STAR7_MINI * astro_params_global->F_ESC7_MINI *
                              astro_params_global->POP3_ION;
    } else {
        // no mini-halos when SOURCE_MODE is mass independent (constant ionization efficiency)
        ION_EFF_FACTOR = astro_params_global->HII_EFF_FACTOR;
    }

    // NOTE: only used without MASS_DEPENDENT_ZETA
    rad_setup->Q_HI_zp = 1 - (ION_EFF_FACTOR * sum_nion + ION_EFF_FACTOR_MINI * sum_nion_mini) /
                                 (1.0 - rad_setup->x_e_ave_zp);

    // Initialise freq tables & prefactors (x_e by R tables)
    fill_freqint_tables(zp, rad_setup, &sc);

    // free the global tables if we used them
    free_global_tables();

    return sum_nion + sum_nion_mini > 1e-15 ? 0 : 1;  // NO_LIGHT returned
}

/*
    This function calculates calculates R-indpendent quantities that are useful for the computation
   of the radiation fields (x-ray heating rate, photoionization rate, lyman alpha flux, etc.). This
   is done by setting the fields in the input rad_setup.
*/
int SetupRadiationFields(float redshift, TsBox *previous_spin_temp,
                         RadiationFieldsSetup *rad_setup) {
    int status;
    Try {  // This Try{} wraps the whole function.
        index_huge box_ct;

        // setup the R_ct 1D arrays
        setup_z_edges(redshift, rad_setup);

        calculate_spectral_factors(redshift, rad_setup);

        double x_e_ave_p = 0.0;
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for reduction(+ : x_e_ave_p)
            for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
                x_e_ave_p += previous_spin_temp->xray_ionised_fraction[box_ct];
            }
        }
        rad_setup->x_e_ave_zp = x_e_ave_p / (float)HII_TOT_NUM_PIXELS;
        LOG_DEBUG("Prev Box: x_e_ave %.3e", rad_setup->x_e_ave_zp);

        // this should initialise and use the global tables (given box average turnovers)
        //   and use them to give: Filling factor at zp (only used for !MASS_DEPENDENT_ZETA to get
        //   ion_eff) global SFRD at each filter radius (numerator of ST_over_PS factor)

        rad_setup->NO_LIGHT = global_reion_properties(redshift, rad_setup);

#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
        {
            float xHII_call;
#pragma omp for
            for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
                xHII_call = previous_spin_temp->xray_ionised_fraction[box_ct];
                // Check if ionized fraction is within boundaries; if not, adjust to be within
                if (xHII_call > x_int_XHII[x_int_NXHII - 1] * 0.999) {
                    xHII_call = x_int_XHII[x_int_NXHII - 1] * 0.999;
                } else if (xHII_call < x_int_XHII[0]) {
                    xHII_call = 1.001 * x_int_XHII[0];
                }
                // these are the index and interpolation term, moved outside the R loop and stored
                // to not calculate them R times
                rad_setup->m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);
                rad_setup->inverse_val_box[box_ct] =
                    (xHII_call - x_int_XHII[rad_setup->m_xHII_low_box[box_ct]]) *
                    rad_setup->inverse_diff[rad_setup->m_xHII_low_box[box_ct]];
            }
        }
    }  // End of try
    Catch(status) { return (status); }
    return (0);
}

/*
    This function helps to calculate the radiation fields (x-ray heating rate, photoionization rate,
   lyman alpha flux, etc.). The radiation fields are all given by an integral over the past source
   emissivities. Numerically, this integral is evaluated via the trapezoidal rule, which is done by
   summing over the contributions from each redshift shell. Thus, this function calculates the
   contribution from a single redshift shell (zpp) and adds it to the total radiation fields. This
   is done by filling the arrays in RadiationFields.
*/
void accumulate_radiation_shell(RadiationFieldsSetup *rad_setup, RadiationFields *radiation_fields,
                                int R_ct) {
    index_huge box_ct;
    double z_edge_factor, dzpp_for_evolve, zpp, xray_R_factor;
    double lya_flux_continuum_prefactor_mini = 0., lya_flux_injected_prefactor_mini = 0.,
           lya_flux_continuum_injected_prefactor_mini = 0.;

    dzpp_for_evolve = rad_setup->dzpp_list[R_ct];
    zpp = rad_setup->zpp_for_evolve_list[R_ct];
    // dtdz'' dz'' -> dR for the radius sum (c included in constants)
    z_edge_factor = fabs(dzpp_for_evolve * rad_setup->dtdz_list[R_ct]);

    xray_R_factor = pow(1 + zpp, -(astro_params_global->X_RAY_SPEC_INDEX));

    // minihalo factors should be separated since they may not be allocated
    if (astro_options_global->USE_MINI_HALOS) {
        if (astro_options_global->USE_LYA_HEATING) {
            lya_flux_continuum_prefactor_mini = rad_setup->lya_flux_continuum_prefactor_MINI[R_ct];
            lya_flux_injected_prefactor_mini = rad_setup->lya_flux_injected_prefactor_MINI[R_ct];
        } else {
            lya_flux_continuum_injected_prefactor_mini =
                rad_setup->lya_flux_continuum_injected_prefactor_MINI[R_ct];
        }
    }

// NOTE: The ionisation box has a final delta dependence of (1+delta_source)/(1+delta_absorber)
//   But here it's just (1+delta_source). This is for photon conservation.
//   If we assume attenuation at mean density as we do in nu_tau_one(), we HAVE to assume mean
//   density absorption otherwise we do not conserve photons
#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        // private variables
        int xidx, freq_table_index;
        double ival;
        double freq_int_heat, freq_int_ion, freq_int_lya;
        double sfr_term, xray_sfr;
        double sfr_term_mini = 0;
        double sfr_term_lw, sfr_term_mini_lw;
        int num_R = astro_params_global->N_STEP_TS;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            sfr_term = radiation_fields->filtered_sfr[box_ct] * z_edge_factor;
            // Minihalos and s->yr conversion are already included here
            xray_sfr =
                radiation_fields->filtered_xray[box_ct] * z_edge_factor * xray_R_factor * 1e38;
            if (astro_options_global->USE_MINI_HALOS &&
                astro_options_global->LYA_MULTIPLE_SCATTERING) {
                sfr_term_lw = radiation_fields->filtered_sfr_lw[box_ct] * z_edge_factor;
            } else {
                sfr_term_lw = sfr_term;
            }
            if (astro_options_global->USE_MINI_HALOS) {
                sfr_term_mini = radiation_fields->filtered_sfr_mini[box_ct] * z_edge_factor;
                if (astro_options_global->LYA_MULTIPLE_SCATTERING) {
                    sfr_term_mini_lw =
                        radiation_fields->filtered_sfr_mini_lw[box_ct] * z_edge_factor;
                } else {
                    sfr_term_mini_lw = sfr_term_mini;
                }
            }

            // Evaluate the frequency integrals for this shell (R_ct) and cell (box_ct) via
            // linear interpolation
            xidx = rad_setup->m_xHII_low_box[box_ct];
            ival = rad_setup->inverse_val_box[box_ct];
            freq_table_index = freq_index(xidx, R_ct, num_R);
            freq_int_heat = rad_setup->freq_int_heat_tbl_diff[freq_table_index] * ival +
                            rad_setup->freq_int_heat_tbl[freq_table_index];
            freq_int_ion = rad_setup->freq_int_ion_tbl_diff[freq_table_index] * ival +
                           rad_setup->freq_int_ion_tbl[freq_table_index];
            freq_int_lya = rad_setup->freq_int_lya_tbl_diff[freq_table_index] * ival +
                           rad_setup->freq_int_lya_tbl[freq_table_index];

            // Evaluate the radiation fields by adding the contribution from this shell
            // (R_ct) This implements trapezoidal integration over the shells
            if (astro_options_global->USE_X_RAY_HEATING) {
                radiation_fields->xray_heating_rate[box_ct] += xray_sfr * freq_int_heat;
            }
            radiation_fields->xray_ionization_rate[box_ct] += xray_sfr * freq_int_ion;
            radiation_fields->xray_lya_flux[box_ct] += xray_sfr * freq_int_lya;
            if (astro_options_global->USE_MINI_HALOS) {
                radiation_fields->lyw_flux[box_ct] +=
                    sfr_term_lw * rad_setup->lyw_flux_prefactor[R_ct] +
                    sfr_term_mini_lw * rad_setup->lyw_flux_prefactor_MINI[R_ct];
            }
            if (astro_options_global->USE_LYA_HEATING) {
                radiation_fields->lya_flux_continuum[box_ct] +=
                    sfr_term * rad_setup->lya_flux_continuum_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_continuum_prefactor_mini;
                radiation_fields->lya_flux_injected[box_ct] +=
                    sfr_term * rad_setup->lya_flux_injected_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_injected_prefactor_mini;
            } else {
                radiation_fields->lya_flux_continuum_injected[box_ct] +=
                    sfr_term * rad_setup->lya_flux_continuum_injected_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_continuum_injected_prefactor_mini;
            }
        }
    }
}

/*
    This function multiplies the radiation fields by the appropriate constants to convert them to
   physical meaningful quantities. The radiation fields are all given by an integral over the past
   source emissivities. Numerically, this integral is evaluated via the trapezoidal rule, which is
   done by summing over the contributions from each redshift shell. For efficiency, the constants
   are not included in the integral, but rather multiplied at the end (outside the integral). This
   function does that multiplication.
*/
void multiply_radiation_fields_by_constants(float redshift, RadiationFields *radiation_fields,
                                            float perturbed_field_redshift,
                                            PerturbedField *perturbed_field,
                                            TsBox *previous_spin_temp) {
    double luminosity_converstion_factor, xray_prefactor, volunit_inv, Nb_zp, lya_star_prefactor;
    double growth_factor_z, growth_factor_zp, inverse_growth_factor_z;

    if (fabs(astro_params_global->X_RAY_SPEC_INDEX - 1.0) < 1e-6) {
        luminosity_converstion_factor =
            (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz *
            log(astro_params_global->NU_X_BAND_MAX / (astro_params_global->NU_X_THRESH));
        luminosity_converstion_factor = 1. / luminosity_converstion_factor;
    } else {
        luminosity_converstion_factor =
            pow((astro_params_global->NU_X_BAND_MAX) * physconst.eV_to_Hz,
                1. - (astro_params_global->X_RAY_SPEC_INDEX)) -
            pow((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz,
                1. - (astro_params_global->X_RAY_SPEC_INDEX));
        luminosity_converstion_factor = 1. / luminosity_converstion_factor;
        luminosity_converstion_factor *=
            pow((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz,
                -(astro_params_global->X_RAY_SPEC_INDEX)) *
            (1 - (astro_params_global->X_RAY_SPEC_INDEX));
    }
    // Finally, convert to the correct units. physconst.eV_to_Hz*physconst.h_p as only want to
    // divide by eV -> erg (owing to the definition of Luminosity)
    luminosity_converstion_factor /= (physconst.h_p);

    // for halos, we just want the SFR -> X-ray part
    // NOTE: compared to Mesinger+11: (1+zpp)^2 (1+zp) -> (1+zp)^3
    //(1+z)^3 is here because we don't want it in the
    // star lya (already in zpp integrand)
    xray_prefactor = luminosity_converstion_factor /
                     ((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz) * physconst.c_cms *
                     pow(1 + redshift, astro_params_global->X_RAY_SPEC_INDEX + 3);
    Nb_zp = N_b0 * (1 + redshift) * (1 + redshift) * (1 + redshift);
    // converts SFR density -> stellar baryon density + prefactors
    lya_star_prefactor = physconst.c_cms / (4.0 * M_PI) * physconst.Msun / physconst.m_p *
                         (1 - 0.75 * cosmo_params_global->Y_He);

    growth_factor_z = dicke(perturbed_field_redshift);
    inverse_growth_factor_z = 1. / growth_factor_z;
    growth_factor_zp = dicke(redshift);

    // converts the grid emissivity unit to per cm-3
    volunit_inv = pow(physconst.cm_per_Mpc, -3);

    index_huge box_ct;
#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        double curr_delta, prev_xe;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            curr_delta =
                perturbed_field->density[box_ct] * growth_factor_zp * inverse_growth_factor_z;
            prev_xe = previous_spin_temp->xray_ionised_fraction[box_ct];
            if (astro_options_global->USE_X_RAY_HEATING) {
                radiation_fields->xray_heating_rate[box_ct] *=
                    xray_prefactor * volunit_inv * 2.0 / 3.0 / physconst.k_B / (1.0 + prev_xe);
                ;
            }
            radiation_fields->xray_ionization_rate[box_ct] *= xray_prefactor * volunit_inv;
            radiation_fields->xray_lya_flux[box_ct] *=
                xray_prefactor * volunit_inv * Nb_zp * (1 + curr_delta);
            if (astro_options_global->USE_MINI_HALOS) {
                radiation_fields->lyw_flux[box_ct] *=
                    lya_star_prefactor * volunit_inv * physconst.h_p * 1e21;
            }
            if (astro_options_global->USE_LYA_HEATING) {
                radiation_fields->lya_flux_continuum[box_ct] *= lya_star_prefactor * volunit_inv;
                radiation_fields->lya_flux_injected[box_ct] *= lya_star_prefactor * volunit_inv;
            } else {
                radiation_fields->lya_flux_continuum_injected[box_ct] *=
                    lya_star_prefactor * volunit_inv;
            }
        }
    }
}

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

int UpdateRadiationFields(float redshift, HaloBox *halobox, double R_inner, double R_outer,
                          int R_ct, double R_star, float perturbed_field_redshift,
                          PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                          RadiationFieldsSetup *rad_setup, RadiationFields *radiation_fields) {
    int status;
    Try {  // This Try{} wraps the whole function.
        // NOTE: we assume that the first iteration corresponds to the largest shell.
        // This is important because we must do some final steps after the last shell is
        // done, see comment below
        if (R_ct == astro_params_global->N_STEP_TS - 1) LOG_DEBUG("starting RadiationFields");

        double sfr_avg, fsfr_avg, sfr_avg_mini = 0., fsfr_avg_mini = 0.;
        double xray_avg, fxray_avg;
        int filter_type = astro_options_global->LYA_MULTIPLE_SCATTERING
                              ? FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING
                              : FILTER_SPHERICAL_SHELL_STRAIGHT_LINE;

        one_annular_filter(halobox->halo_sfr, radiation_fields->filtered_sfr, R_inner, R_outer,
                           R_star, filter_type, &sfr_avg, &fsfr_avg);
        one_annular_filter(halobox->halo_xray, radiation_fields->filtered_xray, R_inner, R_outer,
                           R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, &xray_avg, &fxray_avg);
        if (astro_options_global->USE_MINI_HALOS) {
            one_annular_filter(halobox->halo_sfr_mini, radiation_fields->filtered_sfr_mini, R_inner,
                               R_outer, R_star, filter_type, &sfr_avg_mini, &fsfr_avg_mini);
            // In case of multiple scattering and mini-halos, we need to filter the SFRD
            // fields again for the the LW feedback, as these photons travel in straight
            // lines
            if (astro_options_global->LYA_MULTIPLE_SCATTERING) {
                one_annular_filter(halobox->halo_sfr, radiation_fields->filtered_sfr_lw, R_inner,
                                   R_outer, R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, &sfr_avg,
                                   &fsfr_avg);
                one_annular_filter(halobox->halo_sfr_mini, radiation_fields->filtered_sfr_mini_lw,
                                   R_inner, R_outer, R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE,
                                   &sfr_avg_mini, &fsfr_avg_mini);
            }
        }

        LOG_SUPER_DEBUG("R = [%8.3f - %8.3f] | mean filtered sfr  = %10.3e unfiltered %10.3e",
                        R_inner, R_outer, fsfr_avg, sfr_avg);
        LOG_ULTRA_DEBUG("mean filtered xray = %10.3e unfiltered %10.3e", fxray_avg, xray_avg);
        if (astro_options_global->USE_MINI_HALOS) {
            LOG_SUPER_DEBUG("MINI: filtered sfr %10.3e unfiltered %10.3e", fsfr_avg_mini,
                            sfr_avg_mini);
        }

        // Given the filtered emissivities, we accumulate the contribution of this shell to
        // the radiation fields
        accumulate_radiation_shell(rad_setup, radiation_fields, R_ct);

        // At the final shell, multiply by constants and free the remaining arrays
        // NOTE: we assume that the last iteration corresponds to the smallest shell
        // This is important in order to be consistent with the shell ordering we make
        // in compute_radiation_fields in python
        if (R_ct == 0) {
            multiply_radiation_fields_by_constants(redshift, radiation_fields,
                                                   perturbed_field_redshift, perturbed_field,
                                                   previous_spin_temp);
            // free fftwf only if we have a full box (with more than one cell)
            if (simulation_options_global->HII_DIM > 1) {
                fftwf_forget_wisdom();
                fftwf_cleanup_threads();
                fftwf_cleanup();
            }
            LOG_DEBUG("finished RadiationFields");
        }
    }  // End of try
    Catch(status) { return (status); }
    return (0);
}
