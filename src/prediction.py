import jax
import jax.numpy as jnp

from water_balance_jax import (
    construct_Kpet_crop,
    construct_Kpet_gen,
    wbm_jax,
)


##########################
# Prediction function
##########################
def make_prediction(theta, constants, x_forcing_nt, x_forcing_nyrs, x_maps):
    # Read inputs
    tas, prcp = x_forcing_nt
    lai = x_forcing_nyrs

    (
        awCap,
        wiltingp,
        Ws_init,
        clayfrac,
        sandfrac,
        siltfrac,
        rootDepth,
        lats,
        elev_std,
        corn,
        cotton,
        rice,
        sorghum,
        soybeans,
        wheat,
        cropland_other,
        evergreen_needleleaf,
        evergreen_broadleaf,
        deciduous_needleleaf,
        deciduous_broadleaf,
        mixed_forest,
        woodland,
        wooded_grassland,
        closed_shurbland,
        open_shrubland,
        grassland,
        barren,
        urban,
    ) = x_maps

    # Define all constants
    Ts, Tm, Wi_init, Sp_init = constants

    # Define all params
    (
        awCap_scalar,
        wiltingp_scalar,
        alpha_claycoef,
        alpha_sandcoef,
        alpha_siltcoef,
        betaHBV_claycoef,
        betaHBV_sandcoef,
        betaHBV_siltcoef,
        betaHBV_elevcoef,
        GS_start_corn,
        GS_end_corn,
        L_ini_corn,
        L_dev_corn,
        L_mid_corn,
        Kc_ini_corn,
        Kc_mid_corn,
        Kc_end_corn,
        Kmin_corn,
        Kmax_corn,
        c_lai_corn,
        GS_start_cotton,
        GS_end_cotton,
        L_ini_cotton,
        L_dev_cotton,
        L_mid_cotton,
        Kc_ini_cotton,
        Kc_mid_cotton,
        Kc_end_cotton,
        Kmin_cotton,
        Kmax_cotton,
        c_lai_cotton,
        GS_start_rice,
        GS_end_rice,
        L_ini_rice,
        L_dev_rice,
        L_mid_rice,
        Kc_ini_rice,
        Kc_mid_rice,
        Kc_end_rice,
        Kmin_rice,
        Kmax_rice,
        c_lai_rice,
        GS_start_sorghum,
        GS_end_sorghum,
        L_ini_sorghum,
        L_dev_sorghum,
        L_mid_sorghum,
        Kc_ini_sorghum,
        Kc_mid_sorghum,
        Kc_end_sorghum,
        Kmin_sorghum,
        Kmax_sorghum,
        c_lai_sorghum,
        GS_start_soybeans,
        GS_end_soybeans,
        L_ini_soybeans,
        L_dev_soybeans,
        L_mid_soybeans,
        Kc_ini_soybeans,
        Kc_mid_soybeans,
        Kc_end_soybeans,
        Kmin_soybeans,
        Kmax_soybeans,
        c_lai_soybeans,
        GS_start_wheat,
        GS_end_wheat,
        L_ini_wheat,
        L_dev_wheat,
        L_mid_wheat,
        Kc_ini_wheat,
        Kc_mid_wheat,
        Kc_end_wheat,
        Kmin_wheat,
        Kmax_wheat,
        c_lai_wheat,
        Kmin_cropland_other,
        Kmax_cropland_other,
        c_lai_cropland_other,
        Kmin_evergreen_needleleaf,
        Kmax_evergreen_needleleaf,
        c_lai_evergreen_needleleaf,
        Kmin_evergreen_broadleaf,
        Kmax_evergreen_broadleaf,
        c_lai_evergreen_broadleaf,
        Kmin_deciduous_needleleaf,
        Kmax_deciduous_needleleaf,
        c_lai_deciduous_needleleaf,
        Kmin_deciduous_broadleaf,
        Kmax_deciduous_broadleaf,
        c_lai_deciduous_broadleaf,
        Kmin_mixed_forest,
        Kmax_mixed_forest,
        c_lai_mixed_forest,
        Kmin_woodland,
        Kmax_woodland,
        c_lai_woodland,
        Kmin_wooded_grassland,
        Kmax_wooded_grassland,
        c_lai_wooded_grassland,
        Kmin_closed_shurbland,
        Kmax_closed_shurbland,
        c_lai_closed_shurbland,
        Kmin_open_shrubland,
        Kmax_open_shrubland,
        c_lai_open_shrubland,
        Kmin_grassland,
        Kmax_grassland,
        c_lai_grassland,
        Kmin_barren,
        Kmax_barren,
        c_lai_barren,
        Kmin_urban,
        Kmax_urban,
        c_lai_urban,
    ) = jnp.exp(theta)

    # Construct Kpet as weighted average
    Kpet_corn = construct_Kpet_crop(
        GS_start_corn,
        GS_end_corn,
        L_ini_corn,
        L_dev_corn,
        L_mid_corn,
        1.0 - (L_ini_corn + L_dev_corn + L_mid_corn),
        Kc_ini_corn,
        Kc_mid_corn,
        Kc_end_corn,
        Kmin_corn,
        Kmax_corn,
        c_lai_corn,
        lai,
    )
    Kpet_cotton = construct_Kpet_crop(
        GS_start_cotton,
        GS_end_cotton,
        L_ini_cotton,
        L_dev_cotton,
        L_mid_cotton,
        1.0 - (L_ini_cotton + L_dev_cotton + L_mid_cotton),
        Kc_ini_cotton,
        Kc_mid_cotton,
        Kc_end_cotton,
        Kmin_cotton,
        Kmax_cotton,
        c_lai_cotton,
        lai,
    )
    Kpet_rice = construct_Kpet_crop(
        GS_start_rice,
        GS_end_rice,
        L_ini_rice,
        L_dev_rice,
        L_mid_rice,
        1.0 - (L_ini_rice + L_dev_rice + L_mid_rice),
        Kc_ini_rice,
        Kc_mid_rice,
        Kc_end_rice,
        Kmin_rice,
        Kmax_rice,
        c_lai_rice,
        lai,
    )
    Kpet_sorghum = construct_Kpet_crop(
        GS_start_sorghum,
        GS_end_sorghum,
        L_ini_sorghum,
        L_dev_sorghum,
        L_mid_sorghum,
        1.0 - (L_ini_sorghum + L_dev_sorghum + L_mid_sorghum),
        Kc_ini_sorghum,
        Kc_mid_sorghum,
        Kc_end_sorghum,
        Kmin_sorghum,
        Kmax_sorghum,
        c_lai_sorghum,
        lai,
    )
    Kpet_soybeans = construct_Kpet_crop(
        GS_start_soybeans,
        GS_end_soybeans,
        L_ini_soybeans,
        L_dev_soybeans,
        L_mid_soybeans,
        1.0 - (L_ini_soybeans + L_dev_soybeans + L_mid_soybeans),
        Kc_ini_soybeans,
        Kc_mid_soybeans,
        Kc_end_soybeans,
        Kmin_soybeans,
        Kmax_soybeans,
        c_lai_soybeans,
        lai,
    )
    Kpet_wheat = construct_Kpet_crop(
        GS_start_wheat,
        GS_end_wheat,
        L_ini_wheat,
        L_dev_wheat,
        L_mid_wheat,
        1.0 - (L_ini_wheat + L_dev_wheat + L_mid_wheat),
        Kc_ini_wheat,
        Kc_mid_wheat,
        Kc_end_wheat,
        Kmin_wheat,
        Kmax_wheat,
        c_lai_wheat,
        lai,
    )

    Kpet_cropland_other = construct_Kpet_gen(
        Kmin_cropland_other, Kmax_cropland_other, c_lai_cropland_other, lai
    )
    Kpet_evergreen_needleleaf = construct_Kpet_gen(
        Kmin_evergreen_needleleaf,
        Kmax_evergreen_needleleaf,
        c_lai_evergreen_needleleaf,
        lai,
    )
    Kpet_evergreen_broadleaf = construct_Kpet_gen(
        Kmin_evergreen_broadleaf,
        Kmax_evergreen_broadleaf,
        c_lai_evergreen_broadleaf,
        lai,
    )
    Kpet_deciduous_needleleaf = construct_Kpet_gen(
        Kmin_deciduous_needleleaf,
        Kmax_deciduous_needleleaf,
        c_lai_deciduous_needleleaf,
        lai,
    )
    Kpet_deciduous_broadleaf = construct_Kpet_gen(
        Kmin_deciduous_broadleaf,
        Kmax_deciduous_broadleaf,
        c_lai_deciduous_broadleaf,
        lai,
    )
    Kpet_mixed_forest = construct_Kpet_gen(
        Kmin_mixed_forest, Kmax_mixed_forest, c_lai_mixed_forest, lai
    )
    Kpet_woodland = construct_Kpet_gen(
        Kmin_woodland, Kmax_woodland, c_lai_woodland, lai
    )
    Kpet_wooded_grassland = construct_Kpet_gen(
        Kmin_wooded_grassland,
        Kmax_wooded_grassland,
        c_lai_wooded_grassland,
        lai,
    )
    Kpet_closed_shurbland = construct_Kpet_gen(
        Kmin_closed_shurbland,
        Kmax_closed_shurbland,
        c_lai_closed_shurbland,
        lai,
    )
    Kpet_open_shrubland = construct_Kpet_gen(
        Kmin_open_shrubland, Kmax_open_shrubland, c_lai_open_shrubland, lai
    )
    Kpet_grassland = construct_Kpet_gen(
        Kmin_grassland, Kmax_grassland, c_lai_grassland, lai
    )
    Kpet_barren = construct_Kpet_gen(
        Kmin_barren, Kmax_barren, c_lai_barren, lai
    )
    Kpet_urban = construct_Kpet_gen(Kmin_urban, Kmax_urban, c_lai_urban, lai)

    weights = jnp.array(
        [
            corn,
            cotton,
            rice,
            sorghum,
            soybeans,
            wheat,
            cropland_other,
            evergreen_needleleaf,
            evergreen_broadleaf,
            deciduous_needleleaf,
            deciduous_broadleaf,
            mixed_forest,
            woodland,
            wooded_grassland,
            closed_shurbland,
            open_shrubland,
            grassland,
            barren,
            urban,
        ]
    )
    Kpets = jnp.array(
        [
            Kpet_corn,
            Kpet_cotton,
            Kpet_rice,
            Kpet_sorghum,
            Kpet_soybeans,
            Kpet_wheat,
            Kpet_cropland_other,
            Kpet_evergreen_needleleaf,
            Kpet_evergreen_broadleaf,
            Kpet_deciduous_needleleaf,
            Kpet_deciduous_broadleaf,
            Kpet_mixed_forest,
            Kpet_woodland,
            Kpet_wooded_grassland,
            Kpet_closed_shurbland,
            Kpet_open_shrubland,
            Kpet_grassland,
            Kpet_barren,
            Kpet_urban,
        ]
    )
    Kpet = jnp.average(Kpets, weights=weights, axis=0)

    # params that WBM sees
    awCap_scaled = awCap_scalar * awCap
    wiltingp_scaled = wiltingp_scalar * wiltingp

    alpha = (
        1.0
        + (alpha_claycoef * clayfrac)
        + (alpha_sandcoef * sandfrac)
        + (alpha_siltcoef * siltfrac)
    )

    betaHBV = (
        1.0
        + (betaHBV_claycoef * clayfrac)
        + (betaHBV_sandcoef * sandfrac)
        + (betaHBV_siltcoef * siltfrac)
        + (betaHBV_elevcoef * elev_std)
    )

    params = (Ts, Tm, awCap_scaled, wiltingp_scaled, rootDepth, alpha, betaHBV)

    # Make prediction
    prediction = wbm_jax(
        tas, prcp, Kpet, Ws_init, Wi_init, Sp_init, lai, lats, params
    )

    return prediction


############################
# Prediction function vmap
############################
make_prediction_vmap = jax.jit(
    jax.vmap(make_prediction, in_axes=(None, None, 0, 0, 0), out_axes=0)
)


#############################################
#############################################
#############################################
#############################################
# OLD
#############################################
#############################################
#############################################
#############################################

# # Prediction functions: VIC
# def make_prediction_vic(
#     theta, constants, x_forcing_nt, x_forcing_nyrs, x_maps
# ):
#     # Read inputs
#     tas, prcp = x_forcing_nt
#     lai = x_forcing_nyrs

#     (
#         awCap_frac,
#         wiltingp_frac,
#         sand,
#         loamy_sand,
#         sandy_loam,
#         silt_loam,
#         silt,
#         loam,
#         sandy_clay_loam,
#         silty_clay_loam,
#         clay_loam,
#         sandy_clay,
#         silty_clay,
#         clay,
#         Ws_init,
#         clayfrac,
#         sandfrac,
#         siltfrac,
#         rootDepth,
#         lats,
#         elev_std,
#         corn,
#         cotton,
#         rice,
#         sorghum,
#         soybeans,
#         wheat,
#     ) = x_maps

#     # Define all constants
#     Ts, Tm, Wi_init, Sp_init = constants

#     # Define all params
#     (
#         awCap_sand,
#         awCap_loamy_sand,
#         awCap_sandy_loam,
#         awCap_silt_loam,
#         awCap_silt,
#         awCap_loam,
#         awCap_sandy_clay_loam,
#         awCap_silty_clay_loam,
#         awCap_clay_loam,
#         awCap_sandy_clay,
#         awCap_silty_clay,
#         awCap_clay,
#         wiltingp_sand,
#         wiltingp_loamy_sand,
#         wiltingp_sandy_loam,
#         wiltingp_silt_loam,
#         wiltingp_silt,
#         wiltingp_loam,
#         wiltingp_sandy_clay_loam,
#         wiltingp_silty_clay_loam,
#         wiltingp_clay_loam,
#         wiltingp_sandy_clay,
#         wiltingp_silty_clay,
#         wiltingp_clay,
#         # awCap_claycoef,
#         # awCap_sandcoef,
#         # awCap_siltcoef,
#         # wiltingp_claycoef,
#         # wiltingp_sandcoef,
#         # wiltingp_siltcoef,
#         alpha_claycoef,
#         alpha_sandcoef,
#         alpha_siltcoef,
#         betaHBV_claycoef,
#         betaHBV_sandcoef,
#         betaHBV_siltcoef,
#         betaHBV_elevcoef,
#         GS_start_corn,
#         GS_end_corn,
#         L_ini_corn,
#         L_dev_corn,
#         L_mid_corn,
#         Kc_ini_corn,
#         Kc_mid_corn,
#         Kc_end_corn,
#         Kmin_corn,
#         Kmax_corn,
#         GS_start_cotton,
#         GS_end_cotton,
#         L_ini_cotton,
#         L_dev_cotton,
#         L_mid_cotton,
#         Kc_ini_cotton,
#         Kc_mid_cotton,
#         Kc_end_cotton,
#         Kmin_cotton,
#         Kmax_cotton,
#         GS_start_rice,
#         GS_end_rice,
#         L_ini_rice,
#         L_dev_rice,
#         L_mid_rice,
#         Kc_ini_rice,
#         Kc_mid_rice,
#         Kc_end_rice,
#         Kmin_rice,
#         Kmax_rice,
#         GS_start_sorghum,
#         GS_end_sorghum,
#         L_ini_sorghum,
#         L_dev_sorghum,
#         L_mid_sorghum,
#         Kc_ini_sorghum,
#         Kc_mid_sorghum,
#         Kc_end_sorghum,
#         Kmin_sorghum,
#         Kmax_sorghum,
#         GS_start_soybeans,
#         GS_end_soybeans,
#         L_ini_soybeans,
#         L_dev_soybeans,
#         L_mid_soybeans,
#         Kc_ini_soybeans,
#         Kc_mid_soybeans,
#         Kc_end_soybeans,
#         Kmin_soybeans,
#         Kmax_soybeans,
#         GS_start_wheat,
#         GS_end_wheat,
#         L_ini_wheat,
#         L_dev_wheat,
#         L_mid_wheat,
#         Kc_ini_wheat,
#         Kc_mid_wheat,
#         Kc_end_wheat,
#         Kmin_wheat,
#         Kmax_wheat,
#     ) = jnp.exp(theta)

#     # Construct Kpet as weighted average
#     Kpet_corn = construct_Kpet_crop(
#         GS_start_corn,
#         GS_end_corn,
#         L_ini_corn,
#         L_dev_corn,
#         L_mid_corn,
#         1.0 - (L_ini_corn + L_dev_corn + L_mid_corn),
#         Kc_ini_corn,
#         Kc_mid_corn,
#         Kc_end_corn,
#         Kmin_corn,
#         Kmax_corn,
#         lai,
#     )
#     Kpet_cotton = construct_Kpet_crop(
#         GS_start_cotton,
#         GS_end_cotton,
#         L_ini_cotton,
#         L_dev_cotton,
#         L_mid_cotton,
#         1.0 - (L_ini_cotton + L_dev_cotton + L_mid_cotton),
#         Kc_ini_cotton,
#         Kc_mid_cotton,
#         Kc_end_cotton,
#         Kmin_cotton,
#         Kmax_cotton,
#         lai,
#     )
#     Kpet_rice = construct_Kpet_crop(
#         GS_start_rice,
#         GS_end_rice,
#         L_ini_rice,
#         L_dev_rice,
#         L_mid_rice,
#         1.0 - (L_ini_rice + L_dev_rice + L_mid_rice),
#         Kc_ini_rice,
#         Kc_mid_rice,
#         Kc_end_rice,
#         Kmin_rice,
#         Kmax_rice,
#         lai,
#     )
#     Kpet_sorghum = construct_Kpet_crop(
#         GS_start_sorghum,
#         GS_end_sorghum,
#         L_ini_sorghum,
#         L_dev_sorghum,
#         L_mid_sorghum,
#         1.0 - (L_ini_sorghum + L_dev_sorghum + L_mid_sorghum),
#         Kc_ini_sorghum,
#         Kc_mid_sorghum,
#         Kc_end_sorghum,
#         Kmin_sorghum,
#         Kmax_sorghum,
#         lai,
#     )
#     Kpet_soybeans = construct_Kpet_crop(
#         GS_start_soybeans,
#         GS_end_soybeans,
#         L_ini_soybeans,
#         L_dev_soybeans,
#         L_mid_soybeans,
#         1.0 - (L_ini_soybeans + L_dev_soybeans + L_mid_soybeans),
#         Kc_ini_soybeans,
#         Kc_mid_soybeans,
#         Kc_end_soybeans,
#         Kmin_soybeans,
#         Kmax_soybeans,
#         lai,
#     )
#     Kpet_wheat = construct_Kpet_crop(
#         GS_start_wheat,
#         GS_end_wheat,
#         L_ini_wheat,
#         L_dev_wheat,
#         L_mid_wheat,
#         1.0 - (L_ini_wheat + L_dev_wheat + L_mid_wheat),
#         Kc_ini_wheat,
#         Kc_mid_wheat,
#         Kc_end_wheat,
#         Kmin_wheat,
#         Kmax_wheat,
#         lai,
#     )

#     other = 1.0 - (corn + cotton + rice + sorghum + soybeans + wheat)
#     weights = jnp.array([corn, cotton, rice, sorghum, soybeans, wheat, other])
#     Kpets = jnp.array(
#         [
#             Kpet_corn,
#             Kpet_cotton,
#             Kpet_rice,
#             Kpet_sorghum,
#             Kpet_soybeans,
#             Kpet_wheat,
#             jnp.ones(365),
#         ]
#     )
#     Kpet = jnp.average(Kpets, weights=weights, axis=0)

#     # params that WBM sees
#     awCap_scaled = 1 * (
#         (awCap_sand * sand)
#         + (awCap_loamy_sand * loamy_sand)
#         + (awCap_sandy_loam * sandy_loam)
#         + (awCap_silt_loam * silt_loam)
#         + (awCap_silt * silt)
#         + (awCap_loam * loam)
#         + (awCap_sandy_clay_loam * sandy_clay_loam)
#         + (awCap_silty_clay_loam * silty_clay_loam)
#         + (awCap_clay_loam * clay_loam)
#         + (awCap_sandy_clay * sandy_clay)
#         + (awCap_silty_clay * silty_clay)
#         + (awCap_clay * clay)
#     )
#     wiltingp_scaled = 1 * (
#         (wiltingp_sand * sand)
#         + (wiltingp_loamy_sand * loamy_sand)
#         + (wiltingp_sandy_loam * sandy_loam)
#         + (wiltingp_silt_loam * silt_loam)
#         + (wiltingp_silt * silt)
#         + (wiltingp_loam * loam)
#         + (wiltingp_sandy_clay_loam * sandy_clay_loam)
#         + (wiltingp_silty_clay_loam * silty_clay_loam)
#         + (wiltingp_clay_loam * clay_loam)
#         + (wiltingp_sandy_clay * sandy_clay)
#         + (wiltingp_silty_clay * silty_clay)
#         + (wiltingp_clay * clay)
#     )

#     alpha = (
#         1.0
#         + (alpha_claycoef * clayfrac)
#         + (alpha_sandcoef * sandfrac)
#         + (alpha_siltcoef * siltfrac)
#     )
#     betaHBV = (
#         1.0
#         + (betaHBV_claycoef * clayfrac)
#         + (betaHBV_sandcoef * sandfrac)
#         + (betaHBV_siltcoef * siltfrac)
#         + (betaHBV_elevcoef * elev_std)
#     )

#     params = (Ts, Tm, wiltingp_scaled, awCap_scaled, rootDepth, alpha, betaHBV)

#     # Make prediction
#     prediction = wbm_jax(
#         tas, prcp, Kpet, Ws_init, Wi_init, Sp_init, lai, lats, params
#     )

#     return prediction
