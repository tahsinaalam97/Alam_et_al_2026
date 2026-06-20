import jax.numpy as jnp

#################
# awCap
#################
awCap_scalar_lower, awCap_scalar_upper = (
    jnp.log(0.1),
    jnp.log(3.0),
)  # Central 1.

#################
# wiltingp
#################
wiltingp_scalar_lower, wiltingp_scalar_upper = (
    jnp.log(0.1),
    jnp.log(3.0),
)  # Central 1.

#################
# alpha
#################
alpha_claycoef_lower, alpha_claycoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
alpha_sandcoef_lower, alpha_sandcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
alpha_siltcoef_lower, alpha_siltcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5

#####################
# betaHBV
#####################
betaHBV_claycoef_lower, betaHBV_claycoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
betaHBV_sandcoef_lower, betaHBV_sandcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
betaHBV_siltcoef_lower, betaHBV_siltcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
betaHBV_elevcoef_lower, betaHBV_elevcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5

#######################
# Crops
#######################
# Corn
GS_start_corn_lower, GS_start_corn_upper = (
    jnp.log(60),
    jnp.log(152),
    # March 1st
)
GS_end_corn_lower, GS_end_corn_upper = (
    jnp.log(244),
    jnp.log(334),
)  # Sep 1st, latest Nov 30th
L_ini_corn_lower, L_ini_corn_upper = (
    jnp.log(0.07),
    jnp.log(0.22),
)  # central 0.17
L_dev_corn_lower, L_dev_corn_upper = (
    jnp.log(0.18),
    jnp.log(0.33),
)  # central 0.28
L_mid_corn_lower, L_mid_corn_upper = (
    jnp.log(0.13),
    jnp.log(0.38),
)  # central 0.33
Kc_ini_corn_lower, Kc_ini_corn_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
)  # central 0.3
Kc_mid_corn_lower, Kc_mid_corn_upper = (
    jnp.log(1.0),
    jnp.log(1.5),
)  # central 1.2
Kc_end_corn_lower, Kc_end_corn_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
Kmin_corn_lower, Kmin_corn_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
    # central 0.3
)
Kmax_corn_lower, Kmax_corn_upper = (
    jnp.log(1.0),
    jnp.log(1.5),
    # central 1.2
)
c_lai_corn_lower, c_lai_corn_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Cotton
GS_start_cotton_lower, GS_start_cotton_upper = (
    jnp.log(60),
    jnp.log(152),
    # March 1st
)
GS_end_cotton_lower, GS_end_cotton_upper = (
    jnp.log(244),
    jnp.log(334),
)  # Sep 1st, Nov 30th
L_ini_cotton_lower, L_ini_cotton_upper = (
    jnp.log(0.07),
    jnp.log(0.25),
)  # central 0.17
L_dev_cotton_lower, L_dev_cotton_upper = (
    jnp.log(0.23),
    jnp.log(0.4),
)  # central 0.33
L_mid_cotton_lower, L_mid_cotton_upper = (
    jnp.log(0.15),
    jnp.log(0.3),
)  # central 0.25
Kc_ini_cotton_lower, Kc_ini_cotton_upper = (
    jnp.log(0.15),
    jnp.log(0.65),
)  # central 0.35
Kc_mid_cotton_lower, Kc_mid_cotton_upper = (
    jnp.log(1.0),
    jnp.log(1.5),
)  # central 1.18
Kc_end_cotton_lower, Kc_end_cotton_upper = (
    jnp.log(0.4),
    jnp.log(0.8),
)  # central 0.6
Kmin_cotton_lower, Kmin_cotton_upper = (
    jnp.log(0.1),
    jnp.log(0.6),
)  # central 0.35
Kmax_cotton_lower, Kmax_cotton_upper = (
    jnp.log(1.0),
    jnp.log(1.5),
    # central 1.18
)
c_lai_cotton_lower, c_lai_cotton_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Rice growing season: usda.gov/topics/crops/rice/rice-sector-at-a-glance/
GS_start_rice_lower, GS_start_rice_upper = (
    jnp.log(60),
    jnp.log(182),
    # March 1st
)
GS_end_rice_lower, GS_end_rice_upper = (
    jnp.log(214),
    jnp.log(334),
)  # August 1st, Nov 30th
L_ini_rice_lower, L_ini_rice_upper = (
    jnp.log(0.07),
    jnp.log(0.21),
)  # central 0.17
L_dev_rice_lower, L_dev_rice_upper = (
    jnp.log(0.18),
    jnp.log(0.32),
)  # central 0.28
L_mid_rice_lower, L_mid_rice_upper = (
    jnp.log(0.34),
    jnp.log(0.48),
)  # central 0.44
Kc_ini_rice_lower, Kc_ini_rice_upper = (
    jnp.log(0.95),
    jnp.log(1.1),
)  # central 1.05
Kc_mid_rice_lower, Kc_mid_rice_upper = (
    jnp.log(1.1),
    jnp.log(1.3),
)  # central 1.2
Kc_end_rice_lower, Kc_end_rice_upper = (
    jnp.log(0.65),
    jnp.log(0.85),
)  # central 0.75
Kmin_rice_lower, Kmin_rice_upper = (
    jnp.log(0.65),
    jnp.log(0.85),
)  # central 0.75
Kmax_rice_lower, Kmax_rice_upper = (
    jnp.log(1.0),
    jnp.log(1.4),
    # central 1.2
)
c_lai_rice_lower, c_lai_rice_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Sorghum
GS_start_sorghum_lower, GS_start_sorghum_upper = (
    jnp.log(60),
    jnp.log(182),
    # March 1st
)
GS_end_sorghum_lower, GS_end_sorghum_upper = (
    jnp.log(214),
    jnp.log(334),
)  # August 1st, Nov 30th
L_ini_sorghum_lower, L_ini_sorghum_upper = (
    jnp.log(0.05),
    jnp.log(0.20),
)  # central 0.15
L_dev_sorghum_lower, L_dev_sorghum_upper = (
    jnp.log(0.18),
    jnp.log(0.33),
)  # central 0.28
L_mid_sorghum_lower, L_mid_sorghum_upper = (
    jnp.log(0.23),
    jnp.log(0.38),
)  # central 0.33
Kc_ini_sorghum_lower, Kc_ini_sorghum_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
)  # central 0.3
Kc_mid_sorghum_lower, Kc_mid_sorghum_upper = (
    jnp.log(1.0),
    jnp.log(1.2),
)  # central 1.1
Kc_end_sorghum_lower, Kc_end_sorghum_upper = (
    jnp.log(0.35),
    jnp.log(0.75),
)  # central 0.55
Kmin_sorghum_lower, Kmin_sorghum_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
)  # central 0.3
Kmax_sorghum_lower, Kmax_sorghum_upper = (
    jnp.log(1.0),
    jnp.log(1.2),
    # central 1.1
)
c_lai_sorghum_lower, c_lai_sorghum_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Soybeans
GS_start_soybeans_lower, GS_start_soybeans_upper = (
    jnp.log(60),
    jnp.log(182),
    # March 1st
)
GS_end_soybeans_lower, GS_end_soybeans_upper = (
    jnp.log(244),
    jnp.log(334),
)  # Sep 1st, Nov 30th
L_ini_soybeans_lower, L_ini_soybeans_upper = (
    jnp.log(0.05),
    jnp.log(0.2),
)  # central 0.15
L_dev_soybeans_lower, L_dev_soybeans_upper = (
    jnp.log(0.1),
    jnp.log(0.25),
)  # central 0.2
L_mid_soybeans_lower, L_mid_soybeans_upper = (
    jnp.log(0.35),
    jnp.log(0.5),
)  # central 0.45
Kc_ini_soybeans_lower, Kc_ini_soybeans_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
Kc_mid_soybeans_lower, Kc_mid_soybeans_upper = (
    jnp.log(1.0),
    jnp.log(1.3),
)  # central 1.15
Kc_end_soybeans_lower, Kc_end_soybeans_upper = (
    jnp.log(0.3),
    jnp.log(0.7),
)  # central 0.5
Kmin_soybeans_lower, Kmin_soybeans_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
Kmax_soybeans_lower, Kmax_soybeans_upper = (
    jnp.log(1.0),
    jnp.log(1.3),
    # central 1.15
)
c_lai_soybeans_lower, c_lai_soybeans_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Assume spring wheat
GS_start_wheat_lower, GS_start_wheat_upper = (
    jnp.log(60),
    jnp.log(152),
    # March 1st
)
GS_end_wheat_lower, GS_end_wheat_upper = (
    jnp.log(214),
    jnp.log(274),
)  # August 1st, Oct 1st
L_ini_wheat_lower, L_ini_wheat_upper = (
    jnp.log(0.05),
    jnp.log(0.2),
)  # central 0.15
L_dev_wheat_lower, L_dev_wheat_upper = (
    jnp.log(0.15),
    jnp.log(0.3),
)  # central 0.25
L_mid_wheat_lower, L_mid_wheat_upper = (
    jnp.log(0.3),
    jnp.log(0.45),
)  # central 0.4
Kc_ini_wheat_lower, Kc_ini_wheat_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
Kc_mid_wheat_lower, Kc_mid_wheat_upper = (
    jnp.log(1.0),
    jnp.log(1.3),
)  # central 1.15
Kc_end_wheat_lower, Kc_end_wheat_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
)  # central 0.3
Kmin_wheat_lower, Kmin_wheat_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
Kmax_wheat_lower, Kmax_wheat_upper = (
    jnp.log(1.0),
    jnp.log(1.3),
    # central 1.15
)
c_lai_wheat_lower, c_lai_wheat_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)
###################################
# Other land cateogories
####################################
# Other cropland
Kmin_cropland_other_lower, Kmin_cropland_other_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_cropland_other_lower, Kmax_cropland_other_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_cropland_other_lower, c_lai_cropland_other_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Evergreen needleleaf
Kmin_evergreen_needleleaf_lower, Kmin_evergreen_needleleaf_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_evergreen_needleleaf_lower, Kmax_evergreen_needleleaf_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_evergreen_needleleaf_lower, c_lai_evergreen_needleleaf_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Evergreen broadleaf
Kmin_evergreen_broadleaf_lower, Kmin_evergreen_broadleaf_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_evergreen_broadleaf_lower, Kmax_evergreen_broadleaf_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_evergreen_broadleaf_lower, c_lai_evergreen_broadleaf_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Deciduous needleleaf
Kmin_deciduous_needleleaf_lower, Kmin_deciduous_needleleaf_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_deciduous_needleleaf_lower, Kmax_deciduous_needleleaf_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_deciduous_needleleaf_lower, c_lai_deciduous_needleleaf_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Deciduous broadleaf
Kmin_deciduous_broadleaf_lower, Kmin_deciduous_broadleaf_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_deciduous_broadleaf_lower, Kmax_deciduous_broadleaf_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_deciduous_broadleaf_lower, c_lai_deciduous_broadleaf_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Mixed forest
Kmin_mixed_forest_lower, Kmin_mixed_forest_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_mixed_forest_lower, Kmax_mixed_forest_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_mixed_forest_lower, c_lai_mixed_forest_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Woodland
Kmin_woodland_lower, Kmin_woodland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_woodland_lower, Kmax_woodland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_woodland_lower, c_lai_woodland_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Wooded grassland
Kmin_wooded_grassland_lower, Kmin_wooded_grassland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_wooded_grassland_lower, Kmax_wooded_grassland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_wooded_grassland_lower, c_lai_wooded_grassland_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Closed shrubland
Kmin_closed_shrubland_lower, Kmin_closed_shrubland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_closed_shrubland_lower, Kmax_closed_shrubland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_closed_shrubland_lower, c_lai_closed_shrubland_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Open shrubland
Kmin_open_shrubland_lower, Kmin_open_shrubland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_open_shrubland_lower, Kmax_open_shrubland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_open_shrubland_lower, c_lai_open_shrubland_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Grassland
Kmin_grassland_lower, Kmin_grassland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_grassland_lower, Kmax_grassland_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_grassland_lower, c_lai_grassland_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Barren
Kmin_barren_lower, Kmin_barren_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_barren_lower, Kmax_barren_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_barren_lower, c_lai_barren_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

# Urban
Kmin_urban_lower, Kmin_urban_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
Kmax_urban_lower, Kmax_urban_upper = (
    jnp.log(0.1),
    jnp.log(2.0),
)
c_lai_urban_lower, c_lai_urban_upper = (
    jnp.log(0.1),
    jnp.log(1.0),
)

################
# Final
################
params_lower = jnp.array(
    [
        awCap_scalar_lower,
        wiltingp_scalar_lower,
        alpha_claycoef_lower,
        alpha_sandcoef_lower,
        alpha_siltcoef_lower,
        betaHBV_claycoef_lower,
        betaHBV_sandcoef_lower,
        betaHBV_siltcoef_lower,
        betaHBV_elevcoef_lower,
        GS_start_corn_lower,
        GS_end_corn_lower,
        L_ini_corn_lower,
        L_dev_corn_lower,
        L_mid_corn_lower,
        Kc_ini_corn_lower,
        Kc_mid_corn_lower,
        Kc_end_corn_lower,
        Kmin_corn_lower,
        Kmax_corn_lower,
        c_lai_corn_lower,
        GS_start_cotton_lower,
        GS_end_cotton_lower,
        L_ini_cotton_lower,
        L_dev_cotton_lower,
        L_mid_cotton_lower,
        Kc_ini_cotton_lower,
        Kc_mid_cotton_lower,
        Kc_end_cotton_lower,
        Kmin_cotton_lower,
        Kmax_cotton_lower,
        c_lai_cotton_lower,
        GS_start_rice_lower,
        GS_end_rice_lower,
        L_ini_rice_lower,
        L_dev_rice_lower,
        L_mid_rice_lower,
        Kc_ini_rice_lower,
        Kc_mid_rice_lower,
        Kc_end_rice_lower,
        Kmin_rice_lower,
        Kmax_rice_lower,
        c_lai_rice_lower,
        GS_start_sorghum_lower,
        GS_end_sorghum_lower,
        L_ini_sorghum_lower,
        L_dev_sorghum_lower,
        L_mid_sorghum_lower,
        Kc_ini_sorghum_lower,
        Kc_mid_sorghum_lower,
        Kc_end_sorghum_lower,
        Kmin_sorghum_lower,
        Kmax_sorghum_lower,
        c_lai_sorghum_lower,
        GS_start_soybeans_lower,
        GS_end_soybeans_lower,
        L_ini_soybeans_lower,
        L_dev_soybeans_lower,
        L_mid_soybeans_lower,
        Kc_ini_soybeans_lower,
        Kc_mid_soybeans_lower,
        Kc_end_soybeans_lower,
        Kmin_soybeans_lower,
        Kmax_soybeans_lower,
        c_lai_soybeans_lower,
        GS_start_wheat_lower,
        GS_end_wheat_lower,
        L_ini_wheat_lower,
        L_dev_wheat_lower,
        L_mid_wheat_lower,
        Kc_ini_wheat_lower,
        Kc_mid_wheat_lower,
        Kc_end_wheat_lower,
        Kmin_wheat_lower,
        Kmax_wheat_lower,
        c_lai_wheat_lower,
        Kmin_cropland_other_lower,
        Kmax_cropland_other_lower,
        c_lai_cropland_other_lower,
        Kmin_evergreen_needleleaf_lower,
        Kmax_evergreen_needleleaf_lower,
        c_lai_evergreen_needleleaf_lower,
        Kmin_evergreen_broadleaf_lower,
        Kmax_evergreen_broadleaf_lower,
        c_lai_evergreen_broadleaf_lower,
        Kmin_deciduous_needleleaf_lower,
        Kmax_deciduous_needleleaf_lower,
        c_lai_deciduous_needleleaf_lower,
        Kmin_deciduous_broadleaf_lower,
        Kmax_deciduous_broadleaf_lower,
        c_lai_deciduous_broadleaf_lower,
        Kmin_mixed_forest_lower,
        Kmax_mixed_forest_lower,
        c_lai_mixed_forest_lower,
        Kmin_woodland_lower,
        Kmax_woodland_lower,
        c_lai_woodland_lower,
        Kmin_wooded_grassland_lower,
        Kmax_wooded_grassland_lower,
        c_lai_wooded_grassland_lower,
        Kmin_closed_shrubland_lower,
        Kmax_closed_shrubland_lower,
        c_lai_closed_shrubland_lower,
        Kmin_open_shrubland_lower,
        Kmax_open_shrubland_lower,
        c_lai_open_shrubland_lower,
        Kmin_grassland_lower,
        Kmax_grassland_lower,
        c_lai_grassland_lower,
        Kmin_barren_lower,
        Kmax_barren_lower,
        c_lai_barren_lower,
        Kmin_urban_lower,
        Kmax_urban_lower,
        c_lai_urban_lower,
    ]
)


params_upper = jnp.array(
    [
        awCap_scalar_upper,
        wiltingp_scalar_upper,
        alpha_claycoef_upper,
        alpha_sandcoef_upper,
        alpha_siltcoef_upper,
        betaHBV_claycoef_upper,
        betaHBV_sandcoef_upper,
        betaHBV_siltcoef_upper,
        betaHBV_elevcoef_upper,
        GS_start_corn_upper,
        GS_end_corn_upper,
        L_ini_corn_upper,
        L_dev_corn_upper,
        L_mid_corn_upper,
        Kc_ini_corn_upper,
        Kc_mid_corn_upper,
        Kc_end_corn_upper,
        Kmin_corn_upper,
        Kmax_corn_upper,
        c_lai_corn_upper,
        GS_start_cotton_upper,
        GS_end_cotton_upper,
        L_ini_cotton_upper,
        L_dev_cotton_upper,
        L_mid_cotton_upper,
        Kc_ini_cotton_upper,
        Kc_mid_cotton_upper,
        Kc_end_cotton_upper,
        Kmin_cotton_upper,
        Kmax_cotton_upper,
        c_lai_cotton_upper,
        GS_start_rice_upper,
        GS_end_rice_upper,
        L_ini_rice_upper,
        L_dev_rice_upper,
        L_mid_rice_upper,
        Kc_ini_rice_upper,
        Kc_mid_rice_upper,
        Kc_end_rice_upper,
        Kmin_rice_upper,
        Kmax_rice_upper,
        c_lai_rice_upper,
        GS_start_sorghum_upper,
        GS_end_sorghum_upper,
        L_ini_sorghum_upper,
        L_dev_sorghum_upper,
        L_mid_sorghum_upper,
        Kc_ini_sorghum_upper,
        Kc_mid_sorghum_upper,
        Kc_end_sorghum_upper,
        Kmin_sorghum_upper,
        Kmax_sorghum_upper,
        c_lai_sorghum_upper,
        GS_start_soybeans_upper,
        GS_end_soybeans_upper,
        L_ini_soybeans_upper,
        L_dev_soybeans_upper,
        L_mid_soybeans_upper,
        Kc_ini_soybeans_upper,
        Kc_mid_soybeans_upper,
        Kc_end_soybeans_upper,
        Kmin_soybeans_upper,
        Kmax_soybeans_upper,
        c_lai_soybeans_upper,
        GS_start_wheat_upper,
        GS_end_wheat_upper,
        L_ini_wheat_upper,
        L_dev_wheat_upper,
        L_mid_wheat_upper,
        Kc_ini_wheat_upper,
        Kc_mid_wheat_upper,
        Kc_end_wheat_upper,
        Kmin_wheat_upper,
        Kmax_wheat_upper,
        c_lai_wheat_upper,
        Kmin_cropland_other_upper,
        Kmax_cropland_other_upper,
        c_lai_cropland_other_upper,
        Kmin_evergreen_needleleaf_upper,
        Kmax_evergreen_needleleaf_upper,
        c_lai_evergreen_needleleaf_upper,
        Kmin_evergreen_broadleaf_upper,
        Kmax_evergreen_broadleaf_upper,
        c_lai_evergreen_broadleaf_upper,
        Kmin_deciduous_needleleaf_upper,
        Kmax_deciduous_needleleaf_upper,
        c_lai_deciduous_needleleaf_upper,
        Kmin_deciduous_broadleaf_upper,
        Kmax_deciduous_broadleaf_upper,
        c_lai_deciduous_broadleaf_upper,
        Kmin_mixed_forest_upper,
        Kmax_mixed_forest_upper,
        c_lai_mixed_forest_upper,
        Kmin_woodland_upper,
        Kmax_woodland_upper,
        c_lai_woodland_upper,
        Kmin_wooded_grassland_upper,
        Kmax_wooded_grassland_upper,
        c_lai_wooded_grassland_upper,
        Kmin_closed_shrubland_upper,
        Kmax_closed_shrubland_upper,
        c_lai_closed_shrubland_upper,
        Kmin_open_shrubland_upper,
        Kmax_open_shrubland_upper,
        c_lai_open_shrubland_upper,
        Kmin_grassland_upper,
        Kmax_grassland_upper,
        c_lai_grassland_upper,
        Kmin_barren_upper,
        Kmax_barren_upper,
        c_lai_barren_upper,
        Kmin_urban_upper,
        Kmax_urban_upper,
        c_lai_urban_upper,
    ]
)