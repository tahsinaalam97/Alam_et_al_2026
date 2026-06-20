import jax.numpy as jnp

# Constants
Ts = -1.0  # Snowfall threshold
Tm = 1.0  # Snowmelt threshold
Wi_init = 0.0  # Initial canopy storage (could spinup to find this but miniscule difference)
Sp_init = 0.0  # Initial snowpack storage (could spinup to find this but miniscule difference)

constants = jnp.array([Ts, Tm, Wi_init, Sp_init])

## Parameters
# awCap
awCap_scalar = jnp.log(1)

# wiltingp
wiltingp_scalar = jnp.log(1.0)

# alpha
alpha_claycoef = jnp.log(0.5)
alpha_sandcoef = jnp.log(0.5)
alpha_siltcoef = jnp.log(0.5)

# betaHBV
betaHBV_claycoef = jnp.log(0.5)
betaHBV_sandcoef = jnp.log(0.5)
betaHBV_siltcoef = jnp.log(0.5)
betaHBV_elevcoef = jnp.log(0.5)

# Corn
GS_start_corn = jnp.log(91)  # April 1st
GS_end_corn = jnp.log(274)  # October 1st
L_ini_corn = jnp.log(0.17)
L_dev_corn = jnp.log(0.28)
L_mid_corn = jnp.log(0.33)
Kc_ini_corn = jnp.log(0.3)
Kc_mid_corn = jnp.log(1.2)
Kc_end_corn = jnp.log(0.4)
Kmin_corn = jnp.log(0.3)
Kmax_corn = jnp.log(1.2)
c_lai_corn = jnp.log(0.7)

# Cotton
GS_start_cotton = jnp.log(91)  # April 1st
GS_end_cotton = jnp.log(274)  # October 1st
L_ini_cotton = jnp.log(0.17)
L_dev_cotton = jnp.log(0.33)
L_mid_cotton = jnp.log(0.25)
Kc_ini_cotton = jnp.log(0.35)
Kc_mid_cotton = jnp.log(1.18)
Kc_end_cotton = jnp.log(0.6)
Kmin_cotton = jnp.log(0.35)
Kmax_cotton = jnp.log(1.18)
c_lai_cotton = jnp.log(0.7)

# Rice growing season: https://www.ers.usda.gov/topics/crops/rice/rice-sector-at-a-glance/
GS_start_rice = jnp.log(91)  # April 1st
GS_end_rice = jnp.log(244)  # September 1st
L_ini_rice = jnp.log(0.17)
L_dev_rice = jnp.log(0.28)
L_mid_rice = jnp.log(0.44)
Kc_ini_rice = jnp.log(1.05)
Kc_mid_rice = jnp.log(1.2)
Kc_end_rice = jnp.log(0.75)
Kmin_rice = jnp.log(0.75)
Kmax_rice = jnp.log(1.2)
c_lai_rice = jnp.log(0.7)

# Sorghum
GS_start_sorghum = jnp.log(91)  # April 1st
GS_end_sorghum = jnp.log(274)  # October 1st
L_ini_sorghum = jnp.log(0.15)
L_dev_sorghum = jnp.log(0.28)
L_mid_sorghum = jnp.log(0.33)
Kc_ini_sorghum = jnp.log(0.3)
Kc_mid_sorghum = jnp.log(1.1)
Kc_end_sorghum = jnp.log(0.55)
Kmin_sorghum = jnp.log(0.3)
Kmax_sorghum = jnp.log(1.1)
c_lai_sorghum = jnp.log(0.7)

# Soybeans
GS_start_soybeans = jnp.log(91)  # April 1st
GS_end_soybeans = jnp.log(274)  # October 1st
L_ini_soybeans = jnp.log(0.15)
L_dev_soybeans = jnp.log(0.2)
L_mid_soybeans = jnp.log(0.45)
Kc_ini_soybeans = jnp.log(0.4)
Kc_mid_soybeans = jnp.log(1.15)
Kc_end_soybeans = jnp.log(0.5)
Kmin_soybeans = jnp.log(0.4)
Kmax_soybeans = jnp.log(1.15)
c_lai_soybeans = jnp.log(0.7)

# Assume spring wheat
GS_start_wheat = jnp.log(91)  # April 1st
GS_end_wheat = jnp.log(244)  # September 1st
L_ini_wheat = jnp.log(0.15)
L_dev_wheat = jnp.log(0.25)
L_mid_wheat = jnp.log(0.4)
Kc_ini_wheat = jnp.log(0.4)
Kc_mid_wheat = jnp.log(1.15)
Kc_end_wheat = jnp.log(0.3)
Kmin_wheat = jnp.log(0.4)
Kmax_wheat = jnp.log(1.15)
c_lai_wheat = jnp.log(0.7)

# Other land cateogories
Kmin_cropland_other = jnp.log(1.0)
Kmax_cropland_other = jnp.log(1.0)
c_lai_cropland_other = jnp.log(0.7)

Kmin_evergreen_needleleaf = jnp.log(1.0)
Kmax_evergreen_needleleaf = jnp.log(1.0)
c_lai_evergreen_needleleaf = jnp.log(0.7)

Kmin_evergreen_broadleaf = jnp.log(1.0)
Kmax_evergreen_broadleaf = jnp.log(1.0)
c_lai_evergreen_broadleaf = jnp.log(0.7)

Kmin_deciduous_needleleaf = jnp.log(1.0)
Kmax_deciduous_needleleaf = jnp.log(1.0)
c_lai_deciduous_needleleaf = jnp.log(0.7)

Kmin_deciduous_broadleaf = jnp.log(1.0)
Kmax_deciduous_broadleaf = jnp.log(1.0)
c_lai_deciduous_broadleaf = jnp.log(0.7)

Kmin_mixed_forest = jnp.log(1.0)
Kmax_mixed_forest = jnp.log(1.0)
c_lai_mixed_forest = jnp.log(0.7)

Kmin_woodland = jnp.log(1.0)
Kmax_woodland = jnp.log(1.0)
c_lai_woodland = jnp.log(0.7)

Kmin_wooded_grassland = jnp.log(1.0)
Kmax_wooded_grassland = jnp.log(1.0)
c_lai_wooded_grassland = jnp.log(0.7)

Kmin_closed_shurbland = jnp.log(1.0)
Kmax_closed_shurbland = jnp.log(1.0)
c_lai_closed_shurbland = jnp.log(0.7)

Kmin_open_shrubland = jnp.log(1.0)
Kmax_open_shrubland = jnp.log(1.0)
c_lai_open_shrubland = jnp.log(0.7)

Kmin_grassland = jnp.log(1.0)
Kmax_grassland = jnp.log(1.0)
c_lai_grassland = jnp.log(0.7)

Kmin_barren = jnp.log(1.0)
Kmax_barren = jnp.log(1.0)
c_lai_barren = jnp.log(0.7)

Kmin_urban = jnp.log(1.0)
Kmax_urban = jnp.log(1.0)
c_lai_urban = jnp.log(0.7)


initial_params = jnp.array(
    [
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
    ]
)