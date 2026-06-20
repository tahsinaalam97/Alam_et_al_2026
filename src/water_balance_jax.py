from functools import partial

import jax
import jax.numpy as jnp


###############################
# STATE UPDATE FUNCTION
###############################
@jax.jit
def update_state(state, forcing, params):
    """
    Stateless update function for WBM soil moisture simulation.
    """

    # Retrieve previous iteration values
    Ws_prev, Wi, Sp = state

    # Retrieve parameter values
    (
        Ts,
        Tm,
        Wcap,
        wiltingp,
        alpha,
        betaHBV,
        phi,
    ) = params

    # Retrieve forcing
    tas, prcp, lai, Kpet, doy = forcing

    # PET function
    def calculate_potential_evapotranspiration(tas, doy, phi):
        """
        Inputs:
         - tas: daily mean temperature [C]
         - N:   day of year
         - phi: latitude [deg]
        Outputs:
         - daily potential evapotranspiration calculated via the Hamon method [mm]
        Notes: (e.g.) http://data.snap.uaf.edu/data/Base/AK_2km/PET/Hamon_PET_equations.pdf
        """

        # Calculate solar declination (delta)
        delta = -23.44 * jnp.cos(jnp.radians((360 / 365) * (doy + 10)))

        # Calculate fractional day length (Lambda)
        pi = 3.14159265359
        Lambda = (1 / pi) * jnp.arccos(
            -jnp.tan(jnp.radians(phi)) * jnp.tan(jnp.radians(delta))
        )

        # Calculate saturation vapor pressure
        tas_gt_zero = tas > 0
        Psat = (
            tas_gt_zero * (0.61078 * jnp.exp((17.26939 * tas) / (tas + 237.3)))
        ) + (
            (1 - tas_gt_zero)
            * (0.61078 * jnp.exp((21.87456 * tas) / (tas + 265.5)))
        )

        # Calculate saturation vapor density (rho_sat)
        rho_sat = (2.167 * Psat) / (tas + 273.15)

        # Calculate potential evapotranspiration (PET)
        PET = 330.2 * Lambda * rho_sat

        return PET

    ######################################
    # Begin simulation
    ######################################

    ################
    # Snowfall
    ################
    # Precipitation is assumed to be entirely snow/rain
    # if temperature is below/above threshold (Ts)
    is_snowfall = tas < Ts

    Ps = (is_snowfall * prcp) + ((1 - is_snowfall) * 0.0)
    Pa = (is_snowfall * 0.0) + ((1 - is_snowfall) * prcp)

    Sp = Sp + Ps

    ################
    # Snowmelt
    ################
    # Snowmelt is assumed to occur if temperature
    # is above a threshold (Tm), but is limited to
    # the volume of the snowpack
    is_snowmelt = tas > Tm
    Ms = is_snowmelt * (2.63 + 2.55 * tas + 0.0912 * tas * Pa) + (
        (1 - is_snowmelt) * 0.0
    )

    snowmelt_gt_snowpack = Ms > Sp
    Ms = (snowmelt_gt_snowpack * Sp) + ((1 - snowmelt_gt_snowpack) * Ms)
    Sp = (snowmelt_gt_snowpack * 0.0) + (
        (1 - snowmelt_gt_snowpack) * (Sp - Ms)
    )

    #########################
    # Canopy & throughfall
    #########################
    # Maximum canopy storage scales with LAI
    Wi_max = 0.25 * lai

    # Open water evaporation rate assumed to be PET
    Eow = calculate_potential_evapotranspiration(tas, doy, phi)
    # Canopy evaporation
    Ec = Eow * ((Wi / Wi_max) ** 0.6666667)

    # Throughfall is rainfall minus (canopy storage plus canopy evaporation)
    # Throughfall if zero if all rainfall goes to canopy
    canopy_full = Wi_max < Wi + Pa - Ec
    Pt = (canopy_full * (Pa - Ec - (Wi_max - Wi))) + ((1 - canopy_full) * 0.0)

    # Update canopy storage
    canopy_space = Wi + (Pa - Pt) - Ec <= Wi_max
    canopy_leftover = Wi + (Pa - Pt) - Ec > 0.0

    Wi = (
        ((canopy_space * canopy_leftover) * (Wi + (Pa - Pt) - Ec))
        + ((canopy_space * (1 - canopy_leftover)) * 0.0)
        + ((1 - canopy_space) * Wi_max)
    )

    ########################
    # Evapotranspiration
    ########################
    # Potential ET scales with (annual) coefficient
    PET = Kpet * calculate_potential_evapotranspiration(tas, doy, phi)

    # Calculate actual evapotranspiration
    # Actual ET is limited by water availability (throughfall + snowmelt)
    # otherwise the difference is scaled by drying function
    avail_water = (Pt + Ms) >= PET
    AET = (avail_water * PET) + (
        (1 - avail_water)
        * ((1 - jnp.exp(-alpha * Ws_prev / Wcap)) / (1 - jnp.exp(-alpha)))
        * (PET - Pt - Ms)
    )

    ################
    # Runoff
    ################
    # HBV direct groundwater recharge (can also be thought of as runoff)
    # scales nonlinearly with saturation in the active zone
    # Direct groundwater recharge (HBV)
    Id = (Pt + Ms) * (Ws_prev / Wcap) ** betaHBV

    ################
    # Soil moisture
    ################
    # Soil surplus is the leftover water after saturating soils
    # It gets partitioned to more runoff and groundwater recharge
    soil_full = Wcap < Ws_prev + (Pt + Ms - Id) - AET
    S = (soil_full * (Ws_prev + (Pt + Ms - Id) - AET - Wcap)) + (
        (1 - soil_full) * 0.0
    )

    # Update soil moisture
    Ws_new = Ws_prev + (Pt + Ms - Id) - AET - S

    # Soil moisture must be positive
    Ws_new = jnp.maximum(Ws_new, 0.0)

    # Soil moisture out (+ wilting point)
    Ws_out = Ws_new + wiltingp

    return (Ws_new, Wi, Sp), Ws_out


###############################
# RUN WBM FUNCTION
###############################
@jax.jit
def wbm_jax(
    tas,
    prcp,
    Kpet,
    Ws_init,
    Wi_init,
    Sp_init,
    lai,
    phi,
    params,
):
    """
    Run JAX version of WBM.
    """

    # Read parameters (must be correct order!)
    (
        Ts,
        Tm,
        awCap,
        wiltingp,
        rootDepth,
        alpha,
        betaHBV,
    ) = params

    # Soil moisture capacity
    Wcap = awCap * rootDepth
    wiltingp_scaled = wiltingp * rootDepth

    # Prepare passing to jax lax scan
    n_yrs = int(tas.shape[0] / 365)
    doy_rep = jnp.tile(jnp.arange(1, 366), n_yrs)
    Kpet_rep = jnp.tile(Kpet, n_yrs)
    lai_rep = jnp.tile(lai, n_yrs)

    scan_forcing = jnp.stack([tas, prcp, lai_rep, Kpet_rep, doy_rep], axis=1)
    scan_forcing = scan_forcing[1:,]  # skip first day since provided by init

    scan_params = (
        Ts,
        Tm,
        Wcap,
        wiltingp_scaled,
        alpha,
        betaHBV,
        phi,
    )

    # Update function
    update_fn = partial(update_state, params=scan_params)

    # Initial conditions
    init = (jnp.maximum(Ws_init - wiltingp_scaled, 0.0), Wi_init, Sp_init)

    # Run it
    outs, Ws_out = jax.lax.scan(update_fn, init, scan_forcing)

    return jnp.insert(
        Ws_out, 0, Ws_init
    )  # return with initial condition included


###########################
# Kc timeseries function
###########################
@jax.jit
def construct_Kpet_crop(
    GS_start,
    GS_end,
    L_ini,
    L_dev,
    L_mid,
    L_late,
    Kc_ini,
    Kc_mid,
    Kc_end,
    Kmin,
    Kmax,
    c_lai,
    lai,
):
    # Out
    Kpet_out = jnp.zeros(365)

    # Day of year
    doy = jnp.arange(365) + 1.0

    # Get days from relative length
    GS_length = GS_end - GS_start
    doy_ini = L_ini * GS_length
    doy_dev = L_dev * GS_length
    doy_mid = L_mid * GS_length
    doy_late = L_late * GS_length

    # Loop through year
    pre_GS = doy < GS_start
    Kpet_out += pre_GS * (Kmin + (Kmax - Kmin) * (1 - jnp.exp(-0.7 * lai)))

    ini_period = (doy < (GS_start + doy_ini)) & (doy >= GS_start)
    Kpet_out += ini_period * Kc_ini

    dev_period = (doy < (GS_start + doy_ini + doy_dev)) & (
        doy >= (GS_start + doy_ini)
    )
    Kpet_out += dev_period * (
        Kc_ini + (Kc_mid - Kc_ini) * ((doy - (GS_start + doy_ini)) / doy_dev)
    )

    mid_period = (doy < (GS_start + doy_ini + doy_dev + doy_mid)) & (
        doy >= (GS_start + doy_ini + doy_dev)
    )
    Kpet_out += mid_period * Kc_mid

    down_period = (
        doy < (GS_start + doy_ini + doy_dev + doy_mid + doy_late)
    ) & (doy >= (GS_start + doy_ini + doy_dev + doy_mid))

    Kpet_out += down_period * (
        Kc_mid
        - (Kc_mid - Kc_end)
        * (doy - (GS_start + doy_ini + doy_dev + doy_mid))
        / doy_late
    )

    post_GS = doy >= (GS_start + doy_ini + doy_dev + doy_mid + doy_late)
    Kpet_out += post_GS * (Kmin + (Kmax - Kmin) * (1 - jnp.exp(-c_lai * lai)))

    return Kpet_out


@jax.jit
def construct_Kpet_gen(
    Kmin,
    Kmax,
    c_lai,
    lai,
):
    return Kmin + (Kmax - Kmin) * (1 - jnp.exp(-c_lai * lai))
