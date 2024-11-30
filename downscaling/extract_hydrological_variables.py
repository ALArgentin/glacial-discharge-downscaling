import hydrobricks as hb
import numpy as np
import pandas as pd


def select_months_in_year(df, year, months=[6,7,8,9]):
    """
    Filters a dataframe to include only rows corresponding to a specific
    year and a set of months.

    @param df (dataframe)
        The input dataframe, assumed to have a datetime index.
    @param year (int)
        The year to filter the dataframe by.
    @param months (list of int)
        A list of integers representing the months to keep
        (default is [6, 7, 8, 9]).

    @return (dataframe) A dataframe containing data from the specified
        year and months.
    """
    df = df[df.index.year == year]
    df = df[df.index.month.isin(months)]
    return df

def select_months(df, months):
    """
    Filters a dataframe to include only rows corresponding to a specific
    set of months.

    @param df (dataframe)
        The input dataframe, assumed to have a datetime index.
    @param months (list of int)
        A list of integers representing the months to keep
        (default is [6, 7, 8, 9]).

    @return (dataframe) A dataframe containing data from the specified
        months.
    """
    df = df[df.index.month.isin(months)]
    return df

def extract_snow_water_eq(results_file, component):
    """
    Extract the snow water equivalent of distributed labels of Hydrobricks.

    @param results_file (str)
        The netcdf results file of Hydrobricks.
    @param component (str)
        The name of the distributed label component to retrieve.
        https://hydrobricks.readthedocs.io/en/latest/doc/basics.html#adding-data-related-parameters

    @return (dataframe) The requested component.
    """

    # Load the netcdf file
    results = hb.Results(results_file)

    # List the hydro units components available
    i_component = results.labels_distributed.index(component)
    df = results.results.hydro_units_values[i_component].to_dataframe()
    component_df = df.unstack(level='hydro_units')
    component_df.columns = component_df.columns=[multicols[-1] for multicols in component_df.columns]

    return component_df

def extract_snow_water_eq_agg(results_file, component):
    """
    Extract the snow water equivalent of aggregated labels of Hydrobricks.

    @param results_file (str)
        The netcdf results file of Hydrobricks.
    @param component (str)
        The name of the aggregated label component to retrieve.
        https://hydrobricks.readthedocs.io/en/latest/doc/basics.html#adding-data-related-parameters

    @return (dataframe) The requested component.
    """

    # Load the netcdf file
    results = hb.Results(results_file)
    # List the hydro units components available
    i_component = results.labels_aggregated.index(component)
    df = results.results.hydro_units_values[i_component].to_dataframe()
    component_df = df.unstack(level='hydro_units')
    component_df.columns = component_df.columns=[multicols[-1] for multicols in component_df.columns]

    return component_df

def extract_hydro_unit_characteristics(results_file):
    """
    Extract the characteristics of the hydrological units of the
    simulation run with Hydrobricks.

    @param results_file (str)
        The netcdf results file of Hydrobricks.

    @return (dataframe, dataframe) The ground areas per hydrological unit,
        and the glaciated areas per hydrological unit.
    """

    # Load the netcdf file
    results = hb.Results(results_file)

    # List the hydro units components available
    results.list_hydro_units_components()
    results.list_sub_basin_components()
    hy_areas = results.results.hydro_units_areas.to_dataframe()
    lo_areas = results.results.land_cover_fractions.to_dataframe()

    # Reformat the dataframes to have the hydro units as columns
    hy_areas = hy_areas.T
    lo_areas = lo_areas.unstack(level='hydro_units')
    lo_areas.columns = lo_areas.columns=[multicols[-1] for multicols in lo_areas.columns]

    # Select only the ground/glacial fraction and compute the hydro unit ground/glaciated areas
    hy_ground_fraction = lo_areas.xs(0, level=0)
    hy_ground_areas = hy_ground_fraction.mul(hy_areas.values, axis=1)
    hy_glacier_fraction = lo_areas.xs(1, level=0)
    hy_glacier_areas = hy_glacier_fraction.mul(hy_areas.values, axis=1)

    return hy_ground_areas, hy_glacier_areas

def process_content_snow(results_file, component, weight_area, div_area):
    """
    Weight and sum state variables such as snow content that are not
    yet weighted.

    Documentation Hydrobricks:
    # State variables (mm) such as content or snow elements represent
    # the water stored in the respective reservoirs. This value is not
    # weighted and cannot be summed over the catchment, but must be
    # weighted by the land cover fraction and the relative hydro
    # unit area first.
    Same for melt:output fluxes.

    @param results_file (str)
        The netcdf results file of Hydrobricks.
    @param component (str)
        The name of the component to retrieve.
        https://hydrobricks.readthedocs.io/en/latest/doc/basics.html#adding-data-related-parameters
    @param weight_area (array of float)
        The ground/glacier areas per hydrological unit, depending on the type of component.
    @param div_area (float)
        The total ground/glacier area.

    @return (... , ....)
    """

    # Retrieve the hydro unit parameters
    comp = extract_snow_water_eq(results_file, component)
    # Weight the parameters by the areas
    comp_w = comp * weight_area
    # Divide by the area of interest
    comp = np.sum(comp_w, axis=1) / div_area
    return comp, comp_w

def extract_meteorological_data(forcing_file, hydro_units_file, with_debris, melt_model):

    if with_debris:
        land_cover_names = ['ground', 'glacier_ice', 'glacier_debris']
        land_cover_types = ['ground', 'glacier', 'glacier']
    else:
        land_cover_names = ['ground', 'glacier']
        land_cover_types = ['ground', 'glacier']

    if melt_model == 'temperature_index' or melt_model == 'degree_day':
        other_columns = None
    elif melt_model == 'degree_day_aspect':
        other_columns = {'slope': 'slope', 'aspect_class': 'aspect_class'}

    # Hydro units
    hyd_units = hb.HydroUnits(land_cover_types, land_cover_names)
    hyd_units.load_from_csv(hydro_units_file, column_area='area', column_elevation='elevation',
                                other_columns=other_columns)

    # Finally, initialize the HydroUnits cover with the first cover values of the
    # BehaviourLandCoverChange object (no need to do it for the ground land cover type).
    #if with_debris:
    #    hyd_units.initialize_from_land_cover_change('glacier_ice', changes_df[0])
    #    hyd_units.initialize_from_land_cover_change('glacier_debris', changes_df[1])
    #else:
    #    hyd_units.initialize_from_land_cover_change('glacier', changes_df[0])

    forcing = hb.Forcing(hyd_units)
    forcing.load_from(forcing_file)

    precipitations = pd.DataFrame(forcing.data2D.data[0], index=forcing.data2D.time)
    temperatures = pd.DataFrame(forcing.data2D.data[1], index=forcing.data2D.time)
    if len(forcing.data2D.data) == 4:
        radiations = pd.DataFrame(forcing.data2D.data[2], index=forcing.data2D.time)
        pet = pd.DataFrame(forcing.data2D.data[3], index=forcing.data2D.time)
    else:
        radiations = np.nan
        pet = pd.DataFrame(forcing.data2D.data[2], index=forcing.data2D.time)
    hydro_units = forcing.hydro_units

    return hydro_units, precipitations, temperatures, radiations, pet

def get_meteorological_hydrological_data(forcing_file, results_file, hydro_units_file, months, melt_model, with_debris):

    hydro_units, precipitations, temperatures, radiations, pet = extract_meteorological_data(forcing_file, hydro_units_file,
                                                                                             with_debris=with_debris, melt_model=melt_model)
    elevations = hydro_units['elevation', 'm']
    areas = hydro_units['area', 'm2']
    total_area = np.sum(areas)

    # Get ground, glacier and total areas for each hydro unit
    hy_ground_areas, hy_glacier_areas = extract_hydro_unit_characteristics(results_file)
    total_ground_area = np.sum(hy_ground_areas, axis=1)
    total_glacier_area = np.sum(hy_glacier_areas, axis=1)

    # Process all :content components
    comp1_m, _ = process_content_snow(results_file, "glacier:outflow_rain_snowmelt:output", hy_glacier_areas.values, total_glacier_area)
    comp2_m, comp2_w = process_content_snow(results_file, "glacier:melt:output",            hy_glacier_areas.values, total_glacier_area)
    comp3_m, comp3_w = process_content_snow(results_file, "ground_snowpack:melt:output",    hy_ground_areas.values,  total_ground_area)
    comp4_m, comp4_w = process_content_snow(results_file, "glacier_snowpack:melt:output",   hy_glacier_areas.values, total_glacier_area)
    comp5_m, comp5_w = process_content_snow(results_file, "ground_snowpack:snow",     hy_ground_areas.values,  total_ground_area)
    comp6_m, comp6_w = process_content_snow(results_file, "glacier_snowpack:snow",    hy_glacier_areas.values, total_glacier_area)
    comp7_m = extract_snow_water_eq_agg(results_file, "glacier_area_icemelt_storage:content")
    comp8_m = extract_snow_water_eq_agg(results_file, "glacier_area_icemelt_storage:outflow:output")

    precip_m = precipitations * areas
    precip_m = np.sum(precip_m, axis=1) / total_area
    precip_m = precip_m.values
    temper_m = temperatures * areas
    temper_m = np.sum(temper_m, axis=1) / total_area
    temper_m = temper_m.values
   # if not np.isnan(radiations):
    radiat_m = radiations * areas
    radiat_m = np.sum(radiat_m, axis=1) / total_area
    radiat_m = radiat_m.values
    gsnow_m = np.sum(comp6_w, axis=1) / total_glacier_area
    gsnow_m = gsnow_m.values
    # Add the parameters that have different land covers
    # Snow melt
    s_melt_w = comp3_w + comp4_w
    s_melt_m = np.sum(s_melt_w, axis=1) / total_area
    s_melt_m = s_melt_m.values
    i_melt_m = np.sum(comp2_w, axis=1) / total_area
    i_melt_m = i_melt_m.values
    snow_w = comp5_w + comp6_w
    snow_m = np.sum(snow_w, axis=1) / total_area
    snow_m = snow_m.values

    glacier_area_percentage = (total_glacier_area / (total_ground_area + total_glacier_area)).values

    meteo_df = pd.DataFrame()
    meteo_df["Date"] = precipitations.index
    meteo_df["Glacier area percentage"] = glacier_area_percentage
    meteo_df["Precipitation"] = precip_m
    meteo_df["Temperature"] = temper_m
  #  if not np.isnan(radiations):
    meteo_df["Radiation"] = radiat_m
    meteo_df["Snow melt"] = s_melt_m
    meteo_df["Ice melt"] = i_melt_m
    meteo_df["All snow"] = snow_m
    meteo_df["Glacier snow"] = gsnow_m
    #meteo_df["glacier_area_icemelt_storage"] = comp7_m
    #meteo_df["glacier_area_icemelt_storage outflow"] = comp8_m
    meteo_df = meteo_df.set_index(precipitations.index)
    #meteo_df = select_months(meteo_df, months)
    #melt_m = melt_m.loc[pd.to_datetime(date1):pd.to_datetime(date2)]

    return meteo_df