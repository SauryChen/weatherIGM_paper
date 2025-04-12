from enum import IntEnum

class vars_table(IntEnum):
    temperature_2m = 0
    mean_sea_level_pressure = 1
    u_component_of_wind_10m = 2
    v_component_of_wind_10m = 3
    geopotential_50 = 4
    geopotential_250 = 5
    geopotential_500 = 6
    geopotential_600 = 7
    geopotential_700 = 8
    geopotential_850 = 9
    geopotential_925 = 10
    u_component_of_wind_50 = 11
    u_component_of_wind_250 = 12
    u_component_of_wind_500 = 13
    u_component_of_wind_600 = 14
    u_component_of_wind_700 = 15
    u_component_of_wind_850 = 16
    u_component_of_wind_925 = 17
    v_component_of_wind_50 = 18
    v_component_of_wind_250 = 19
    v_component_of_wind_500 = 20
    v_component_of_wind_600 = 21
    v_component_of_wind_700 = 22
    v_component_of_wind_850 = 23
    v_component_of_wind_925 = 24
    temperature_50 = 25
    temperature_250 = 26
    temperature_500 = 27
    temperature_600 = 28
    temperature_700 = 29
    temperature_850 = 30
    temperature_925 = 31
    specific_humidity_50 = 32
    specific_humidity_250 = 33
    specific_humidity_500 = 34
    specific_humidity_600 = 35
    specific_humidity_700 = 36
    specific_humidity_850 = 37
    specific_humidity_925 = 38

class cons_table(IntEnum):
    lat = 0
    lon = 1
    geopotential_at_surface = 2
    land_sea_mask = 3
    soil_type = 4



