import numpy as np
import pandas as pd


ROAD_SEGMENT_BIN_SIZE = 100
GEO_GRID_SIZE = 10_000


def add_road_context_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["ROAD_IS_URBAN"] = (
        result["SUG_DEREH"]
        .map({
            1: True,
            2: True,
            3: False,
            4: False,
        })
        .astype("boolean")
    )

    result["ROAD_IS_INTERSECTION"] = (
        result["SUG_DEREH"]
        .map({
            1: True,
            2: False,
            3: True,
            4: False,
        })
        .astype("boolean")
    )

    return result


# Time features
def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["HOUR"] = (
        (result["SHAA"] - 1) // 4
    ).astype("Int64")

    result["TIME_OF_DAY"] = pd.cut(
        result["HOUR"],
        bins=[-1, 6, 11, 17, 21, 23],
        labels=[
            "Night / Early Morning",
            "Morning",
            "Afternoon",
            "Evening",
            "Late Evening",
        ],
    )

    result["IS_WEEKEND"] = (
        result["YOM_BASHAVUA"]
        .map({
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: True,
            7: True,
        })
        .astype("boolean")
    )

    return result


# Road segment
def add_road_segment(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    has_valid_location = (
        result["KVISH1"].notna()
        & result["KM"].notna()
        & result["KM"].ge(0)
    )

    km_bin = (
        np.floor(
            result["KM"] / ROAD_SEGMENT_BIN_SIZE
        )
        * ROAD_SEGMENT_BIN_SIZE
    )

    road_segment = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    road_segment.loc[has_valid_location] = (
        result.loc[has_valid_location, "KVISH1"]
        .astype("Int64")
        .astype(str)
        + "_"
        + km_bin.loc[has_valid_location]
        .astype("Int64")
        .astype(str)
    )

    result["ROAD_SEGMENT"] = road_segment

    return result

# Geographic grid
def add_geographic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    has_coordinates = (
        result["X"].notna()
        & result["Y"].notna()
    )

    x_bin = (
        np.floor(result["X"] / GEO_GRID_SIZE)
        * GEO_GRID_SIZE
    )

    y_bin = (
        np.floor(result["Y"] / GEO_GRID_SIZE)
        * GEO_GRID_SIZE
    )

    geo_grid = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    geo_grid.loc[has_coordinates] = (
        x_bin.loc[has_coordinates]
        .astype("Int64")
        .astype(str)
        + "_"
        + y_bin.loc[has_coordinates]
        .astype("Int64")
        .astype(str)
    )

    result["GEO_GRID"] = geo_grid

    return result


# Carriageway
def add_carriageway_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["ROAD_CARRIAGEWAY"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result["ROAD_CARRIAGEWAY_TYPE"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    # HAD_MASLUL
    result.loc[
        result["HAD_MASLUL"].isin([1, 2, 3, 4]),
        "ROAD_CARRIAGEWAY",
    ] = "Single carriageway"

    result.loc[
        result["HAD_MASLUL"].eq(1),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "SINGLE_1"  # One-way

    result.loc[
        result["HAD_MASLUL"].eq(2),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "SINGLE_2"  # Two-way with continuous dividing line

    result.loc[
        result["HAD_MASLUL"].eq(3),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "SINGLE_3"  # Two-way without continuous dividing line

    result.loc[
        result["HAD_MASLUL"].eq(4),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "SINGLE_4"  # Other

    result.loc[
        result["HAD_MASLUL"].eq(9),
        "ROAD_CARRIAGEWAY",
    ] = "Unknown"

    result.loc[
        result["HAD_MASLUL"].eq(9),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "Unknown"  # Number of carriageways unknown

    # RAV_MASLUL
    result.loc[
        result["RAV_MASLUL"].isin([1, 2, 3, 4, 5]),
        "ROAD_CARRIAGEWAY",
    ] = "Multiple carriageways"

    result.loc[
        result["RAV_MASLUL"].eq(1),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "MULTI_1"  # Median marked with paint

    result.loc[
        result["RAV_MASLUL"].eq(2),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "MULTI_2"  # Median with safety barrier

    result.loc[
        result["RAV_MASLUL"].eq(3),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "MULTI_3"  # Built median without safety barrier

    result.loc[
        result["RAV_MASLUL"].eq(4),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "MULTI_4"  # Unbuilt median

    result.loc[
        result["RAV_MASLUL"].eq(5),
        "ROAD_CARRIAGEWAY_TYPE",
    ] = "MULTI_5"  # Other

    return result


# Road defect
def add_road_defect_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["ROAD_DEFECT"] = (
        result["TKINUT"]
        .map({
            0: "Unknown",     # Unknown
            1: "No defect",   # No defect
            2: "Defect",      # Poor shoulders
            3: "Defect",      # Damaged road surface
            4: "Defect",      # Poor shoulders and damaged road surface
        })
        .astype("string")
    )

    return result


# Visibility / lighting
def add_visibility_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["VISIBILITY_OR_LIGHTING_ISSUE"] = (
        result["TEURA"]
        .map({
            1: "No issue",    # Normal daylight
            2: "Issue",       # Limited visibility due to weather
            3: "No issue",    # Night, lighting operational
            4: "Issue",       # Lighting exists but defective or not operating
            5: "Issue",       # Night, no lighting
            6: "Unknown",     # Night, unknown
            7: "Issue",       # Night, proper lighting with limited visibility
            8: "Issue",       # Night, defective lighting with limited visibility
            9: "Issue",       # Night, no lighting with limited visibility
            10: "Twilight",   # Twilight
            11: "Unknown",    # Day, unknown
        })
        .astype("string")
    )

    return result

# Road surface
def add_road_surface_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["ROAD_SURFACE_CONDITION"] = (
        result["PNE_KVISH"]
        .map({
            1: "Dry",             # Dry
            2: "Wet",             # Wet
            3: "Other condition", # Covered with fuel
            4: "Other condition", # Covered with mud
            5: "Other condition", # Sand or gravel on road
            6: "Other condition", # Other
            9: "Unknown",         # Unknown
        })
        .astype("string")
        )

    return result

# Pedestrian activity
def add_pedestrian_activity_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["PEDESTRIAN_ACTIVITY"] = (
        result["LO_HAZA"]
        .map({
            1: "Walking",          # Walking in direction of traffic
            2: "Walking",          # Walking against traffic
            3: "Playing",          # Playing on the road
            4: "Standing",         # Standing on the road
            5: "Roadside / median",# On a traffic island / median
            6: "Roadside / median",# On the shoulder / sidewalk
            7: "Other",            # Other
            9: "Unknown",          # Unknown
        })
        .astype("string")
    )

    return result

# Crosswalk usage
def add_crosswalk_usage_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["CROSSWALK_USAGE"] = (
        result["MEKOM_HAZIYA"]
        .map({
            0: "Not applicable",   # [Not applicable]
            1: "No crosswalk",     # Not at crosswalk, near intersection
            2: "No crosswalk",     # Not at crosswalk, not near intersection
            3: "Crosswalk",        # At crosswalk without traffic light
            4: "Crosswalk",        # At crosswalk with traffic light
            9: "Unknown",          # Crossing location unknown
        })
        .astype("string")
    )

    return result


# Locality features
def add_locality_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["LOCALITY_SIZE"] = (
        result["ZURAT_ISHUV"]
        .map({
            9: "Other / special",       # Urban refugee camp

            12: "500,000+",             # Jewish locality, 500,000+ residents
            13: "200,000-499,999",      # Jewish locality, 200,000-499,999 residents
            14: "100,000-199,999",      # Jewish locality, 100,000-199,999 residents
            15: "50,000-99,999",        # Jewish locality, 50,000-99,999 residents
            16: "20,000-49,999",        # Jewish locality, 20,000-49,999 residents
            17: "10,000-19,999",        # Jewish locality, 10,000-19,999 residents
            18: "5,000-9,999",          # Jewish locality, 5,000-9,999 residents
            19: "2,000-4,999",          # Jewish locality, 2,000-4,999 residents

            25: "50,000-99,999",        # Non-Jewish locality, 50,000-99,999 residents
            26: "20,000-49,999",        # Non-Jewish locality, 20,000-49,999 residents
            27: "10,000-19,999",        # Non-Jewish locality, 10,000-19,999 residents
            28: "5,000-9,999",          # Non-Jewish locality, 5,000-9,999 residents
            29: "2,000-4,999",          # Non-Jewish locality, 2,000-4,999 residents

            31: "Rural / special",       # Moshav / cooperative village
            32: "Rural / special",       # Collective moshav
            33: "Rural / special",       # Kibbutz
            34: "Rural / special",       # Jewish institutional locality
            35: "Rural / special",       # Other Jewish rural locality
            37: "Rural / special",       # Community locality

            44: "Rural / special",       # Non-Jewish institutional locality
            45: "Rural / special",       # Other non-Jewish rural locality
            46: "Rural / special",       # Bedouin tribe
            49: "Rural / special",       # Other rural locality

            51: "Other / special",       # Place
            52: "Other / special",       # Employment center
            53: "Other / special",       # Collective locality symbol
            59: "Other / special",       # Rural refugee camp

            99: "Not applicable",        # [Not applicable]
        })
        .astype("string")
    )

    result["LOCALITY_FORM"] = (
        result["ZURAT_ISHUV"]
        .map({
            9: "Urban",                  # Urban refugee camp

            12: "Urban",                 # Jewish locality, 500,000+ residents
            13: "Urban",                 # Jewish locality, 200,000-499,999 residents
            14: "Urban",                 # Jewish locality, 100,000-199,999 residents
            15: "Urban",                 # Jewish locality, 50,000-99,999 residents
            16: "Urban",                 # Jewish locality, 20,000-49,999 residents
            17: "Urban",                 # Jewish locality, 10,000-19,999 residents
            18: "Urban",                 # Jewish locality, 5,000-9,999 residents
            19: "Urban",                 # Jewish locality, 2,000-4,999 residents

            25: "Urban",                 # Non-Jewish locality, 50,000-99,999 residents
            26: "Urban",                 # Non-Jewish locality, 20,000-49,999 residents
            27: "Urban",                 # Non-Jewish locality, 10,000-19,999 residents
            28: "Urban",                 # Non-Jewish locality, 5,000-9,999 residents
            29: "Urban",                 # Non-Jewish locality, 2,000-4,999 residents

            31: "Rural",                 # Moshav / cooperative village
            32: "Rural",                 # Collective moshav
            33: "Rural",                 # Kibbutz
            34: "Rural",                 # Jewish institutional locality
            35: "Rural",                 # Other Jewish rural locality
            37: "Rural",                 # Community locality

            44: "Rural",                 # Non-Jewish institutional locality
            45: "Rural",                 # Other non-Jewish rural locality
            46: "Rural",                 # Bedouin tribe
            49: "Rural",                 # Other rural locality

            51: "Other / special",       # Place
            52: "Other / special",       # Employment center
            53: "Other / special",       # Collective locality symbol
            59: "Other / special",       # Rural refugee camp

            99: "Not applicable",        # [Not applicable]
        })
        .astype("string")
    )

    result["LOCALITY_SECTOR"] = (
        result["ZURAT_ISHUV"]
        .map({
            9: "Other / unspecified",    # Urban refugee camp

            12: "Jewish",                # Jewish locality, 500,000+ residents
            13: "Jewish",                # Jewish locality, 200,000-499,999 residents
            14: "Jewish",                # Jewish locality, 100,000-199,999 residents
            15: "Jewish",                # Jewish locality, 50,000-99,999 residents
            16: "Jewish",                # Jewish locality, 20,000-49,999 residents
            17: "Jewish",                # Jewish locality, 10,000-19,999 residents
            18: "Jewish",                # Jewish locality, 5,000-9,999 residents
            19: "Jewish",                # Jewish locality, 2,000-4,999 residents

            25: "Non-Jewish",            # Non-Jewish locality, 50,000-99,999 residents
            26: "Non-Jewish",            # Non-Jewish locality, 20,000-49,999 residents
            27: "Non-Jewish",            # Non-Jewish locality, 10,000-19,999 residents
            28: "Non-Jewish",            # Non-Jewish locality, 5,000-9,999 residents
            29: "Non-Jewish",            # Non-Jewish locality, 2,000-4,999 residents

            31: "Jewish",                # Moshav / cooperative village
            32: "Jewish",                # Collective moshav
            33: "Jewish",                # Kibbutz
            34: "Jewish",                # Jewish institutional locality
            35: "Jewish",                # Other Jewish rural locality
            37: "Jewish",                # Community locality

            44: "Non-Jewish",            # Non-Jewish institutional locality
            45: "Non-Jewish",            # Other non-Jewish rural locality
            46: "Non-Jewish",            # Bedouin tribe
            49: "Other / unspecified",   # Other rural locality

            51: "Other / unspecified",   # Place
            52: "Other / unspecified",   # Employment center
            53: "Other / unspecified",   # Collective locality symbol
            59: "Other / unspecified",   # Rural refugee camp

            99: "Not applicable",        # [Not applicable]
        })
        .astype("string")
    )

    return result

# build_features
def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result = add_road_context_features(result)
    result = add_time_features(result)
    result = add_road_segment(result)
    result = add_geographic_features(result)

    result = add_carriageway_features(result)
    result = add_road_defect_feature(result)
    result = add_visibility_feature(result)
    result = add_road_surface_feature(result)

    result = add_pedestrian_activity_feature(result)
    result = add_crosswalk_usage_feature(result)
    result = add_locality_features(result)

    return result