import pandas as pd
import numpy as np

from splink.duckdb.duckdb_linker import DuckDBLinker
from splink.charts import save_offline_chart


pd.options.display.max_rows = 1000
df = pd.read_parquet("./historical_figures_5k.parquet")


# Initialise the linker, passing in the input dataset(s)
linker = DuckDBLinker(input_tables={"df": df})

c = linker.profile_columns(
    ["first_name", "postcode_fake", "substr(dob, 1,4)"], top_n=10, bottom_n=5
)

save_offline_chart(c, "./profile_columns.html", overwrite=True)


from splink.comparison_library import exact_match, levenshtein

settings = {
    "proportion_of_matches": 1e-5,
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        "l.first_name = r.first_name and l.surname = r.surname",
        "l.surname = r.surname and l.dob = r.dob",
        "l.first_name = r.first_name and l.dob = r.dob",
        "l.postcode_fake = r.postcode_fake and l.first_name = r.first_name",
    ],
    "comparisons": [
        levenshtein("first_name", 2, term_frequency_adjustments=True),
        levenshtein("surname", 2, term_frequency_adjustments=True),
        levenshtein("dob", 2, term_frequency_adjustments=True),
        levenshtein("postcode_fake", 2),
        exact_match("birth_place", term_frequency_adjustments=True),
        exact_match("occupation", term_frequency_adjustments=True),
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
    "max_iterations": 10,
    "em_convergence": 0.01,
}

linker.initialise_settings(settings)
linker.train_u_using_random_sampling(target_rows=4e6)

blocking_rule = "l.first_name = r.first_name and l.surname = r.surname"
training_session_names = linker.train_m_using_expectation_maximisation(blocking_rule)
c = training_session_names.match_weights_interactive_history_chart()

blocking_rule = "l.dob = r.dob"
training_session_dob = linker.train_m_using_expectation_maximisation(blocking_rule)
c = training_session_dob.match_weights_interactive_history_chart()

c = linker.settings_obj.match_weights_chart()

df_e = linker.predict().as_pandas_dataframe()

df_e.to_parquet("./predictions.parquet", index=False)