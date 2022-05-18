import pandas as pd
import random
from splink.comparison_library import exact_match, levenshtein

from splink.duckdb.duckdb_linker import DuckDBLinker
from splink.charts import save_offline_chart


pd.options.display.max_rows = 1000
df = pd.read_parquet("./synthetic_1m.parquet")


# Initialise the linker, passing in the input dataset(s)
linker = DuckDBLinker(df, connection=":temporary:")

c = linker.profile_columns(
    ["first_name", "postcode_fake", "substr(dob, 1,4)"], top_n=10, bottom_n=5
)

save_offline_chart(c.spec, "./profile_columns.html", overwrite=True)



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
linker.estimate_u_using_random_sampling(target_rows=4e7)

blocking_rule = "l.first_name = r.first_name and l.surname = r.surname"
training_session_names = linker.estimate_parameters_using_expectation_maximisation(
    blocking_rule
)
c = training_session_names.match_weights_interactive_history_chart()

blocking_rule = "l.dob = r.dob and l.postcode_fake = r.postcode_fake"
training_session_dob = linker.estimate_parameters_using_expectation_maximisation(
    blocking_rule
)
c = training_session_dob.match_weights_interactive_history_chart()

c = linker.match_weights_chart()

df_predict = linker.predict(threshold_match_probability=0.8)

sql = f"""
select count(*) as count
from {df_predict.physical_name}

"""

print(linker._con.execute(sql).fetch_df())


df_predict_pd = df_predict.as_pandas_dataframe()
df_predict_pd.to_parquet("./predictions.parquet", index=False)


df_clusters = linker.cluster_pairwise_predictions_at_threshold(df_predict, 0.1)
df_clusters.as_pandas_dataframe().to_csv("./clusters.csv", index=False)


linker.splink_comparison_viewer(df_predict, "./splink_comparison_viewer.html", True, 2)


sql = f"""
select cluster_id, count(*) as count
from {df_clusters.physical_name}
group by cluster_id
having count(*)> 1
"""

clusters_two_or_more = linker._con.execute(sql).fetch_df()


cluster_ids = random.choices(clusters_two_or_more["cluster_id"].unique(), k=10)


linker.cluster_studio(
    df_predict, df_clusters, cluster_ids, "./splink_cluster_studio.html", True
)
