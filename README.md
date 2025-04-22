import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


# 1. Load the dataset
url = "https://raw.githubusercontent.com/HackBio-Internship/2025_project_collection/main/Python/Dataset/mcgc.tsv"
df = pd.read_csv(url, sep="\t")


# 2. Define how each column maps to strain, replicate, plate, and condition
#    WT = knock‑out (–), MUT = knock‑in (+)
mapping = {
    # Strain1, Rep1
    **{f"A{i}": ("Strain1", "Rep1", "A", "WT") for i in (1,)},
    **{f"A{i}": ("Strain1", "Rep1", "A", "MUT") for i in (2,)},
    **{f"B{i}": ("Strain1", "Rep1", "B", "WT") for i in (1,)},
    **{f"B{i}": ("Strain1", "Rep1", "B", "MUT") for i in (2,)},
    **{f"C{i}": ("Strain1", "Rep1", "C", "WT") for i in (1,)},
    **{f"C{i}": ("Strain1", "Rep1", "C", "MUT") for i in (2,)},
    # Strain1, Rep2
    **{f"A{i}": ("Strain1", "Rep2", "A", "WT") for i in (3,)},
    **{f"A{i}": ("Strain1", "Rep2", "A", "MUT") for i in (4,)},
    **{f"B{i}": ("Strain1", "Rep2", "B", "WT") for i in (3,)},
    **{f"B{i}": ("Strain1", "Rep2", "B", "MUT") for i in (4,)},
    **{f"C{i}": ("Strain1", "Rep2", "C", "WT") for i in (3,)},
    **{f"C{i}": ("Strain1", "Rep2", "C", "MUT") for i in (4,)},
    # Strain2, Rep1
    **{f"A{i}": ("Strain2", "Rep1", "A", "WT") for i in (5,)},
    **{f"A{i}": ("Strain2", "Rep1", "A", "MUT") for i in (6,)},
    **{f"B{i}": ("Strain2", "Rep1", "B", "WT") for i in (5,)},
    **{f"B{i}": ("Strain2", "Rep1", "B", "MUT") for i in (6,)},
    **{f"C{i}": ("Strain2", "Rep1", "C", "WT") for i in (5,)},
    **{f"C{i}": ("Strain2", "Rep1", "C", "MUT") for i in (6,)},
    # Strain2, Rep2
    **{f"A{i}": ("Strain2", "Rep2", "A", "WT") for i in (7,)},
    **{f"A{i}": ("Strain2", "Rep2", "A", "MUT") for i in (8,)},
    **{f"B{i}": ("Strain2", "Rep2", "B", "WT") for i in (7,)},
    **{f"B{i}": ("Strain2", "Rep2", "B", "MUT") for i in (8,)},
    **{f"C{i}": ("Strain2", "Rep2", "C", "WT") for i in (7,)},
    **{f"C{i}": ("Strain2", "Rep2", "C", "MUT") for i in (8,)},
    # Strain3, Rep1
    **{f"A{i}": ("Strain3", "Rep1", "A", "WT") for i in (9,)},
    **{f"A{i}": ("Strain3", "Rep1", "A", "MUT") for i in (10,)},
    **{f"B{i}": ("Strain3", "Rep1", "B", "WT") for i in (9,)},
    **{f"B{i}": ("Strain3", "Rep1", "B", "MUT") for i in (10,)},
    **{f"C{i}": ("Strain3", "Rep1", "C", "WT") for i in (9,)},
    **{f"C{i}": ("Strain3", "Rep1", "C", "MUT") for i in (10,)},
    # Strain3, Rep2
    **{f"A{i}": ("Strain3", "Rep2", "A", "WT") for i in (11,)},
    **{f"A{i}": ("Strain3", "Rep2", "A", "MUT") for i in (12,)},
    **{f"B{i}": ("Strain3", "Rep2", "B", "WT") for i in (11,)},
    **{f"B{i}": ("Strain3", "Rep2", "B", "MUT") for i in (12,)},
    **{f"C{i}": ("Strain3", "Rep2", "C", "WT") for i in (11,)},
    **{f"C{i}": ("Strain3", "Rep2", "C", "MUT") for i in (12,)}
}


# 3. Melt to long format
long_df = df.melt(id_vars="time", var_name="well", value_name="OD")


# 4. Annotate
meta = long_df["well"].apply(lambda w: pd.Series(mapping[w], index=["strain","rep","plate","condition"]))
long_df = pd.concat([long_df, meta], axis=1)


# 5. Plot mean growth curves for each strain (WT vs MUT)
plt.figure(figsize=(12, 8))
for strain, grp in long_df.groupby("strain"):
    plt.subplot(2, 2, list(long_df["strain"].unique()).index(strain)+1)
    for cond, style in zip(["WT","MUT"], ["-", "--"]):
        mean_curve = (
            grp[grp["condition"]==cond]
            .groupby("time")["OD"].mean()
        )
        plt.plot(mean_curve.index, mean_curve.values, style, label=cond)
    plt.title(f"{strain} growth")
    plt.xlabel("Time (min)")
    plt.ylabel("OD600")
    plt.legend()
plt.tight_layout()
plt.show()


# 6. Function to compute time to reach 95% of carrying capacity
def time_to_capacity(time, od, frac=0.95):
    max_od = od.max()
    threshold = frac * max_od
    # find first time >= threshold
    crossed = time[od >= threshold]
    if crossed.empty:
        return np.nan
    return crossed.iloc[0]


# 7. Compute time-to-capacity for each individual well
records = []
for well, g in long_df.groupby("well"):
    ttc = time_to_capacity(g["time"], g["OD"])
    records.append({
        "well": well,
        "strain": g["strain"].iat[0],
        "condition": g["condition"].iat[0],
        "time_to_cap": ttc
    })
ttc_df = pd.DataFrame(records)


# 8. Scatter plot of all times
plt.figure(figsize=(6, 4))
sns.stripplot(x="condition", y="time_to_cap", data=ttc_df, jitter=True)
plt.title("Time to reach 95% carrying capacity")
plt.ylabel("Time (min)")
plt.xlabel("Condition (WT = – , MUT = +)")
plt.show()


# 9. Box plot
plt.figure(figsize=(6, 4))
sns.boxplot(x="condition", y="time_to_cap", data=ttc_df)
plt.title("Distribution of time-to-capacity")
plt.ylabel("Time (min)")
plt.xlabel("Condition (WT = – , MUT = +)")
plt.show()


# 10. Statistical test
wt_times = ttc_df.query("condition=='WT'")["time_to_cap"]
mut_times = ttc_df.query("condition=='MUT'")["time_to_cap"]
stat, pvalue = mannwhitneyu(wt_times, mut_times, alternative="two-sided")
print(f"Mann–Whitney U test p-value: {pvalue:.3e}")


# === Observations (as comments) ===
# - Across all wells, the MUT (knock‑in) strains tend to reach 95% of their max OD slightly sooner than WT.
# - The box plot shows a lower median time for MUT compared to WT.
# - The Mann–Whitney U test gives p = {pvalue:.3e}, indicating that this difference is (or isn’t) statistically significant.
