import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from .adjustments import split_discrete_continuous

if __name__ == "__main__":
    df = pd.read_csv("./data/avery/GSE37199/clinical.csv", index_col=False)
    discrete, continuous = split_discrete_continuous(df)
    continuous_norm = (continuous - continuous.min()) / (continuous.max() - continuous.min())
    for n_features in [10, 100, 1000, 10000]:
        first_n = continuous[continuous.columns[:n_features]]
        random_n = continuous[continuous.sample(n_features, axis="columns").columns]
        most_varying = continuous[continuous_norm.var(axis="rows").sort_values(ascending=False).head(n_features).index]
        k_best_centre = continuous.iloc[:, SelectKBest(f_classif, k=n_features).fit(continuous_norm, discrete['centre']).get_support()]
        k_best_plate = continuous.iloc[:, SelectKBest(f_classif, k=n_features).fit(continuous_norm, discrete['plate']).get_support()]
        discrete.merge(first_n, left_index=True, right_index=True).to_csv('./data/feature_selection/first_{:05d}.csv'.format(n_features), index=False)
        discrete.merge(random_n, left_index=True, right_index=True).to_csv('./data/feature_selection/random_{:05d}.csv'.format(n_features), index=False)
        discrete.merge(most_varying, left_index=True, right_index=True).to_csv('./data/feature_selection/variance_{:05d}.csv'.format(n_features), index=False)
        discrete.merge(k_best_centre, left_index=True, right_index=True).to_csv('./data/feature_selection/k_best_centre_{:05d}.csv'.format(n_features), index=False)
        discrete.merge(k_best_plate, left_index=True, right_index=True).to_csv('./data/feature_selection/k_best_plate_{:05d}.csv'.format(n_features), index=False)
    rf_centre = continuous.iloc[:, SelectFromModel(RandomForestClassifier().fit(continuous, discrete['centre']), prefit=True).get_support()]
    rf_plate = continuous.iloc[:, SelectFromModel(RandomForestClassifier().fit(continuous, discrete['plate']), prefit=True).get_support()]
    discrete.merge(rf_centre, left_index=True, right_index=True).to_csv('./data/feature_selection/randomforests_centre.csv', index=False)
    discrete.merge(rf_plate, left_index=True, right_index=True).to_csv('./data/feature_selection/randomforests_plate.csv', index=False)
