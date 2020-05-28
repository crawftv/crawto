def correlation_report(df):
    sns.heatmap(df[numeric_features].corr())


def target_distribution_report(self):
    if self.problem == "regression":
        print(sns.distplot(self.data[self.target]))
    elif self.problem == "classification":
        print(sns.countplot(self.data[self.target]))


def numeric_columns_distribution_report(self):
    self.distribution_r()


#         print(
#             sns.PairGrid(
#                 self.data, x_vars=self.numeric_features, y_vars=self.target
#             ).map(sns.distplot)
#         )


def distribution_r(self):
    display(
        pandas.DataFrame(
            [
                self.distribution_fit(self.data, i)
                for i in self.numeric_features + [self.target]
            ],
            index=self.numeric_features + [self.target],
        )
    )


def distribution_fit(self, data, numeric_features):
    """
    x is a column_name
    """
    shapiro_values = shapiro(data[numeric_features])
    test_indication = True if shapiro_values[1] > 0.05 else False

    distribution_types = ["norm", "expon", "logistic", "gumbel"]
    # anderson_values = anderson(automl.data[numeric_column], dist=i)

    return {
        "Shapiro-Wilks_Test_Statistic": shapiro_values[0],
        "Shapiro-Wilks_p_Value": shapiro_values[1],
        "Normal distribution ?": test_indication
        # "Anderson_Darling_Test_Statistic_Normal": anderson_values[0][0],
    }


def nan_report(self):
    display(
        pandas.DataFrame(
            round((self.data.isna().sum() / self.data.shape[0]) * 100, 2),
            columns=["Percent of data encoded NAN"],
        )
    )


def correlation_report(self, threshold=0.95):
    corr_matrix = self.data[[self.target] + self.numeric_features].corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    highly_correlated_features = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    sns.heatmap(corr_matrix)
    if len(highly_correlated_features) > 0:
        print(f"Highly Correlated features are {highly_correlated_features}")
    else:
        print("No Features are correlated above the threshold")


def probability_plots(self):
    c = self.numeric_features + self.transformed_numeric_features
    c.sort()
    fig = plt.figure(figsize=(12, len(c) * 4))
    fig.tight_layout()
    chart_count = 1
    for i in range(1, (len(c) + 1), 1):
        fig.add_subplot(len(c), 2, chart_count)
        chart_count += 1
        probplot(self.data[c[i - 1]], plot=plt)
        plt.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.35
        )
        plt.title(c[i - 1] + " Probability Plot")
        fig.add_subplot(len(c), 2, chart_count)
        chart_count += 1
        sns.distplot(self.data[c[i - 1]])
    plt.show()


def categorical_bar_plots(self):
    c = self.categorical_features
    c.sort()
    fig = plt.figure(figsize=(12, len(c) * 4))
    fig.tight_layout()
    chart_count = 1
    for i in range(1, len(c) + 1):
        fig.add_subplot(len(c), 2, chart_count)
        sns.barplot(x=c[i - 1], y=self.target, data=self.data)
        chart_count += 1
        fig.add_subplot(len(c), 2, chart_count)
        sns.countplot(x=c[i - 1], data=self.data)
        chart_count += 1


def na_report(dataframe):
    print("NA's in the DataFrame")
    print(dataframe.isna().sum())


def skew_report(dataframe, threshold=5):
    highly_skewed = [
        i[0]
        for i in zip(dataframe.columns.values, abs(dataframe.skew(numeric_only=True)))
        if i[1] > threshold
    ]
    print(
        "There are %d highly skewed data columns. Please check them for miscoded na's"
        % len(highly_skewed)
    )
    print(highly_skewed)


def tsne_viz(self):
    t = TSNE()
    ta = t.fit_transform(self.train_transformed_data)
    d = pd.DataFrame(
        np.concatenate((ta, self.train_hbos_column.T.reshape(-1, 1)), axis=1),
        columns=["X", "Y", "Outlier"],
    )
    in_df = d[d["Outlier"] == 0]
    out_df = d[d["Outlier"] == 1]
    s = ScatterChart()
    s.add_DataSet("Outliers", out_df.X, out_df.Y)
    s.add_DataSet("Inliers", in_df.X, in_df.Y)
    p = Plot()
    p.add_column(s)
    return p.display


@task
def fit_svd(df):
    svd = TruncatedSVD()
    svd.fit(df)
    return svd


@task
def svd_transform(svd, df, name, tiny_db):
    data = svd.transform(df).T
    x = [float(ii) for ii in data[0]]
    y = [float(ii) for ii in data[1]]
    tiny_db.insert({"chunk": f"svd-{name}", "x": x, "y": y})
    return svd.transform(df)


@task
def spectral_clustering(df):
    s = SpectralClustering()
    s.fit()
