import pandas as pd
from scipy.stats import shapiro
import seaborn

class CrawtoML:
    """
    import pandas
    pandas.DataFrame
    """

    def __init__(self, data, target, features):
        self.data = data
        self.target = target
        self.features = features
        self.numeric_columns = self.numerics()

    def numerics(self):
        numerics = []
        for i in self.features:
            if self.data[i].dtypes in ["int64", "float64"]:
                numerics.append(i)
        self.numeric_columns = numerics
        return self.numeric_columns

    def other_types(self):
        others = [i for i in self.features if i not in self.numeric_columns]
        self.other_types = others
        return self.other_types

    def __repr__(self):
        return "Target Column: %s \n \
        Feature columns: %s\n \
        Numeric Columns: %s"(
            self.target, self.features, self.numeric_columns
        )

    def correlation_report(self):
        seaborn.heatmap(self.data[self.numeric_columns].corr())

    def distribution_report(self):
        self.distribution_r()
        print(seaborn.distplot(self.data[self.target]))
        print(
            seaborn.PairGrid(self.data, x_vars=self.features, y_vars=self.target).map(
                seaborn.scatterplot
            )
        )

    def distribution_r(self):
        print(
            pd.DataFrame(
                [
                    self.distribution_fit(self.data, i)
                    for i in self.numeric_columns + [self.target]
                ],
                index=self.numeric_columns + [self.target],
            )
        )

    def distribution_fit(self, data, numeric_column):

        """
        x is a column_name
        """
        shapiro_values = shapiro(data[numeric_column])
        test_indication = True if shapiro_values[1] > 0.05 else False

        distribution_types = ["norm", "expon", "logistic", "gumbel"]
        # anderson_values = anderson(automl.data[numeric_column], dist=i)

        return {
            "Shapiro-Wilks_Test_Statistic": shapiro_values[0],
            "Shapiro-Wilks_p_Value": shapiro_values[1],
            "Normal distribution ?": test_indication
            # "Anderson_Darling_Test_Statistic_Normal": anderson_values[0][0],
        }
