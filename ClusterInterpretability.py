import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil

from sklearn.utils.validation import check_is_fitted

class ClusterInterpretability:
    def __init__(self, model, X, feature_names, n_clusters, prefix = "", method = None, y = None):
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.n_clusters = n_clusters
        self.prefix = prefix
        self.method = method if method is not None else "shap"
        self.y = y

        if self.method == "shap" :
            self.explainer = shap.KernelExplainer(self.model.predict, self.X)
        else:
            raise NotImplementedError("Method not implemented")
        self.explanations_by_cluster_dict = self.explanations_by_cluster()
        
    def explain(self, X = None):
        X = self.X if X is None else X
        if self.method == "shap":
            return self.explainer.shap_values(X)
        else:
            raise NotImplementedError("Method not implemented")
        
    def explanations_by_cluster(self):
        explanations = self.explain()
        clusters = self.model.predict(self.X)

        explanations_by_cluster = {}
        for cluster in range(self.n_clusters):
            cluster_mask = clusters == cluster
            explanations_by_cluster[cluster] = explanations[cluster_mask]
        
        return explanations_by_cluster

    def plot_explanations(self, cluster, res_dir = None):
        if self.method == "shap":
            explanations = self.explanations_by_cluster_dict[cluster]
            X_cluster = self.X[self.model.predict(self.X) == cluster]
            shap.summary_plot(explanations, X_cluster, feature_names=self.feature_names, show=False)
            # save plot
            if res_dir is not None:
                plt.savefig(res_dir / (self.prefix + "_cluster_" + str(cluster) + ".png"))
            plt.show()
        else:
            raise NotImplementedError("Method not implemented")
        
    def global_explanations(self, cluster, op = np.mean):
        explanations = self.explanations_by_cluster_dict[cluster]
        # calculate mean of absolute values per feature
        abs_mean = op(np.abs(explanations), axis = 0)
        # sort features by mean of absolute values
        sorted_features = np.argsort(abs_mean)
        # return feature_names sorted by mean of absolute values
        return self.feature_names[sorted_features], abs_mean[sorted_features]
    
    def plot_global_explanations(self, op = np.mean, nrows = 1, figsize = (20, 5), res_dir = None):
        ncols = ceil(self.n_clusters / nrows)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        ax = ax.flatten()
        for k, v in self.explanations_by_cluster_dict.items():
            feature_names_ordered, abs_mean = self.global_explanations(k, op = op)
            ax[k].barh(feature_names_ordered, abs_mean)
            ax[k].set_title("Cluster {}".format(k+1))
        plt.tight_layout()
        # save plot
        if res_dir is not None:
            if op == np.mean:
                plt.savefig(res_dir / (self.prefix + "_global_explanations_mean.png"))
            elif op == np.median:
                plt.savefig(res_dir / (self.prefix + "_global_explanations_median.png"))
            else:
                raise NotImplementedError("Operation not implemented")
        plt.show() 

    def plot_global_explanations_grouped(self, op = np.mean, figsize = (20, 5), res_dir = None):

        global_explanations_df = pd.DataFrame()
        for k, v in self.explanations_by_cluster_dict.items():
            feature_names_ordered, abs_mean = self.global_explanations(k, op = op)
            cluster_dict = {}
            for i, feature in enumerate(feature_names_ordered):
                cluster_dict[feature] = abs_mean[i]
            global_explanations_df = global_explanations_df.append(cluster_dict, ignore_index=True)

        global_explanations_df.index = global_explanations_df.index + 1
        global_explanations_df.index.name = "Cluster"
        global_explanations_df = global_explanations_df.transpose()
        
        # plot
        fig = global_explanations_df.plot.barh(figsize=figsize).get_figure()

        # save plot
        if res_dir is not None:
            if op == np.mean:
                fig.savefig(res_dir / (self.prefix + "_global_explanations_grouped_mean.png"))
            elif op == np.median:
                fig.savefig(res_dir / (self.prefix + "_global_explanations_grouped_median.png"))
            else:
                raise NotImplementedError("Operation not implemented")
        fig.show() 
