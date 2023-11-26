import shap

from sklearn.utils.validation import check_is_fitted

class ClusterInterpretability:
    def __init__(self, model, X, feature_names, n_clusters, method = None, y = None):
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.n_clusters = n_clusters
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

    def plot_explanations(self, cluster):
        if self.method == "shap":
            explanations = self.explanations_by_cluster_dict[cluster]
            X_cluster = self.X[self.model.predict(self.X) == cluster]
            shap.summary_plot(explanations, X_cluster, feature_names=self.feature_names)
        else:
            raise NotImplementedError("Method not implemented")
