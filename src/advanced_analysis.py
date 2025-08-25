"""
Advanced Asymmetry Analysis Module

This module provides sophisticated analytical methods for deeper insights into
brain asymmetry patterns, including dimensionality reduction, clustering,
connectivity analysis, and machine learning approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.manifold import TSNE, UMAP
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import warnings

warnings.filterwarnings("ignore")


class AdvancedAsymmetryAnalyzer:
    """Advanced analytical methods for asymmetry pattern discovery"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca_model = None
        self.clustering_model = None

    def dimensionality_reduction_analysis(
        self, asymmetry_data, methods=["PCA", "ICA", "UMAP"]
    ):
        """
        Perform multiple dimensionality reduction techniques to identify
        latent asymmetry patterns
        """
        results = {}

        # Prepare data matrix (patients x regions)
        patient_region_matrix = self._prepare_patient_region_matrix(asymmetry_data)
        scaled_data = self.scaler.fit_transform(patient_region_matrix)

        for method in methods:
            if method == "PCA":
                model = PCA(n_components=min(10, scaled_data.shape[1]))
                transformed = model.fit_transform(scaled_data)
                results["PCA"] = {
                    "transformed_data": transformed,
                    "explained_variance": model.explained_variance_ratio_,
                    "components": model.components_,
                    "model": model,
                }

            elif method == "ICA":
                model = FastICA(
                    n_components=min(5, scaled_data.shape[1]), random_state=42
                )
                transformed = model.fit_transform(scaled_data)
                results["ICA"] = {
                    "transformed_data": transformed,
                    "components": model.components_,
                    "model": model,
                }

            elif method == "UMAP":
                try:
                    model = UMAP(n_components=2, random_state=42)
                    transformed = model.fit_transform(scaled_data)
                    results["UMAP"] = {"transformed_data": transformed, "model": model}
                except ImportError:
                    print("UMAP not available. Install with: pip install umap-learn")

        return results, patient_region_matrix

    def hierarchical_asymmetry_clustering(self, asymmetry_data, linkage_method="ward"):
        """
        Perform hierarchical clustering on asymmetry patterns to identify
        patient subgroups with similar asymmetry profiles
        """
        # Prepare patient-region matrix
        patient_region_matrix = self._prepare_patient_region_matrix(asymmetry_data)
        scaled_data = self.scaler.fit_transform(patient_region_matrix)

        # Hierarchical clustering
        linkage_matrix = linkage(scaled_data, method=linkage_method)

        # Determine optimal number of clusters using silhouette analysis
        silhouette_scores = []
        cluster_range = range(2, min(8, len(scaled_data)))

        for n_clusters in cluster_range:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                score = silhouette_score(scaled_data, cluster_labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)

        optimal_clusters = (
            cluster_range[np.argmax(silhouette_scores)] if silhouette_scores else 3
        )
        cluster_labels = fcluster(
            linkage_matrix, optimal_clusters, criterion="maxclust"
        )

        return {
            "linkage_matrix": linkage_matrix,
            "cluster_labels": cluster_labels,
            "optimal_clusters": optimal_clusters,
            "silhouette_scores": silhouette_scores,
            "patient_matrix": patient_region_matrix,
        }

    def asymmetry_network_analysis(self, asymmetry_data, correlation_threshold=0.6):
        """
        Create and analyze networks based on asymmetry correlations between regions
        """
        # Prepare region correlation matrix
        region_corr_matrix = self._calculate_region_correlations(asymmetry_data)

        # Create network graph
        G = nx.Graph()
        regions = region_corr_matrix.columns

        # Add nodes
        for region in regions:
            G.add_node(region)

        # Add edges based on correlation threshold
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i < j:  # Avoid duplicates
                    corr_val = region_corr_matrix.iloc[i, j]
                    if abs(corr_val) > correlation_threshold:
                        G.add_edge(
                            region1, region2, weight=abs(corr_val), correlation=corr_val
                        )

        # Calculate network metrics
        network_metrics = {
            "centrality_betweenness": nx.betweenness_centrality(G),
            "centrality_degree": nx.degree_centrality(G),
            "centrality_eigenvector": nx.eigenvector_centrality(G)
            if nx.is_connected(G)
            else {},
            "clustering_coefficient": nx.clustering(G),
            "communities": list(nx.community.greedy_modularity_communities(G)),
        }

        return {
            "graph": G,
            "correlation_matrix": region_corr_matrix,
            "network_metrics": network_metrics,
        }

    def temporal_asymmetry_analysis(self, longitudinal_data):
        """
        Analyze asymmetry changes over time (if longitudinal data available)
        """
        if "timepoint" not in longitudinal_data.columns:
            raise ValueError("Longitudinal analysis requires 'timepoint' column")

        results = {}
        patients = longitudinal_data["patient_id"].unique()

        for patient in patients:
            patient_data = longitudinal_data[longitudinal_data["patient_id"] == patient]
            timepoints = sorted(patient_data["timepoint"].unique())

            if len(timepoints) > 1:
                # Calculate asymmetry change rates
                change_rates = {}
                for region in patient_data["region"].unique():
                    region_data = patient_data[patient_data["region"] == region]
                    if len(region_data) == len(timepoints):
                        asymmetry_values = region_data.sort_values("timepoint")[
                            "laterality_index"
                        ].values
                        # Linear trend analysis
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            timepoints, asymmetry_values
                        )
                        change_rates[region] = {
                            "slope": slope,
                            "r_squared": r_value**2,
                            "p_value": p_value,
                        }

                results[patient] = change_rates

        return results

    def asymmetry_outlier_detection(
        self, asymmetry_data, methods=["IQR", "Z-score", "Isolation_Forest"]
    ):
        """
        Identify outlier patients or regions using multiple detection methods
        """
        outlier_results = {}

        # Prepare patient-region matrix
        patient_region_matrix = self._prepare_patient_region_matrix(asymmetry_data)

        for method in methods:
            if method == "IQR":
                outliers = self._detect_iqr_outliers(patient_region_matrix)
            elif method == "Z-score":
                outliers = self._detect_zscore_outliers(patient_region_matrix)
            elif method == "Isolation_Forest":
                outliers = self._detect_isolation_forest_outliers(patient_region_matrix)

            outlier_results[method] = outliers

        return outlier_results

    def asymmetry_prediction_model(self, asymmetry_data, target_variable=None):
        """
        Build machine learning models to predict clinical outcomes from asymmetry patterns
        """
        if target_variable is None:
            # Unsupervised analysis - predict cluster membership
            clustering_results = self.hierarchical_asymmetry_clustering(asymmetry_data)
            target_variable = clustering_results["cluster_labels"]

        patient_region_matrix = self._prepare_patient_region_matrix(asymmetry_data)

        # Random Forest for feature importance
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(patient_region_matrix, target_variable)

        feature_importance = pd.DataFrame(
            {
                "region": [
                    f"region_{i}" for i in range(patient_region_matrix.shape[1])
                ],
                "importance": rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return {
            "model": rf_model,
            "feature_importance": feature_importance,
            "accuracy": rf_model.score(patient_region_matrix, target_variable),
        }

    def create_advanced_visualizations(self, analysis_results, save_dir=None):
        """
        Create sophisticated visualizations for advanced analysis results
        """
        figs = []

        # 1. Dimensionality reduction visualization
        if "PCA" in analysis_results:
            fig_pca = self._plot_pca_analysis(analysis_results["PCA"])
            figs.append(("PCA_analysis", fig_pca))

        # 2. Hierarchical clustering dendrogram
        if "clustering" in analysis_results:
            fig_cluster = self._plot_clustering_analysis(analysis_results["clustering"])
            figs.append(("clustering_analysis", fig_cluster))

        # 3. Network visualization
        if "network" in analysis_results:
            fig_network = self._plot_network_analysis(analysis_results["network"])
            figs.append(("network_analysis", fig_network))

        # Save figures if directory provided
        if save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)
            for fig_name, fig in figs:
                fig.write_html(f"{save_dir}/{fig_name}.html")

        return figs

    # Helper methods
    def _prepare_patient_region_matrix(self, asymmetry_data):
        """Convert asymmetry data to patient x region matrix"""
        pivot_data = asymmetry_data.pivot_table(
            index="patient_id",
            columns="region",
            values="laterality_index",
            fill_value=0,
        )
        return pivot_data.values

    def _calculate_region_correlations(self, asymmetry_data):
        """Calculate correlations between regions across patients"""
        pivot_data = asymmetry_data.pivot_table(
            index="patient_id",
            columns="region",
            values="laterality_index",
            fill_value=0,
        )
        return pivot_data.corr()

    def _detect_iqr_outliers(self, data):
        """Detect outliers using IQR method"""
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
        return outliers

    def _detect_zscore_outliers(self, data, threshold=3):
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(data, axis=0))
        outliers = (z_scores > threshold).any(axis=1)
        return outliers

    def _detect_isolation_forest_outliers(self, data):
        """Detect outliers using Isolation Forest"""
        from sklearn.ensemble import IsolationForest

        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(data)
        outliers = outlier_labels == -1
        return outliers

    def _plot_pca_analysis(self, pca_results):
        """Create PCA visualization"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "PC1 vs PC2",
                "Explained Variance",
                "Component Loadings",
                "Scree Plot",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "scatter"}],
            ],
        )

        # PC1 vs PC2 scatter
        transformed_data = pca_results["transformed_data"]
        fig.add_trace(
            go.Scatter(
                x=transformed_data[:, 0],
                y=transformed_data[:, 1],
                mode="markers",
                name="Patients",
            ),
            row=1,
            col=1,
        )

        # Explained variance
        explained_var = pca_results["explained_variance"]
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(explained_var) + 1)),
                y=explained_var,
                name="Explained Variance",
            ),
            row=1,
            col=2,
        )

        return fig

    def _plot_clustering_analysis(self, clustering_results):
        """Create clustering visualization"""
        from scipy.cluster.hierarchy import dendrogram

        fig = go.Figure()

        # Create dendrogram data
        linkage_matrix = clustering_results["linkage_matrix"]
        dendro = dendrogram(linkage_matrix, no_plot=True)

        # Plot dendrogram
        fig.add_trace(
            go.Scatter(
                x=dendro["dcoord"][0],
                y=dendro["icoord"][0],
                mode="lines",
                name="Dendrogram",
            )
        )

        fig.update_layout(
            title="Hierarchical Clustering of Patients",
            xaxis_title="Distance",
            yaxis_title="Patients",
        )

        return fig

    def _plot_network_analysis(self, network_results):
        """Create network visualization"""
        G = network_results["graph"]
        pos = nx.spring_layout(G)

        # Extract node and edge information
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode="markers+text",
            text=list(G.nodes()),
            textposition="middle center",
            marker=dict(size=10),
        )

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="gray"),
            hoverinfo="none",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Asymmetry Correlation Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        return fig


class AsymmetryPatternAnalyzer:
    """Specialized analysis for asymmetry pattern detection"""

    def __init__(self):
        pass

    def detect_systematic_patterns(self, asymmetry_data):
        """
        Detect systematic asymmetry patterns across regions and patients
        """
        patterns = {}

        # 1. Regional consistency analysis
        regional_patterns = self._analyze_regional_consistency(asymmetry_data)
        patterns["regional"] = regional_patterns

        # 2. Patient-specific patterns
        patient_patterns = self._analyze_patient_patterns(asymmetry_data)
        patterns["patient"] = patient_patterns

        # 3. Hemispheric dominance patterns
        hemispheric_patterns = self._analyze_hemispheric_dominance(asymmetry_data)
        patterns["hemispheric"] = hemispheric_patterns

        return patterns

    def asymmetry_stability_analysis(self, asymmetry_data):
        """
        Analyze stability of asymmetry measurements across different metrics
        """
        stability_results = {}

        asymmetry_metrics = [
            "laterality_index",
            "asymmetry_index",
            "normalized_asymmetry",
        ]

        for patient in asymmetry_data["patient_id"].unique():
            patient_data = asymmetry_data[asymmetry_data["patient_id"] == patient]

            metric_correlations = {}
            for i, metric1 in enumerate(asymmetry_metrics):
                for metric2 in asymmetry_metrics[i + 1 :]:
                    if (
                        metric1 in patient_data.columns
                        and metric2 in patient_data.columns
                    ):
                        corr = patient_data[metric1].corr(patient_data[metric2])
                        metric_correlations[f"{metric1}_vs_{metric2}"] = corr

            stability_results[patient] = metric_correlations

        return stability_results

    def _analyze_regional_consistency(self, asymmetry_data):
        """Analyze consistency of asymmetry direction across patients for each region"""
        regional_stats = {}

        for region in asymmetry_data["region"].unique():
            region_data = asymmetry_data[asymmetry_data["region"] == region][
                "laterality_index"
            ]

            # Calculate consistency metrics
            left_dominant = (region_data > 0).sum()
            right_dominant = (region_data < 0).sum()
            symmetric = (region_data == 0).sum()

            consistency_ratio = max(left_dominant, right_dominant) / len(region_data)

            regional_stats[region] = {
                "left_dominant_count": left_dominant,
                "right_dominant_count": right_dominant,
                "symmetric_count": symmetric,
                "consistency_ratio": consistency_ratio,
                "mean_asymmetry": region_data.mean(),
                "std_asymmetry": region_data.std(),
            }

        return regional_stats

    def _analyze_patient_patterns(self, asymmetry_data):
        """Analyze individual patient asymmetry patterns"""
        patient_stats = {}

        for patient in asymmetry_data["patient_id"].unique():
            patient_data = asymmetry_data[asymmetry_data["patient_id"] == patient]

            # Overall bias
            overall_bias = patient_data["laterality_index"].mean()
            max_asymmetry = patient_data["laterality_index"].abs().max()

            # Regional variation
            regional_variation = patient_data["laterality_index"].std()

            patient_stats[patient] = {
                "overall_bias": overall_bias,
                "max_asymmetry": max_asymmetry,
                "regional_variation": regional_variation,
                "hemisphere_preference": "left"
                if overall_bias > 0
                else "right"
                if overall_bias < 0
                else "symmetric",
            }

        return patient_stats

    def _analyze_hemispheric_dominance(self, asymmetry_data):
        """Analyze overall hemispheric dominance patterns"""
        dominance_stats = {
            "group_mean_laterality": asymmetry_data["laterality_index"].mean(),
            "group_std_laterality": asymmetry_data["laterality_index"].std(),
            "left_dominant_regions": (
                asymmetry_data.groupby("region")["laterality_index"].mean() > 0
            ).sum(),
            "right_dominant_regions": (
                asymmetry_data.groupby("region")["laterality_index"].mean() < 0
            ).sum(),
            "symmetric_regions": (
                asymmetry_data.groupby("region")["laterality_index"].mean() == 0
            ).sum(),
        }

        return dominance_stats
