from typing import List

# Visualization intent keywords
VISUALIZATION_KEYWORDS: List[str] = [
    'visualization', 'plot', 'chart', 'graph', 'histogram', 
    'scatter', 'box', 'bar', 'visualize', 'pairplot', 'heatmap',
    'line', 'pie', 'show', 'display'
]

# Advanced analysis intent keywords
ADVANCED_KEYWORDS: List[str] = [
    'eda', 'analysis', 'correlation', 'cluster', 'regression', 
    'patterns', 'algorithm', 'complexity', 'optimize', 'performance',
    'efficiency', 'statistical', 'model', 'predict'
]

# Causal Analysis keywords
CAUSAL_ANALYSIS_KEYWORDS: List[str] = [
    'causal', 'causality', 'cause', 'effect', 'causal analysis',
    'causal relationship', 'causal inference', 'granger causality',
    'intervention', 'counterfactual', 'causal impact', 'causal discovery',
    'interrelation', 'interrelations', 'causation', 'causal network'
]

# Incident Detection keywords
INCIDENT_DETECTION_KEYWORDS: List[str] = [
    'incident detection', 'detect incident', 'identify incident',
    'find incident', 'incident identification', 'incident discovery',
    'detect accident', 'accident detection', 'event detection',
    'anomaly incident', 'incident occurrence'
]

# Incident Classification keywords
INCIDENT_CLASSIFICATION_KEYWORDS: List[str] = [
    'incident classification', 'classify incident', 'incident type',
    'incident category', 'categorize incident', 'incident types',
    'type of incident', 'accident classification', 'event classification',
    'incident categorization', 'precaution', 'precautionary measures'
]

# Traffic Anomaly Detection keywords
TRAFFIC_ANOMALY_KEYWORDS: List[str] = [
    'anomaly', 'anomaly detection', 'traffic anomaly', 'abnormal',
    'outlier', 'unusual pattern', 'irregular traffic', 'traffic outlier',
    'detect anomaly', 'anomalous traffic', 'traffic irregularity',
    'abnormal traffic', 'unusual traffic'
]

# Traffic Forecasting keywords
TRAFFIC_FORECASTING_KEYWORDS: List[str] = [
    'forecast', 'forecasting', 'predict', 'prediction', 'traffic forecast',
    'traffic prediction', 'incident forecasting', 'future traffic',
    'predict incident', 'incident prediction', 'post-incident forecast',
    'post-incident prediction', 'traffic projection', 'future incident',
    'predict traffic', 'incident impact forecasting'
]

# Spatial and Temporal Analysis keywords
SPATIAL_TEMPORAL_KEYWORDS: List[str] = [
    'spatial', 'temporal', 'spatio-temporal', 'spatiotemporal',
    'space-time', 'spatial analysis', 'temporal analysis',
    'time series', 'timeseries', 'location-based', 'geographic',
    'geographical', 'road segment', 'road node', 'local analysis',
    'global analysis', 'node-level', 'segment-level'
]

# Traffic Impact Analysis keywords
TRAFFIC_IMPACT_KEYWORDS: List[str] = [
    'impact', 'quantify impact', 'impact analysis', 'traffic impact',
    'incident impact', 'impact quantification', 'impact assessment',
    'measure impact', 'impact measurement', 'traffic index',
    'traffic indexes', 'impact on traffic', 'effect on traffic'
]

# Traffic Meta-Attributes keywords
TRAFFIC_META_KEYWORDS: List[str] = [
    'meta-attribute', 'meta attribute', 'traffic attribute',
    'road attribute', 'traffic feature', 'traffic characteristic',
    'meta-data', 'metadata', 'traffic metadata', 'attribute analysis',
    'feature analysis', 'traffic properties'
]