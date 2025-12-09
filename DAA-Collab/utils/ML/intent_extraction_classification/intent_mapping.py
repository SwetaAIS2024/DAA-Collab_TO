from typing import Dict, List
from utils.ML.intent_extraction_classification.intent_keywords_static import (
    VISUALIZATION_KEYWORDS, 
    ADVANCED_KEYWORDS,
    CAUSAL_ANALYSIS_KEYWORDS,
    INCIDENT_DETECTION_KEYWORDS,
    INCIDENT_CLASSIFICATION_KEYWORDS,
    TRAFFIC_ANOMALY_KEYWORDS,
    TRAFFIC_FORECASTING_KEYWORDS,
    SPATIAL_TEMPORAL_KEYWORDS,
    TRAFFIC_IMPACT_KEYWORDS,
    TRAFFIC_META_KEYWORDS
)

# Intent mapping configuration
INTENT_MAPPING: Dict[str, List[str]] = {
    "visualization": VISUALIZATION_KEYWORDS,
    "advanced": ADVANCED_KEYWORDS,
    "causal_analysis": CAUSAL_ANALYSIS_KEYWORDS,
    "incident_detection": INCIDENT_DETECTION_KEYWORDS,
    "incident_classification": INCIDENT_CLASSIFICATION_KEYWORDS,
    "traffic_anomaly": TRAFFIC_ANOMALY_KEYWORDS,
    "traffic_forecasting": TRAFFIC_FORECASTING_KEYWORDS,
    "spatial_temporal": SPATIAL_TEMPORAL_KEYWORDS,
    "traffic_impact": TRAFFIC_IMPACT_KEYWORDS,
    "meta_attributes": TRAFFIC_META_KEYWORDS
}