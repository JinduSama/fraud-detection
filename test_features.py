"""Quick test of new fraud detection features."""

import pandas as pd
from src.models import IsolationForestDetector, DBSCANDetector, EnsembleDetector, FusionStrategy

# Create test data
test_data = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006'],
    'surname': ['Mueller', 'Muller', 'Smith', 'RandomXYZ', 'Mueller', 'Williams'],
    'first_name': ['Hans', 'Hans', 'John', 'Test123', 'Hans', 'Bob'],
    'strasse': ['Main St', 'Main St', 'Oak Ave', 'Fake', 'Main St', 'Elm St'],
    'hausnummer': ['1', '1', '5', '123', '1', '20'],
    'plz': ['10115', '10115', '20095', '00000', '10115', '99999'],
    'stadt': ['Berlin', 'Berlin', 'Hamburg', 'Nowhere', 'Berlin', 'London'],
    'address': [
        'Main St 1, 10115 Berlin',
        'Main St 1, 10115 Berlin',
        'Oak Ave 5, 20095 Hamburg',
        'Fake 123, 00000 Nowhere',
        'Main St 1, 10115 Berlin',
        'Elm St 20, 99999 London'
    ],
    'email': ['hans@test.com', 'hans@test.com', 'john@test.com', 'x1y2@fake.net', 'h@test.com', 'bob@outlook.com'],
    'iban': ['DE123', 'DE123', 'DE456', 'XX000', 'DE123', 'GB999'],
    'date_of_birth': ['1990-01-01', '1990-01-01', '1985-06-15', '2000-01-01', '1990-01-01', '1988-12-25'],
    'nationality': ['German', 'German', 'British', 'Unknown', 'German', 'British'],
})

print('=' * 50)
print('FRAUD DETECTION FEATURE TEST')
print('=' * 50)

print('\n1. Testing Isolation Forest Detector...')
iso = IsolationForestDetector(contamination=0.3, random_state=42)
iso.fit(test_data)
iso_results = iso.predict(test_data)
print(f'   Flagged: {iso_results["is_fraud"].sum()}/6')
print(f'   Scores: {iso_results["score"].round(2).tolist()}')

print('\n2. Testing DBSCAN Detector...')
dbscan = DBSCANDetector(eps=0.4, min_samples=2)
dbscan.fit(test_data)
dbscan_results = dbscan.predict(test_data)
print(f'   Flagged: {dbscan_results["is_fraud"].sum()}/6')
print(f'   Clusters found: {len(dbscan.clusters)}')

print('\n3. Testing LOF Detector...')
from src.models import LocalOutlierFactorDetector
lof = LocalOutlierFactorDetector(n_neighbors=3, contamination=0.3)
lof.fit(test_data)
lof_results = lof.predict(test_data)
print(f'   Flagged: {lof_results["is_fraud"].sum()}/6')

print('\n4. Testing Ensemble Detector (WEIGHTED_AVG)...')
iso2 = IsolationForestDetector(contamination=0.3, random_state=42)
dbscan2 = DBSCANDetector(eps=0.4, min_samples=2)
ensemble = EnsembleDetector(
    detectors=[(iso2, 0.5), (dbscan2, 0.5)],
    strategy=FusionStrategy.WEIGHTED_AVG
)
ensemble.fit(test_data)
ensemble_results = ensemble.predict(test_data)
print(f'   Flagged: {ensemble_results["is_fraud"].sum()}/6')
print(f'   Scores: {ensemble_results["score"].round(2).tolist()}')

print('\n5. Testing Ensemble Detector (VOTING)...')
iso3 = IsolationForestDetector(contamination=0.3, random_state=42)
dbscan3 = DBSCANDetector(eps=0.4, min_samples=2)
lof2 = LocalOutlierFactorDetector(n_neighbors=3, contamination=0.3)
ensemble_vote = EnsembleDetector(
    detectors=[(iso3, 1.0), (dbscan3, 1.0), (lof2, 1.0)],
    strategy=FusionStrategy.VOTING
)
ensemble_vote.fit(test_data)
vote_results = ensemble_vote.predict(test_data)
print(f'   Flagged: {vote_results["is_fraud"].sum()}/6')

print('\n6. Testing Feature Extractor...')
from src.models.features import FeatureExtractor
extractor = FeatureExtractor()
features = extractor.extract_features(test_data)
print(f'   Feature columns: {len(features.columns)}')
print(f'   Sample features: {list(features.columns[:5])}...')

print('\n7. Testing Graph Detector (networkx)...')
from src.models.detectors.graph import GraphDetector
graph = GraphDetector(similarity_threshold=0.6, min_community_size=2)
graph.fit(test_data)
graph_results = graph.predict(test_data)
print(f'   Flagged: {graph_results["is_fraud"].sum()}/6')
print(f'   Communities found: {len(graph.communities)}')

print('\n8. Testing Explainer (SHAP)...')
from src.models.explainer import FraudExplainer
from src.models import IsolationForestDetector

# Create and fit a detector for explanation
iso_explain = IsolationForestDetector(contamination=0.3, random_state=42)
iso_explain.fit(test_data)
iso_results = iso_explain.predict(test_data)

# Get flagged record indices
flagged_indices = test_data.index[iso_results['is_fraud']].tolist()

explainer = FraudExplainer(iso_explain)
explanations = []
for idx in flagged_indices:
    exp = explainer.explain_isolation_forest(test_data, idx)
    explanations.append(exp)

print(f'   Explanations generated: {len(explanations)}')
if explanations:
    print(f'   Sample explanation keys: {list(explanations[0].keys())}')
    top_feats = explanations[0].get("top_features", [])
    print(f'   Top feature for first flagged: {top_feats[0] if top_feats else "N/A"}')

print('\n9. Testing Ensemble with Graph + Explainer...')
iso4 = IsolationForestDetector(contamination=0.3, random_state=42)
graph2 = GraphDetector(similarity_threshold=0.6, min_community_size=2)
ensemble_graph = EnsembleDetector(
    detectors=[(iso4, 0.6), (graph2, 0.4)],
    strategy=FusionStrategy.WEIGHTED_AVG
)
ensemble_graph.fit(test_data)
ensemble_graph_results = ensemble_graph.predict(test_data)
print(f'   Flagged: {ensemble_graph_results["is_fraud"].sum()}/6')

# Explain ensemble results for first flagged record
flagged_ensemble_idx = test_data.index[ensemble_graph_results['is_fraud']].tolist()
ensemble_explainer = FraudExplainer(ensemble_graph)
ensemble_explanations = []
for idx in flagged_ensemble_idx[:2]:  # Just first 2 for speed
    exp = ensemble_explainer.explain_ensemble(test_data, idx)
    ensemble_explanations.append(exp)
print(f'   Ensemble explanations: {len(ensemble_explanations)}')
if ensemble_explanations:
    print(f'   Detectors in breakdown: {list(ensemble_explanations[0].get("detector_contributions", {}).keys())}')

print('\n' + '=' * 50)
print('ALL TESTS PASSED!')
print('=' * 50)
