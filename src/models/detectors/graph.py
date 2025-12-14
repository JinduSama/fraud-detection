"""
Graph-Based Fraud Detector.

Implements fraud detection using graph analysis to identify
networks of suspicious accounts through community detection
and centrality measures.
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..base import BaseDetector
from ..preprocessing import DataPreprocessor


class GraphDetector(BaseDetector):
    """
    Graph-based fraud detector using network analysis.
    
    Builds a similarity graph where nodes are records and edges
    represent high similarity. Detects fraud through:
    - Dense subgraph detection (fraud rings)
    - High betweenness centrality (bridge accounts)
    - Community detection (coordinated fraud)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_community_size: int = 3,
        use_betweenness: bool = True,
        betweenness_threshold: float = 0.1,
        name: str = "GraphDetector"
    ):
        """
        Initialize the graph detector.
        
        Args:
            similarity_threshold: Minimum similarity to create an edge.
            min_community_size: Minimum community size to flag as suspicious.
            use_betweenness: Whether to use betweenness centrality.
            betweenness_threshold: Threshold for flagging high-centrality nodes.
            name: Human-readable name.
        """
        super().__init__(name=name)
        
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for GraphDetector. Install with: pip install networkx")
        
        self.similarity_threshold = similarity_threshold
        self.min_community_size = min_community_size
        self.use_betweenness = use_betweenness
        self.betweenness_threshold = betweenness_threshold
        
        self._graph: Optional[nx.Graph] = None
        self._communities: list[set[int]] = []
        self._betweenness: dict[int, float] = {}
        self._preprocessor = DataPreprocessor()
    
    def _compute_similarity(self, row1: pd.Series, row2: pd.Series) -> float:
        """Compute similarity between two records based on shared attributes."""
        similarity = 0.0

        has_structured_address = all(
            c in row1.index and c in row2.index for c in ["strasse", "hausnummer", "plz", "stadt"]
        )

        weights = {
            "iban": 0.38,
            "email": 0.18,
            "surname": 0.10,
            # Prefer structured address parts when available
            "strasse": 0.12,
            "hausnummer": 0.06,
            "plz": 0.08,
            "stadt": 0.08,
            # Fallback full address
            "address": 0.18,
        }
        total_weight = sum(weights.values())
        
        for field, weight in weights.items():
            if field == "address" and has_structured_address:
                continue
            if field in row1.index and field in row2.index:
                val1 = str(row1[field]).lower() if pd.notna(row1[field]) else ""
                val2 = str(row2[field]).lower() if pd.notna(row2[field]) else ""
                
                if val1 and val2 and val1 == val2:
                    similarity += weight
        
        return similarity / total_weight
    
    def _build_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build similarity graph from DataFrame.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            NetworkX graph with records as nodes.
        """
        G = nx.Graph()
        n = len(df)
        
        # Add all records as nodes
        for i in range(n):
            G.add_node(i, **df.iloc[i].to_dict())
        
        # Add edges based on similarity
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._compute_similarity(df.iloc[i], df.iloc[j])
                if sim >= self.similarity_threshold:
                    G.add_edge(i, j, weight=sim)
        
        return G
    
    def _detect_communities(self, G: nx.Graph) -> list[set[int]]:
        """
        Detect communities using greedy modularity optimization.
        
        Args:
            G: NetworkX graph.
            
        Returns:
            List of communities (sets of node indices).
        """
        if len(G.edges()) == 0:
            return []
        
        try:
            # Use greedy modularity communities
            communities = list(nx.community.greedy_modularity_communities(G))
            return [set(c) for c in communities if len(c) >= self.min_community_size]
        except Exception:
            # Fallback: use connected components
            components = list(nx.connected_components(G))
            return [c for c in components if len(c) >= self.min_community_size]
    
    def fit(self, df: pd.DataFrame) -> "GraphDetector":
        """
        Fit the graph detector by building the similarity graph.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            self: The fitted detector.
        """
        # Build similarity graph
        self._graph = self._build_graph(df)
        
        # Detect communities
        self._communities = self._detect_communities(self._graph)
        
        # Compute betweenness centrality if enabled
        if self.use_betweenness and len(self._graph.edges()) > 0:
            self._betweenness = nx.betweenness_centrality(self._graph)
        else:
            self._betweenness = {}
        
        self._is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud based on graph analysis.
        
        Args:
            df: DataFrame with customer records.
            
        Returns:
            DataFrame with 'score', 'is_fraud', 'reason' columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        
        n = len(df)
        scores = np.zeros(n)
        is_fraud = np.zeros(n, dtype=bool)
        reasons = [""] * n
        
        # Flag records in suspicious communities
        for community in self._communities:
            community_score = len(community) / n  # Larger communities = higher score
            community_score = min(community_score * 5, 1.0)  # Scale up but cap at 1.0
            
            for node_idx in community:
                if node_idx < n:
                    scores[node_idx] = max(scores[node_idx], community_score)
                    is_fraud[node_idx] = True
                    if reasons[node_idx]:
                        reasons[node_idx] += ", dense_subgraph"
                    else:
                        reasons[node_idx] = f"dense_subgraph (size={len(community)})"
        
        # Flag high betweenness centrality nodes
        if self.use_betweenness:
            for node_idx, centrality in self._betweenness.items():
                if centrality > self.betweenness_threshold and node_idx < n:
                    scores[node_idx] = max(scores[node_idx], centrality)
                    is_fraud[node_idx] = True
                    if reasons[node_idx]:
                        reasons[node_idx] += f", high_centrality ({centrality:.3f})"
                    else:
                        reasons[node_idx] = f"high_centrality ({centrality:.3f})"
        
        return pd.DataFrame({
            "score": scores,
            "is_fraud": is_fraud,
            "reason": reasons
        }, index=df.index)
    
    def explain(self, df: pd.DataFrame, idx: int) -> dict:
        """Explain why a specific record was flagged."""
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before explanation")
        
        predictions = self.predict(df.iloc[[idx]] if isinstance(idx, int) else df.loc[[idx]])
        
        # Find community membership
        community_info = None
        for i, community in enumerate(self._communities):
            if idx in community:
                community_info = {
                    "community_id": i,
                    "community_size": len(community),
                    "members": list(community)[:10]
                }
                break
        
        # Get neighbors in graph
        neighbors = []
        if self._graph and idx in self._graph:
            neighbors = [
                {"node": n, "similarity": self._graph[idx][n].get("weight", 0)}
                for n in self._graph.neighbors(idx)
            ]
            neighbors.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "index": idx,
            "score": float(predictions["score"].iloc[0]),
            "is_fraud": bool(predictions["is_fraud"].iloc[0]),
            "reason": predictions["reason"].iloc[0],
            "detector": self.name,
            "community_info": community_info,
            "betweenness_centrality": self._betweenness.get(idx, 0.0),
            "neighbors": neighbors[:5]
        }
    
    @property
    def graph(self) -> Optional[nx.Graph]:
        """Access the similarity graph."""
        return self._graph
    
    @property
    def communities(self) -> list[set[int]]:
        """Get detected communities."""
        return self._communities


if __name__ == "__main__":
    # Test the detector
    test_data = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005", "C006"],
        "surname": ["Mueller", "Mueller", "Smith", "Johnson", "Mueller", "Williams"],
        "first_name": ["Hans", "Hans", "John", "Jane", "Hans", "Bob"],
        "address": ["Main St 1", "Main St 1", "Oak Ave 5", "Pine Rd 10", "Main St 1", "Elm St 20"],
        "email": ["hans@test.com", "hans@test.com", "john@test.com", 
                 "jane@test.com", "h@test.com", "bob@outlook.com"],
        "iban": ["DE123", "DE123", "DE456", "DE789", "DE123", "GB999"],
    })
    
    detector = GraphDetector(similarity_threshold=0.5, min_community_size=2)
    detector.fit(test_data)
    results = detector.predict(test_data)
    
    print("Graph Detector Results:")
    print(results)
    print(f"\nCommunities found: {len(detector.communities)}")
    for i, c in enumerate(detector.communities):
        print(f"  Community {i}: {c}")
