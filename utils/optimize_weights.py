# optimize_weights_improved.py
# Improved version with better evaluation metrics and scoring

import asyncio
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import traceback
import nest_asyncio
from datetime import datetime, timedelta

# Import your existing modules
from config import *
from services.embedding_generator import EmbeddingGenerator
from services.brave_searcher import BraveSearcher
from services.pinecone_manager import PineconeManager
from services.llm_service import LLMService
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder

# Global service instances
openai_client = None
embedding_generator: EmbeddingGenerator = None
brave_searcher: BraveSearcher = None
pinecone_manager: PineconeManager = None
llm_service: LLMService = None
cross_encoder_model = None

# IMPROVED: More specific and diverse evaluation queries
EVALUATION_QUERIES = [
    # Specific company queries
    "TCS Q1 FY24 earnings results financial performance",
    "Reliance Industries debt equity ratio latest",
    "HDFC Bank NPA provisions quarterly results",
    
    # Market-specific queries  
    "Nifty 50 volatility index current levels",
    "India VIX fear gauge market sentiment",
    
    # Recent news queries
    "RBI monetary policy repo rate decision today",
    "SEBI new mutual fund regulations impact",
    
    # Sector-specific queries
    "IT sector revenue growth forecast 2024",
    "Banking sector credit growth trends",
    "Pharmaceutical export data recent"
]

# IMPROVED: Multiple evaluation metrics
class EvaluationMetrics:
    @staticmethod
    def calculate_ndcg_at_k(relevance_scores, k=5):
        """Calculate Normalized Discounted Cumulative Gain at K"""
        if not relevance_scores or len(relevance_scores) == 0:
            return 0.0
            
        # DCG calculation
        dcg = relevance_scores[0]
        for i in range(1, min(len(relevance_scores), k)):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        # IDCG calculation (ideal ranking)
        sorted_scores = sorted(relevance_scores, reverse=True)
        idcg = sorted_scores[0]
        for i in range(1, min(len(sorted_scores), k)):
            idcg += sorted_scores[i] / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def calculate_diversity_score(passages):
        """Calculate diversity of content sources and topics"""
        if not passages:
            return 0.0
            
        unique_sources = set()
        for p in passages:
            source_link = p.get('metadata', {}).get('link', '')
            if source_link:
                from urllib.parse import urlparse
                try:
                    domain = urlparse(source_link).netloc
                    unique_sources.add(domain)
                except:
                    pass
        
        # Normalize by max expected sources (e.g., 5)
        return min(1.0, len(unique_sources) / 5.0)
    
    @staticmethod
    def calculate_freshness_score(passages):
        """Calculate average freshness of retrieved content"""
        if not passages:
            return 0.0
            
        total_freshness = 0
        count = 0
        
        for p in passages:
            time_score = p.get('time_decay_score', 0)
            if time_score > 0:
                total_freshness += time_score
                count += 1
        
        return total_freshness / count if count > 0 else 0.0

# Enhanced LLMService wrapper for optimization
class OptimizedLLMService:
    """Wrapper around LLMService with enhanced methods for optimization"""
    
    def __init__(self, base_llm_service):
        self.base_service = base_llm_service
        # Copy all attributes from base service
        for attr_name in dir(base_llm_service):
            if not attr_name.startswith('_') or attr_name in ['_calculate_sentiment_score', '_calculate_time_decay_score', '_calculate_impact_score']:
                setattr(self, attr_name, getattr(base_llm_service, attr_name))
    
    def _calculate_sentiment_score_improved(self, text: str, query: str = "") -> float:
        """Enhanced sentiment scoring with query context"""
        try:
            result = self.base_service.sentiment_analyzer(text[:512])
            sentiment = result[0]['label']
            confidence = result[0]['score']
            
            # Query-aware sentiment evaluation
            query_lower = query.lower()
            seeking_risks = any(word in query_lower for word in ['risk', 'problem', 'issue', 'concern', 'decline', 'fall', 'loss', 'npa', 'provisions'])
            seeking_opportunities = any(word in query_lower for word in ['growth', 'profit', 'gain', 'opportunity', 'rise', 'increase', 'earnings', 'performance'])
            
            if seeking_risks:
                # For risk-seeking queries, negative sentiment is valuable
                if sentiment == 'negative':
                    return 0.5 + (confidence * 0.5)  # Range [0.5, 1.0]
                elif sentiment == 'positive':
                    return 0.5 - (confidence * 0.3)  # Range [0.2, 0.5]
                else:  # neutral
                    return 0.5
            elif seeking_opportunities:
                # For opportunity-seeking queries, positive sentiment is valuable
                if sentiment == 'positive':
                    return 0.5 + (confidence * 0.5)  # Range [0.5, 1.0]
                elif sentiment == 'negative':
                    return 0.5 - (confidence * 0.3)  # Range [0.2, 0.5]
                else:  # neutral
                    return 0.5
            else:
                # For neutral queries, use absolute confidence regardless of polarity
                return 0.3 + (confidence * 0.4)  # Range [0.3, 0.7]
                
        except Exception as e:
            print(f"WARNING: Failed to calculate sentiment score: {e}")
            return 0.5
    
    def _calculate_time_decay_score_improved(self, publication_date_str: str, query: str = "") -> float:
        """Enhanced time decay with query-aware decay rates"""
        if not publication_date_str:
            return 0.4
            
        try:
            from datetime import datetime
            published_date = datetime.fromisoformat(publication_date_str)
            now = datetime.now()
            age_in_days = (now - published_date).days
            
            # Query-dependent decay rates
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['latest', 'recent', 'today', 'current', 'now']):
                half_life_days = 15  # Fast decay for recency queries
            elif any(word in query_lower for word in ['earnings', 'results', 'quarterly', 'q1', 'q2', 'q3', 'q4']):
                half_life_days = 45  # Medium decay for periodic reports
            elif any(word in query_lower for word in ['policy', 'regulation', 'sebi', 'rbi', 'reform']):
                half_life_days = 90  # Slow decay for regulatory content
            else:
                half_life_days = 30  # Default
                
            decay_constant = 0.693 / half_life_days
            
            # Gentler sigmoid decay
            score = 1 / (1 + np.exp(decay_constant * (age_in_days - half_life_days)))
            return max(0.1, min(1.0, score))
            
        except Exception as e:
            print(f"WARNING: Failed to calculate time decay score: {e}")
            return 0.4
    
    async def _get_reranked_passages_enhanced(self, question: str, query_results,
                                            w_relevance: float,
                                            w_sentiment: float,
                                            w_time_decay: float,
                                            w_impact: float) -> list[dict]:
        """Enhanced re-ranking with improved scoring methods"""
        passages = []

        for match in query_results.matches:
            content_for_rerank = ""
            if match.metadata and 'full_content_preview' in match.metadata and match.metadata['full_content_preview']:
                content_for_rerank = match.metadata['full_content_preview']
            elif match.metadata and 'snippet' in match.metadata:
                content_for_rerank = match.metadata['snippet']
            elif match.metadata and 'original_text' in match.metadata:
                content_for_rerank = match.metadata['original_text']

            if content_for_rerank:
                passages.append({
                    "text": content_for_rerank,
                    "metadata": match.metadata,
                    "pinecone_score": match.score
                })

        if not passages:
            return []

        sentence_pairs = [[question, p["text"]] for p in passages]
        cross_encoder_scores = self.base_service.cross_encoder_model.predict(sentence_pairs)

        for i, passage in enumerate(passages):
            # Use existing methods but with enhanced versions where applicable
            relevance_score = (cross_encoder_scores[i] + 1) / 2 
            
            # Use enhanced sentiment scoring
            sentiment_score = self._calculate_sentiment_score_improved(passage["text"], question)
            
            # Use enhanced time decay
            publication_date_str = passage["metadata"].get("publication_date")
            time_decay_score = self._calculate_time_decay_score_improved(publication_date_str, question)
            
            # Use existing impact score method
            source_link = passage["metadata"].get("link")
            impact_score = self.base_service._calculate_impact_score(passage["text"], source_link)

            final_combined_score = (
                w_relevance * relevance_score +
                w_sentiment * sentiment_score +
                w_time_decay * time_decay_score +
                w_impact * impact_score
            )
            
            passage["final_combined_score"] = final_combined_score
            passage["relevance_score"] = relevance_score
            passage["sentiment_score"] = sentiment_score
            passage["time_decay_score"] = time_decay_score
            passage["impact_score"] = impact_score

        reranked_passages = sorted(passages, key=lambda x: x["final_combined_score"], reverse=True)
        return reranked_passages

def initialize_services_sync():
    """Initialize services with enhanced wrapper"""
    global openai_client, embedding_generator, brave_searcher, pinecone_manager, llm_service, cross_encoder_model

    print("Initializing services for optimization...")
    try:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        embedding_generator = EmbeddingGenerator(openai_api_key=OPENAI_API_KEY)
        brave_searcher = BraveSearcher(brave_api_key=BRAVE_API_KEY)
        pinecone_manager = PineconeManager(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name=PINECONE_INDEX_NAME
        )
        pinecone_manager.connect_to_index()
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        base_llm_service = LLMService(
            openai_client=openai_client,
            cross_encoder_model=cross_encoder_model,
            embedding_generator=embedding_generator,
            brave_searcher=brave_searcher,
            pinecone_manager=pinecone_manager
        )
        
        # Wrap with enhanced service
        llm_service = OptimizedLLMService(base_llm_service)
        
        print("Services initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize services: {e}")
        raise

# IMPROVED: Constrained search space ensuring minimum weights
space = [
    Real(0.1, 0.8, name='w_relevance'),    # Ensure minimum relevance
    Real(0.0, 0.6, name='w_sentiment'),    # Reduce sentiment max
    Real(0.1, 0.6, name='w_time_decay'),   # Ensure minimum time consideration
    Real(0.1, 0.7, name='w_impact')        # Reduce impact max
]

@use_named_args(space)
def objective(w_relevance, w_sentiment, w_time_decay, w_impact):
    """
    IMPROVED: Multi-metric objective function
    """
    # Normalize weights
    total_weight = w_relevance + w_sentiment + w_time_decay + w_impact
    if total_weight == 0:
        return 1.0
        
    norm_w_relevance = w_relevance / total_weight
    norm_w_sentiment = w_sentiment / total_weight
    norm_w_time_decay = w_time_decay / total_weight
    norm_w_impact = w_impact / total_weight

    print(f"\n--- Evaluating weights: R={norm_w_relevance:.4f}, S={norm_w_sentiment:.4f}, T={norm_w_time_decay:.4f}, I={norm_w_impact:.4f} ---")

    # Collect multiple metrics
    all_combined_scores = []
    all_ndcg_scores = []
    all_diversity_scores = []
    all_freshness_scores = []

    async def run_evaluation_for_query_async(query):
        try:
            query_embedding = await embedding_generator.get_embedding(query)
            query_results = pinecone_manager.query_index(query_embedding)
            
            if not query_results.matches:
                print(f"WARNING: No results for query: '{query}'")
                return None
            
            reranked_passages = await llm_service._get_reranked_passages_enhanced(
                query, query_results,
                w_relevance=norm_w_relevance,
                w_sentiment=norm_w_sentiment,
                w_time_decay=norm_w_time_decay,
                w_impact=norm_w_impact
            )
            
            if not reranked_passages:
                return None
                
            # Calculate multiple metrics
            top_passages = reranked_passages[:MAX_RERANKED_CONTEXT_ITEMS]
            combined_scores = [p['final_combined_score'] for p in top_passages]
            relevance_scores = [p['relevance_score'] for p in top_passages]
            
            # NDCG based on relevance scores
            ndcg = EvaluationMetrics.calculate_ndcg_at_k(relevance_scores, k=5)
            
            # Diversity score
            diversity = EvaluationMetrics.calculate_diversity_score(top_passages)
            
            # Freshness score
            freshness = EvaluationMetrics.calculate_freshness_score(top_passages)
            
            return {
                'combined_scores': combined_scores,
                'ndcg': ndcg,
                'diversity': diversity,
                'freshness': freshness
            }
            
        except Exception as e:
            print(f"ERROR: Error during evaluation for query '{query}': {e}")
            return None

    # Run evaluations
    evaluation_coroutines = [run_evaluation_for_query_async(query) for query in EVALUATION_QUERIES]
    results = asyncio.run(asyncio.gather(*evaluation_coroutines))
    
    # Aggregate results
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("WARNING: No valid evaluation results.")
        return 1.0
    
    for result in valid_results:
        all_combined_scores.extend(result['combined_scores'])
        all_ndcg_scores.append(result['ndcg'])
        all_diversity_scores.append(result['diversity'])
        all_freshness_scores.append(result['freshness'])
    
    if not all_combined_scores:
        return 1.0
    
    # IMPROVED: Composite score with multiple factors
    avg_combined_score = np.mean(all_combined_scores)
    avg_ndcg = np.mean(all_ndcg_scores) if all_ndcg_scores else 0
    avg_diversity = np.mean(all_diversity_scores) if all_diversity_scores else 0
    avg_freshness = np.mean(all_freshness_scores) if all_freshness_scores else 0
    
    # Composite objective (weights can be adjusted)
    composite_score = (
        0.4 * avg_combined_score +     # Primary score
        0.3 * avg_ndcg +               # Ranking quality
        0.2 * avg_diversity +          # Source diversity
        0.1 * avg_freshness            # Content freshness
    )
    
    print(f"Metrics - Combined: {avg_combined_score:.4f}, NDCG: {avg_ndcg:.4f}, Diversity: {avg_diversity:.4f}, Freshness: {avg_freshness:.4f}")
    print(f"Composite Score: {composite_score:.4f}")
    
    # Add penalty for extreme weight distributions
    weight_entropy = -sum([w * np.log(w + 1e-8) for w in [norm_w_relevance, norm_w_sentiment, norm_w_time_decay, norm_w_impact]])
    balance_bonus = 0.1 * (weight_entropy / np.log(4))  # Normalize by max entropy
    
    final_score = composite_score + balance_bonus
    print(f"Final Score (with balance bonus): {final_score:.4f}")
    
    return -final_score  # Negative for minimization

async def main():
    initialize_services_sync()
    
    print("\nStarting Improved Bayesian Optimization...")
    OPTIMIZATION_CALLS = 25  # Increased for better exploration
    OPTIMIZATION_RANDOM_STARTS = 8
    
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=OPTIMIZATION_CALLS,
        n_random_starts=OPTIMIZATION_RANDOM_STARTS,
        random_state=42,
        verbose=True,
        acq_func="EI"  # Expected Improvement
    )
    
    print("\n--- IMPROVED Optimization Results ---")
    print(f"Best composite score: {-res.fun:.4f}")
    
    best_weights = res.x
    total_best_weight = sum(best_weights)
    norm_best_weights = [w / total_best_weight for w in best_weights]
    
    print(f"Optimal Weights (Normalized):")
    print(f"  W_RELEVANCE: {norm_best_weights[0]:.4f}")
    print(f"  W_SENTIMENT: {norm_best_weights[1]:.4f}")
    print(f"  W_TIME_DECAY: {norm_best_weights[2]:.4f}")
    print(f"  W_IMPACT: {norm_best_weights[3]:.4f}")
    
    # Validation check
    if norm_best_weights[0] < 0.05:  # Relevance too low
        print("\n⚠️  WARNING: Relevance weight is very low. Consider constraining the search space.")
    
    print(f"\nWeight distribution entropy: {-sum([w * np.log(w + 1e-8) for w in norm_best_weights]):.4f}")
    print("Higher entropy indicates more balanced weights.")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())