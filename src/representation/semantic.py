"""
Semantic Distance Amplification
Multi-layer encoding for pattern distance amplification.
"""

from typing import List, Tuple
import numpy as np
from sequence import Direction, SequenceEncoder


class SemanticAmplifier:
    """
    Amplifies semantic distances between transaction patterns.
    Based on: financial-representation.tex Semantic Distance section
    
    Total amplification Γ ≈ 658 through 4 layers:
    - Layer 1 (Directional): γ₁ ≈ 3.7
    - Layer 2 (Positional): γ₂ ≈ 4.2  
    - Layer 3 (Frequency): γ₃ ≈ 5.8
    - Layer 4 (Compression): γ₄ ≈ 7.3
    """
    
    def __init__(self):
        self.gamma_1 = 3.7  # Directional mapping
        self.gamma_2 = 4.2  # Positional context
        self.gamma_3 = 5.8  # Pattern frequency
        self.gamma_4 = 7.3  # Ambiguous compression
        self.total_gamma = self.gamma_1 * self.gamma_2 * self.gamma_3 * self.gamma_4
    
    def layer1_directional_embedding(self, directions: List[Direction]) -> np.ndarray:
        """
        Layer 1: Embed directions into 6D space.
        Amplification: γ₁ ≈ 3.7
        """
        direction_map = {
            Direction.NORTH: np.array([1, 0, 0, 0, 0, 0]),
            Direction.SOUTH: np.array([0, 1, 0, 0, 0, 0]),
            Direction.EAST: np.array([0, 0, 1, 0, 0, 0]),
            Direction.WEST: np.array([0, 0, 0, 1, 0, 0]),
            Direction.UP: np.array([0, 0, 0, 0, 1, 0]),
            Direction.DOWN: np.array([0, 0, 0, 0, 0, 1]),
        }
        
        embeddings = [direction_map[d] for d in directions]
        return np.concatenate(embeddings)
    
    def layer2_positional_context(self, embedding: np.ndarray, position: int, sequence_length: int) -> np.ndarray:
        """
        Layer 2: Add positional context encoding.
        Amplification: γ₂ ≈ 4.2
        
        Uses sinusoidal position encoding like in Transformers.
        """
        d_model = len(embedding)
        pos_encoding = np.zeros(d_model)
        
        for i in range(0, d_model, 2):
            div_term = np.exp(i * -np.log(10000.0) / d_model)
            pos_encoding[i] = np.sin(position * div_term)
            if i + 1 < d_model:
                pos_encoding[i + 1] = np.cos(position * div_term)
        
        # Concatenate original + positional
        return np.concatenate([embedding, pos_encoding])
    
    def layer3_frequency_encoding(self, embedding: np.ndarray, frequency_rank: float) -> np.ndarray:
        """
        Layer 3: Encode pattern frequency information.
        Amplification: γ₃ ≈ 5.8
        
        Args:
            embedding: Input embedding
            frequency_rank: Normalized frequency [0, 1]
        """
        d = len(embedding)
        
        # Frequency features: linear, quadratic, exponential
        freq_features = np.array([
            frequency_rank,
            frequency_rank ** 2,
            np.exp(-frequency_rank),
            np.log(frequency_rank + 1e-6),
            np.sin(2 * np.pi * frequency_rank),
        ])
        
        # Replicate to match embedding dimension
        freq_extended = np.tile(freq_features, (d // len(freq_features)) + 1)[:d]
        
        # Element-wise modulation
        modulated = embedding * (1 + freq_extended)
        return np.concatenate([embedding, modulated])
    
    def layer4_ambiguous_compression(self, embedding: np.ndarray, compression_rate: float = 0.5) -> np.ndarray:
        """
        Layer 4: Ambiguous compression for meta-information.
        Amplification: γ₄ ≈ 7.3
        
        Compresses then expands, extracting meta-information.
        """
        d = len(embedding)
        compressed_dim = max(int(d * compression_rate), 1)
        
        # Random projection for compression (simulates learned compression)
        projection_matrix = np.random.randn(d, compressed_dim) / np.sqrt(compressed_dim)
        compressed = embedding @ projection_matrix
        
        # Expansion back with amplification
        expansion_matrix = np.random.randn(compressed_dim, d) / np.sqrt(d)
        expanded = compressed @ expansion_matrix
        
        # Combine original with compressed-expanded (residual)
        amplified = embedding + expanded * self.gamma_4
        return amplified
    
    def encode_pattern(self, 
                      directions: List[Direction], 
                      position: int,
                      sequence_length: int,
                      frequency_rank: float) -> np.ndarray:
        """
        Full 4-layer encoding pipeline.
        
        Args:
            directions: Directional sequence
            position: Position in overall sequence
            sequence_length: Total sequence length
            frequency_rank: Pattern frequency [0, 1]
        
        Returns:
            Amplified semantic embedding
        """
        # Layer 1: Directional
        emb = self.layer1_directional_embedding(directions)
        
        # Layer 2: Positional
        emb = self.layer2_positional_context(emb, position, sequence_length)
        
        # Layer 3: Frequency
        emb = self.layer3_frequency_encoding(emb, frequency_rank)
        
        # Layer 4: Compression
        emb = self.layer4_ambiguous_compression(emb)
        
        return emb
    
    def semantic_distance(self, pattern1: np.ndarray, pattern2: np.ndarray, weights: np.ndarray = None) -> float:
        """
        Calculate semantic distance with amplification.
        
        d_semantic = Σ w_i ||φ(d₁,ᵢ) - φ(d₂,ᵢ)||₂
        
        Args:
            pattern1, pattern2: Encoded patterns
            weights: Optional dimension weights
        
        Returns:
            Semantic distance
        """
        if weights is None:
            weights = np.ones(len(pattern1))
        
        diff = pattern1 - pattern2
        weighted_diff = weights * diff
        distance = np.linalg.norm(weighted_diff)
        
        return distance


class FinancialLanguageModel:
    """
    LLM-style continuous learning from transaction sequences.
    Based on: financial-representation.tex LLM section
    """
    
    def __init__(self, embedding_dim: int = 64, learning_rate: float = 0.01):
        """
        Initialize language model.
        
        Args:
            embedding_dim: Dimension of learned embeddings
            learning_rate: Learning rate for online updates
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Vocabulary: 6 directions
        self.vocab_size = 6
        
        # Weight matrix for prediction (randomly initialized)
        self.W = np.random.randn(self.vocab_size, embedding_dim) / np.sqrt(embedding_dim)
        
        # Direction to index mapping
        self.dir_to_idx = {
            Direction.NORTH: 0,
            Direction.SOUTH: 1,
            Direction.EAST: 2,
            Direction.WEST: 3,
            Direction.UP: 4,
            Direction.DOWN: 5,
        }
        self.idx_to_dir = {v: k for k, v in self.dir_to_idx.items()}
    
    def embed_sequence(self, sequence: List[Direction]) -> np.ndarray:
        """
        Convert direction sequence to embedding.
        Simple average of direction embeddings.
        """
        embeddings = []
        for direction in sequence:
            idx = self.dir_to_idx[direction]
            one_hot = np.zeros(self.vocab_size)
            one_hot[idx] = 1.0
            embeddings.append(one_hot)
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        return np.zeros(self.vocab_size)
    
    def predict_next(self, sequence: List[Direction]) -> Tuple[Direction, np.ndarray]:
        """
        Predict next direction given sequence.
        
        P(d_{t+1} | S_{1:t}) = softmax(W · Embed(S_{1:t}))
        
        Args:
            sequence: Historical direction sequence
        
        Returns:
            (predicted_direction, probability_distribution)
        """
        embedding = self.embed_sequence(sequence)
        
        # Linear projection
        logits = self.W @ embedding
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Most likely direction
        predicted_idx = np.argmax(probs)
        predicted_dir = self.idx_to_dir[predicted_idx]
        
        return predicted_dir, probs
    
    def update(self, sequence: List[Direction], actual_next: Direction):
        """
        Online learning update.
        
        W(t+1) = W(t) + η · ∇_W L(d_{t+1}, d̂_{t+1})
        
        Args:
            sequence: Input sequence
            actual_next: Actual next direction (ground truth)
        """
        embedding = self.embed_sequence(sequence)
        predicted_dir, probs = self.predict_next(sequence)
        
        # Target distribution (one-hot)
        target = np.zeros(self.vocab_size)
        target[self.dir_to_idx[actual_next]] = 1.0
        
        # Cross-entropy gradient
        grad = np.outer(probs - target, embedding)
        
        # Gradient descent
        self.W -= self.learning_rate * grad
    
    def continuous_learn(self, stream: List[List[Direction]], window_size: int = 10):
        """
        Continuously learn from transaction stream.
        
        Args:
            stream: Stream of direction sequences
            window_size: Context window size
        """
        for i in range(len(stream) - 1):
            # Use last window_size directions as context
            start = max(0, i - window_size)
            context = [d for seq in stream[start:i+1] for d in seq]
            
            # Next sequence to predict
            next_seq = stream[i + 1]
            
            # Update for each direction in next sequence
            for next_dir in next_seq:
                self.update(context, next_dir)
                context.append(next_dir)
