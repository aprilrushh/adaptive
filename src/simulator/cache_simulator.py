import numpy as np
from collections import OrderedDict, deque
import time
import torch  # ← Add this line!

class AdaptiveCacheSimulator:
    """Cache simulator with AI optimization for Intel processors"""
    
    def __init__(self, cache_size=1000, block_size=4096):
        self.cache_size = cache_size
        self.block_size = block_size
        self.cache = OrderedDict()
        self.io_history = deque(maxlen=100)
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.total_latency = 0.0
        
    def access(self, block_id, predictor=None):
        """Simulate block access"""
        # Check cache hit
        if block_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(block_id)  # LRU update
            latency = 0.1  # Cache hit: 0.1ms
        else:
            self.misses += 1
            latency = 10.0  # Cache miss: 10ms
            
            # Add to cache
            if len(self.cache) >= self.cache_size:
                # Evict LRU
                self.cache.popitem(last=False)
            
            self.cache[block_id] = True
        
        self.total_latency += latency
        self.io_history.append(block_id)
        
        # AI-based prefetching
        if predictor and len(self.io_history) >= 10:
            self._prefetch_with_ai(predictor)
        
        return latency < 1.0  # Return True if hit
    
    def _prefetch_with_ai(self, predictor):
        """Use AI to prefetch blocks"""
        # Prepare input (simplified)
        recent_ios = list(self.io_history)[-50:]
        if len(recent_ios) < 10:
            return
        
        # Create input tensor (simplified encoding)
        input_data = torch.FloatTensor([hash(b) % 128 / 128.0 for b in recent_ios])
        input_data = input_data.unsqueeze(0).unsqueeze(-1)  # Add batch and feature dims
        
        # Get predictions
        try:
            predicted_indices, _ = predictor.predict_next_blocks(input_data, top_k=5)
            
            # Prefetch predicted blocks
            for idx in predicted_indices[0]:
                block_id = f"block_{idx}"
                if block_id not in self.cache and len(self.cache) < self.cache_size:
                    self.cache[block_id] = True
        except:
            pass  # Ignore errors in demo
    
    def get_stats(self):
        """Get performance statistics"""
        total = self.hits + self.misses
        if total == 0:
            return {
                'hit_rate': 0,
                'avg_latency': 0,
                'total_accesses': 0
            }
        
        return {
            'hit_rate': self.hits / total,
            'avg_latency': self.total_latency / total,
            'total_accesses': total,
            'cache_utilization': len(self.cache) / self.cache_size
        }
