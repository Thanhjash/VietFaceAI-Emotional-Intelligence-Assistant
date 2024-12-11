from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
import streamlit as st
import logging
import gc
from pathlib import Path
import sys
from face_analysis.app.config.app_config import AppConfig

@dataclass
class CacheEntry:
    data: Any
    timestamp: datetime
    ttl: Optional[timedelta]
    size: int = 0

class CacheManager:
    """Memory-efficient cache manager with monitoring and statistics"""
    
    def __init__(self, max_memory_mb: int = AppConfig.CACHE_CONFIG['max_memory_mb']):
        """
        Initialize CacheManager with memory limits and monitoring
        
        Args:
            max_memory_mb (int): Maximum memory usage in MB
        """
        self.logger = logging.getLogger(__name__)
        self.cache_config = AppConfig.CACHE_CONFIG  # Store config reference
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self._initialize_cache()
        
    def cache_analysis_result(self, image: np.ndarray, result: Dict[str, Any]) -> bool:
        """Cache analysis result"""
        try:
            image_hash = str(hash(image.tobytes()))
            return self.add_to_cache(
                key=image_hash,
                data=result,
                cache_type='analysis',
                ttl=AppConfig.CACHE_CONFIG['analysis_ttl']  # Use AppConfig consistently
            )
        except Exception as e:
            self.logger.error(f"Error caching analysis result: {str(e)}")
            return False
        
    def _initialize_cache(self) -> None:
        """Initialize cache state with monitoring metrics"""
        if 'cache_state' not in st.session_state:
            st.session_state.cache_state = {
                'model_cache': {},        # For model weights/data
                'analysis_cache': {},     # For analysis results
                'memory_usage': 0,        # Current memory usage
                'last_cleanup': datetime.now(),
                'hits': 0,               # Cache hit counter
                'misses': 0,             # Cache miss counter
                'evictions': 0           # Number of entries evicted
            }
        self.state = st.session_state.cache_state
        
    def _check_memory_usage(self, new_entry_size: int) -> bool:
        """
        Check if new entry can fit in memory
        
        Args:
            new_entry_size (int): Size of new entry in bytes
            
        Returns:
            bool: True if entry can be added, False otherwise
        """
        if self.state['memory_usage'] + new_entry_size > self.max_memory:
            self._cleanup_old_entries()
        return self.state['memory_usage'] + new_entry_size <= self.max_memory
    
    def _cleanup_old_entries(self) -> None:
        """Clean up expired or old entries to free memory"""
        current_time = datetime.now()
        removed_count = 0
        freed_memory = 0
        
        for cache_dict in [self.state['model_cache'], self.state['analysis_cache']]:
            for key, entry in list(cache_dict.items()):
                # Remove if TTL expired or memory pressure high
                if (entry.ttl and current_time - entry.timestamp > entry.ttl) or \
                (self.state['memory_usage'] > self.max_memory * 0.9):
                    freed_memory += entry.size
                    self.state['memory_usage'] -= entry.size
                    self.state['evictions'] += 1
                    del cache_dict[key]
                    removed_count += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        
        Returns:
            Dict containing cache metrics and status
        """
        return {
            'analysis_cache_size': len(self.state['analysis_cache']),
            'model_cache_size': len(self.state['model_cache']),
            'memory_usage_mb': round(self.state['memory_usage'] / (1024 * 1024), 2),
            'last_cleanup': self.state['last_cleanup'].strftime("%Y-%m-%d %H:%M:%S"),
            'cache_hits': self.state['hits'],
            'cache_misses': self.state['misses'],
            'hit_rate': round(
                self.state['hits'] / (self.state['hits'] + self.state['misses']) * 100 
                if (self.state['hits'] + self.state['misses']) > 0 else 0,
                2
            ),
            'eviction_count': self.state['evictions']
        }
    
    def add_to_cache(self, key: str, data: Any, cache_type: str = 'analysis', 
                    ttl: Optional[int] = None) -> bool:
        """
        Add item to cache with memory management
        
        Args:
            key (str): Cache key
            data (Any): Data to cache
            cache_type (str): Either 'model' or 'analysis'
            ttl (Optional[int]): Time to live in seconds
            
        Returns:
            bool: True if added successfully, False if memory constraints prevent adding
        """
        try:
            # Calculate size
            if isinstance(data, np.ndarray):
                size = data.nbytes
            else:
                size = sys.getsizeof(data)
                
            # Check memory constraints
            if not self._check_memory_usage(size):
                self.logger.warning(f"Cannot add to cache: memory limit exceeded")
                return False
                
            # Create cache entry
            entry = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                ttl=timedelta(seconds=ttl) if ttl else None,
                size=size
            )
            
            # Add to appropriate cache
            cache_dict = (self.state['model_cache'] if cache_type == 'model' 
                        else self.state['analysis_cache'])
            cache_dict[key] = entry
            self.state['memory_usage'] += size
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding to cache: {str(e)}")
            return False
    
    def get_from_cache(self, key: str, cache_type: str = 'analysis') -> Optional[Any]:
        """
        Retrieve item from cache with hit/miss tracking
        
        Args:
            key (str): Cache key
            cache_type (str): Either 'model' or 'analysis'
            
        Returns:
            Optional[Any]: Cached data if found and valid, None otherwise
        """
        cache_dict = (self.state['model_cache'] if cache_type == 'model' 
                     else self.state['analysis_cache'])
        
        if key in cache_dict:
            entry = cache_dict[key]
            current_time = datetime.now()
            
            # Check if entry is still valid
            if entry.ttl and current_time - entry.timestamp > entry.ttl:
                del cache_dict[key]
                self.state['memory_usage'] -= entry.size
                self.state['misses'] += 1
                return None
                
            self.state['hits'] += 1
            return entry.data
            
        self.state['misses'] += 1
        return None
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear specified cache type or all caches
        
        Args:
            cache_type (Optional[str]): 'model', 'analysis', or None for all
        """
        try:
            if cache_type == 'model' or cache_type is None:
                self.state['memory_usage'] -= sum(
                    entry.size for entry in self.state['model_cache'].values()
                )
                self.state['model_cache'].clear()
                
            if cache_type == 'analysis' or cache_type is None:
                self.state['memory_usage'] -= sum(
                    entry.size for entry in self.state['analysis_cache'].values()
                )
                self.state['analysis_cache'].clear()
                
            gc.collect()
            self.logger.info(f"Cache cleared: {cache_type or 'all'}")
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")

    def get_analysis_result(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        try:
            # Create image hash for key
            image_hash = str(hash(image.tobytes()))
            return self.get_from_cache(image_hash, 'analysis')
        except Exception as e:
            self.logger.error(f"Error getting analysis result: {str(e)}")
            return None

