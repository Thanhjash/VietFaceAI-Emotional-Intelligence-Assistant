# face_analysis/app/state/event_bus/event_bus.py

import logging
from typing import Callable, Any, Dict
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Event:
    """Event data structure"""
    type: str
    data: Any
    timestamp: datetime = datetime.now()

class EventMetrics:
    """Track event performance metrics"""
    def __init__(self):
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.processing_times: Dict[str, float] = defaultdict(float)
        
    def record_event(self, event_type: str, processing_time: float):
        self.event_counts[event_type] += 1
        self.processing_times[event_type] += processing_time

class EventBus:
    """Enhanced event bus with metrics and error handling"""
    
    def __init__(self):
        self._subscribers = defaultdict(list)
        self._metrics = EventMetrics()
        self.logger = logging.getLogger(__name__)
        
    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event with type checking"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self._subscribers[event_type].append(callback)
        self.logger.info(f"New subscriber added for event: {event_type}")
        
    def publish(self, event_type: str, data: Any) -> None:
        """Publish an event with performance tracking"""
        start_time = datetime.now()
        event = Event(type=event_type, data=data)
        
        try:
            for callback in self._subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {str(e)}")
                    
            processing_time = (datetime.now() - start_time).total_seconds()
            self._metrics.record_event(event_type, processing_time)
            
        except Exception as e:
            self.logger.error(f"Error publishing event {event_type}: {str(e)}")
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get event processing metrics"""
        return {
            'event_counts': dict(self._metrics.event_counts),
            'processing_times': dict(self._metrics.processing_times)
        }