"""
🜂 AETHERION Long-Term RAG Memory
Vector-based memory with multimodal context and Oracle integration
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue, Range, GeoBoundingBox
)
import torch
from PIL import Image
import requests
from io import BytesIO
from core.oracle_engine import OracleEngine

@dataclass
class MemoryEntry:
    """Memory entry with multimodal context"""
    id: str
    timestamp: datetime
    content: str
    content_type: str  # text, image, audio, video, code, data
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    tags: List[str] = None
    source: str = None
    confidence: float = 1.0
    oracle_context: Optional[Dict[str, Any]] = None
    agent_context: Optional[Dict[str, Any]] = None
    emotional_context: Optional[Dict[str, Any]] = None

@dataclass
class MemoryQuery:
    """Memory query with filters and context"""
    query: str
    query_type: str  # semantic, exact, hybrid
    filters: Dict[str, Any] = None
    limit: int = 10
    threshold: float = 0.7
    include_metadata: bool = True
    temporal_range: Optional[Tuple[datetime, datetime]] = None

class RAGMemory:
    """
    🜂 Long-Term RAG Memory System
    Vector-based memory with multimodal context and Oracle integration
    """
    
    def __init__(self, oracle_engine: OracleEngine, collection_name: str = "aetherion_memory"):
        self.oracle_engine = oracle_engine
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Initialize Qdrant client
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        
        # Initialize collection
        self._initialize_collection()
        
        # Memory statistics
        self.stats = {
            "total_entries": 0,
            "text_entries": 0,
            "image_entries": 0,
            "audio_entries": 0,
            "code_entries": 0,
            "last_updated": datetime.now()
        }
        
        logging.info("🜂 RAG Memory System initialized")
    
    def _initialize_collection(self):
        """Initialize Qdrant collection with proper schema"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload indexes for efficient filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="content_type",
                    field_schema="keyword"
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="timestamp",
                    field_schema="datetime"
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="tags",
                    field_schema="keyword"
                )
                
                logging.info(f"🜂 Created Qdrant collection: {self.collection_name}")
            else:
                logging.info(f"🜂 Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant collection: {e}")
            raise
    
    def store_memory(self, content: str, content_type: str = "text", 
                    metadata: Dict[str, Any] = None, tags: List[str] = None,
                    source: str = None, confidence: float = 1.0,
                    oracle_context: Dict[str, Any] = None,
                    agent_context: Dict[str, Any] = None,
                    emotional_context: Dict[str, Any] = None) -> str:
        """
        Store a memory entry with multimodal context
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Generate embedding
        embedding = self._generate_embedding(content, content_type)
        
        # Create memory entry
        entry = MemoryEntry(
            id=memory_id,
            timestamp=timestamp,
            content=content,
            content_type=content_type,
            embedding=embedding,
            metadata=metadata or {},
            tags=tags or [],
            source=source,
            confidence=confidence,
            oracle_context=oracle_context,
            agent_context=agent_context,
            emotional_context=emotional_context
        )
        
        # Store in Qdrant
        self._store_in_qdrant(entry)
        
        # Update statistics
        self._update_stats(content_type)
        
        # Link to Oracle Engine if context provided
        if oracle_context:
            self._link_to_oracle(entry)
        
        logging.info(f"🜂 Memory stored: {memory_id} ({content_type})")
        return memory_id
    
    def _generate_embedding(self, content: str, content_type: str) -> List[float]:
        """Generate embedding for different content types"""
        if content_type == "text":
            return self.embedding_model.encode(content).tolist()
        elif content_type == "image":
            # For images, we'll use a text description or extract text
            # In a full implementation, you'd use a vision model here
            return self.embedding_model.encode(f"Image: {content}").tolist()
        elif content_type == "code":
            return self.embedding_model.encode(f"Code: {content}").tolist()
        else:
            return self.embedding_model.encode(content).tolist()
    
    def _store_in_qdrant(self, entry: MemoryEntry):
        """Store memory entry in Qdrant"""
        point = PointStruct(
            id=entry.id,
            vector=entry.embedding,
            payload={
                "content": entry.content,
                "content_type": entry.content_type,
                "timestamp": entry.timestamp.isoformat(),
                "metadata": entry.metadata,
                "tags": entry.tags,
                "source": entry.source,
                "confidence": entry.confidence,
                "oracle_context": entry.oracle_context,
                "agent_context": entry.agent_context,
                "emotional_context": entry.emotional_context
            }
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
    
    def _link_to_oracle(self, entry: MemoryEntry):
        """Link memory entry to Oracle Engine for timeline analysis"""
        if entry.oracle_context:
            self.oracle_engine.analyze_memory_context(
                memory_id=entry.id,
                content=entry.content,
                context=entry.oracle_context,
                timestamp=entry.timestamp
            )
    
    def query_memory(self, query: MemoryQuery) -> List[MemoryEntry]:
        """
        Query memory with semantic search and filters
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query.query).tolist()
        
        # Build filters
        filters = self._build_filters(query.filters, query.temporal_range)
        
        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filters,
            limit=query.limit,
            score_threshold=query.threshold,
            with_payload=True
        )
        
        # Convert to MemoryEntry objects
        entries = []
        for result in search_result:
            payload = result.payload
            entry = MemoryEntry(
                id=result.id,
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                content=payload["content"],
                content_type=payload["content_type"],
                embedding=result.vector,
                metadata=payload.get("metadata", {}),
                tags=payload.get("tags", []),
                source=payload.get("source"),
                confidence=payload.get("confidence", 1.0),
                oracle_context=payload.get("oracle_context"),
                agent_context=payload.get("agent_context"),
                emotional_context=payload.get("emotional_context")
            )
            entries.append(entry)
        
        return entries
    
    def _build_filters(self, filters: Dict[str, Any], 
                      temporal_range: Optional[Tuple[datetime, datetime]]) -> Optional[Filter]:
        """Build Qdrant filters from query parameters"""
        conditions = []
        
        if filters:
            for key, value in filters.items():
                if key == "content_type":
                    conditions.append(
                        FieldCondition(key="content_type", match=MatchValue(value=value))
                    )
                elif key == "tags":
                    conditions.append(
                        FieldCondition(key="tags", match=MatchValue(value=value))
                    )
                elif key == "source":
                    conditions.append(
                        FieldCondition(key="source", match=MatchValue(value=value))
                    )
                elif key == "confidence_min":
                    conditions.append(
                        FieldCondition(key="confidence", range=Range(gte=value))
                    )
        
        if temporal_range:
            start_time, end_time = temporal_range
            conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=Range(
                        gte=start_time.isoformat(),
                        lte=end_time.isoformat()
                    )
                )
            )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve memory entry by ID"""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True
            )
            
            if result:
                payload = result[0].payload
                return MemoryEntry(
                    id=result[0].id,
                    timestamp=datetime.fromisoformat(payload["timestamp"]),
                    content=payload["content"],
                    content_type=payload["content_type"],
                    embedding=result[0].vector,
                    metadata=payload.get("metadata", {}),
                    tags=payload.get("tags", []),
                    source=payload.get("source"),
                    confidence=payload.get("confidence", 1.0),
                    oracle_context=payload.get("oracle_context"),
                    agent_context=payload.get("agent_context"),
                    emotional_context=payload.get("emotional_context")
                )
        except Exception as e:
            logging.error(f"Failed to retrieve memory {memory_id}: {e}")
        
        return None
    
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory entry"""
        try:
            # Get current memory
            current = self.get_memory_by_id(memory_id)
            if not current:
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(current, key):
                    setattr(current, key, value)
            
            # Regenerate embedding if content changed
            if "content" in updates:
                current.embedding = self._generate_embedding(current.content, current.content_type)
            
            # Update timestamp
            current.timestamp = datetime.now()
            
            # Store updated entry
            self._store_in_qdrant(current)
            
            logging.info(f"🜂 Memory updated: {memory_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory entry"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )
            
            logging.info(f"🜂 Memory deleted: {memory_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            
            # Get content type distribution
            content_types = {}
            for content_type in ["text", "image", "audio", "code", "data"]:
                try:
                    result = self.client.count(
                        collection_name=self.collection_name,
                        count_filter=Filter(
                            must=[FieldCondition(key="content_type", match=MatchValue(value=content_type))]
                        )
                    )
                    content_types[content_type] = result.count
                except:
                    content_types[content_type] = 0
            
            return {
                "total_entries": total_points,
                "content_type_distribution": content_types,
                "collection_size_mb": collection_info.config.params.vectors.size * total_points * 4 / (1024 * 1024),
                "last_updated": self.stats["last_updated"].isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to get memory statistics: {e}")
            return self.stats
    
    def _update_stats(self, content_type: str):
        """Update memory statistics"""
        self.stats["total_entries"] += 1
        self.stats["last_updated"] = datetime.now()
        
        if content_type == "text":
            self.stats["text_entries"] += 1
        elif content_type == "image":
            self.stats["image_entries"] += 1
        elif content_type == "audio":
            self.stats["audio_entries"] += 1
        elif content_type == "code":
            self.stats["code_entries"] += 1
    
    def export_memory(self, format: str = "json", filters: Dict[str, Any] = None) -> str:
        """Export memory entries"""
        try:
            # Get all entries with filters
            query = MemoryQuery(
                query="",
                query_type="all",
                filters=filters,
                limit=10000  # Large limit for export
            )
            
            entries = self.query_memory(query)
            
            if format == "json":
                export_data = [asdict(entry) for entry in entries]
                export_file = f"aetherion_memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, default=str, indent=2)
                
                return export_file
            
            elif format == "csv":
                import csv
                export_file = f"aetherion_memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                with open(export_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "id", "timestamp", "content", "content_type", "tags", 
                        "source", "confidence"
                    ])
                    
                    for entry in entries:
                        writer.writerow([
                            entry.id, entry.timestamp, entry.content, 
                            entry.content_type, ",".join(entry.tags),
                            entry.source, entry.confidence
                        ])
                
                return export_file
            
        except Exception as e:
            logging.error(f"Failed to export memory: {e}")
            return None
    
    def import_memory(self, import_file: str) -> int:
        """Import memory entries from file"""
        try:
            imported_count = 0
            
            if import_file.endswith('.json'):
                with open(import_file, 'r') as f:
                    data = json.load(f)
                
                for item in data:
                    # Create memory entry from imported data
                    entry = MemoryEntry(
                        id=item.get("id", str(uuid.uuid4())),
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        content=item["content"],
                        content_type=item["content_type"],
                        metadata=item.get("metadata", {}),
                        tags=item.get("tags", []),
                        source=item.get("source"),
                        confidence=item.get("confidence", 1.0),
                        oracle_context=item.get("oracle_context"),
                        agent_context=item.get("agent_context"),
                        emotional_context=item.get("emotional_context")
                    )
                    
                    # Generate embedding
                    entry.embedding = self._generate_embedding(entry.content, entry.content_type)
                    
                    # Store in Qdrant
                    self._store_in_qdrant(entry)
                    imported_count += 1
            
            logging.info(f"🜂 Imported {imported_count} memory entries")
            return imported_count
            
        except Exception as e:
            logging.error(f"Failed to import memory: {e}")
            return 0 