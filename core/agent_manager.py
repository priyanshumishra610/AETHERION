"""
🜂 AETHERION Autonomous Multi-Agent Orchestration
LangChain-style Task Chains with Celery Async Orchestration
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio
from celery import Celery, Task
from celery.result import AsyncResult
import redis
from core.keeper_seal import KeeperSeal
from core.rag_memory import RAGMemory

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    APPROVED = "approved"
    REJECTED = "rejected"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentTask:
    """Agent task definition"""
    id: str
    name: str
    description: str
    agent_type: str
    parameters: Dict[str, Any]
    priority: TaskPriority
    dependencies: List[str] = None
    timeout: int = 300  # seconds
    retry_count: int = 3
    keeper_approval_required: bool = False
    created_at: datetime = None
    scheduled_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

@dataclass
class TaskChain:
    """Task chain definition"""
    id: str
    name: str
    description: str
    tasks: List[AgentTask]
    workflow: Dict[str, List[str]]  # task_id -> [dependent_task_ids]
    keeper_id: str
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING
    completed_tasks: List[str] = None
    failed_tasks: List[str] = None

@dataclass
class Agent:
    """Agent definition"""
    id: str
    name: str
    agent_type: str
    capabilities: List[str]
    status: str = "active"
    current_task: Optional[str] = None
    task_history: List[str] = None
    performance_metrics: Dict[str, Any] = None

class AgentManager:
    """
    🜂 Autonomous Multi-Agent Orchestration System
    Manages task chains, agent allocation, and async execution
    """
    
    def __init__(self, keeper_seal: KeeperSeal, rag_memory: RAGMemory):
        self.keeper_seal = keeper_seal
        self.rag_memory = rag_memory
        
        # Initialize Celery
        self.celery_app = Celery(
            'aetherion_agents',
            broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        )
        
        # Configure Celery
        self.celery_app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=3600,
            task_soft_time_limit=3000,
            worker_prefetch_multiplier=1,
            task_acks_late=True
        )
        
        # Initialize Redis for task coordination
        self.redis_client = redis.Redis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        )
        
        # Agent registry
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[AgentTask] = []
        self.task_chains: Dict[str, TaskChain] = {}
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, AgentTask] = {}
        
        # Load existing agents and tasks
        self._load_agents()
        self._load_task_chains()
        
        # Register task handlers
        self._register_task_handlers()
        
        logging.info("🜂 Agent Manager initialized")
    
    def _load_agents(self):
        """Load agent definitions from storage"""
        agents_file = Path("aetherion_agents.json")
        if agents_file.exists():
            try:
                with open(agents_file, 'r') as f:
                    data = json.load(f)
                    for agent_data in data:
                        agent = Agent(**agent_data)
                        self.agents[agent.id] = agent
            except Exception as e:
                logging.error(f"Failed to load agents: {e}")
    
    def _save_agents(self):
        """Save agent definitions to storage"""
        agents_file = Path("aetherion_agents.json")
        try:
            with open(agents_file, 'w') as f:
                json.dump([asdict(agent) for agent in self.agents.values()], f, 
                         default=str, indent=2)
        except Exception as e:
            logging.error(f"Failed to save agents: {e}")
    
    def _load_task_chains(self):
        """Load task chains from storage"""
        chains_file = Path("aetherion_task_chains.json")
        if chains_file.exists():
            try:
                with open(chains_file, 'r') as f:
                    data = json.load(f)
                    for chain_data in data:
                        # Reconstruct tasks and chain
                        tasks = [AgentTask(**task_data) for task_data in chain_data.get("tasks", [])]
                        chain = TaskChain(
                            id=chain_data["id"],
                            name=chain_data["name"],
                            description=chain_data["description"],
                            tasks=tasks,
                            workflow=chain_data["workflow"],
                            keeper_id=chain_data["keeper_id"],
                            created_at=datetime.fromisoformat(chain_data["created_at"]),
                            status=TaskStatus(chain_data["status"]),
                            completed_tasks=chain_data.get("completed_tasks", []),
                            failed_tasks=chain_data.get("failed_tasks", [])
                        )
                        self.task_chains[chain.id] = chain
            except Exception as e:
                logging.error(f"Failed to load task chains: {e}")
    
    def _save_task_chains(self):
        """Save task chains to storage"""
        chains_file = Path("aetherion_task_chains.json")
        try:
            with open(chains_file, 'w') as f:
                json.dump([asdict(chain) for chain in self.task_chains.values()], f, 
                         default=str, indent=2)
        except Exception as e:
            logging.error(f"Failed to save task chains: {e}")
    
    def _register_task_handlers(self):
        """Register Celery task handlers"""
        
        @self.celery_app.task(bind=True)
        def execute_agent_task(self, task_data: Dict[str, Any]):
            """Execute agent task with error handling"""
            try:
                task_id = task_data["id"]
                agent_type = task_data["agent_type"]
                parameters = task_data["parameters"]
                
                # Execute based on agent type
                if agent_type == "nlp_agent":
                    result = self._execute_nlp_task(parameters)
                elif agent_type == "math_agent":
                    result = self._execute_math_task(parameters)
                elif agent_type == "research_agent":
                    result = self._execute_research_task(parameters)
                elif agent_type == "creative_agent":
                    result = self._execute_creative_task(parameters)
                elif agent_type == "analysis_agent":
                    result = self._execute_analysis_task(parameters)
                else:
                    result = {"error": f"Unknown agent type: {agent_type}"}
                
                return {
                    "task_id": task_id,
                    "success": "error" not in result,
                    "result": result,
                    "execution_time": self.request.execution_time
                }
                
            except Exception as e:
                return {
                    "task_id": task_data.get("id", "unknown"),
                    "success": False,
                    "error": str(e),
                    "execution_time": self.request.execution_time
                }
        
        self.execute_agent_task = execute_agent_task
    
    def register_agent(self, agent_id: str, name: str, agent_type: str, 
                      capabilities: List[str]) -> bool:
        """Register a new agent"""
        if agent_id in self.agents:
            logging.warning(f"Agent {agent_id} already exists")
            return False
        
        agent = Agent(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            capabilities=capabilities,
            task_history=[],
            performance_metrics={
                "tasks_completed": 0,
                "tasks_failed": 0,
                "average_execution_time": 0.0,
                "success_rate": 1.0
            }
        )
        
        self.agents[agent_id] = agent
        self._save_agents()
        
        logging.info(f"🜂 Agent registered: {agent_id} ({agent_type})")
        return True
    
    def create_task(self, name: str, description: str, agent_type: str,
                   parameters: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL,
                   dependencies: List[str] = None, keeper_approval_required: bool = False) -> str:
        """Create a new agent task"""
        task_id = str(uuid.uuid4())
        
        task = AgentTask(
            id=task_id,
            name=name,
            description=description,
            agent_type=agent_type,
            parameters=parameters,
            priority=priority,
            dependencies=dependencies or [],
            keeper_approval_required=keeper_approval_required,
            created_at=datetime.now()
        )
        
        self.task_queue.append(task)
        self._sort_task_queue()
        
        logging.info(f"🜂 Task created: {task_id} ({agent_type})")
        return task_id
    
    def create_task_chain(self, name: str, description: str, tasks: List[Dict[str, Any]],
                         workflow: Dict[str, List[str]], keeper_id: str) -> str:
        """Create a task chain"""
        chain_id = str(uuid.uuid4())
        
        # Create AgentTask objects
        agent_tasks = []
        for task_data in tasks:
            task = AgentTask(
                id=str(uuid.uuid4()),
                name=task_data["name"],
                description=task_data["description"],
                agent_type=task_data["agent_type"],
                parameters=task_data["parameters"],
                priority=TaskPriority(task_data.get("priority", 2)),
                dependencies=task_data.get("dependencies", []),
                keeper_approval_required=task_data.get("keeper_approval_required", False),
                created_at=datetime.now()
            )
            agent_tasks.append(task)
        
        chain = TaskChain(
            id=chain_id,
            name=name,
            description=description,
            tasks=agent_tasks,
            workflow=workflow,
            keeper_id=keeper_id,
            created_at=datetime.now(),
            completed_tasks=[],
            failed_tasks=[]
        )
        
        self.task_chains[chain_id] = chain
        self._save_task_chains()
        
        logging.info(f"🜂 Task chain created: {chain_id}")
        return chain_id
    
    def submit_task_chain(self, chain_id: str) -> bool:
        """Submit a task chain for execution"""
        chain = self.task_chains.get(chain_id)
        if not chain:
            return False
        
        # Check if keeper approval is required
        if any(task.keeper_approval_required for task in chain.tasks):
            if not self.keeper_seal.verify_keeper_authority(chain.keeper_id, "task_approval"):
                logging.error(f"Keeper approval required for chain {chain_id}")
                return False
        
        # Add tasks to queue
        for task in chain.tasks:
            self.task_queue.append(task)
        
        chain.status = TaskStatus.RUNNING
        self._sort_task_queue()
        self._save_task_chains()
        
        # Start execution
        self._process_task_queue()
        
        logging.info(f"🜂 Task chain submitted: {chain_id}")
        return True
    
    def _sort_task_queue(self):
        """Sort task queue by priority and dependencies"""
        # Sort by priority (highest first)
        self.task_queue.sort(key=lambda x: x.priority.value, reverse=True)
        
        # TODO: Implement dependency-based sorting
        # For now, just priority-based sorting
    
    def _process_task_queue(self):
        """Process the task queue and assign tasks to agents"""
        available_agents = [
            agent for agent in self.agents.values() 
            if agent.status == "active" and agent.current_task is None
        ]
        
        for task in self.task_queue[:]:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Find suitable agent
            suitable_agent = self._find_suitable_agent(task, available_agents)
            if suitable_agent:
                # Assign task to agent
                self._assign_task_to_agent(task, suitable_agent)
                available_agents.remove(suitable_agent)
                self.task_queue.remove(task)
    
    def _find_suitable_agent(self, task: AgentTask, available_agents: List[Agent]) -> Optional[Agent]:
        """Find a suitable agent for a task"""
        for agent in available_agents:
            if (agent.agent_type == task.agent_type or 
                task.agent_type in agent.capabilities):
                return agent
        return None
    
    def _assign_task_to_agent(self, task: AgentTask, agent: Agent):
        """Assign a task to an agent and start execution"""
        task.status = TaskStatus.RUNNING
        agent.current_task = task.id
        self.active_tasks[task.id] = task
        
        # Submit to Celery
        async_result = self.execute_agent_task.delay(asdict(task))
        
        # Store async result reference
        self.redis_client.set(f"task_result:{task.id}", async_result.id)
        
        logging.info(f"🜂 Task {task.id} assigned to agent {agent.id}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and result"""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            async_result_id = self.redis_client.get(f"task_result:{task_id}")
            
            if async_result_id:
                async_result = AsyncResult(async_result_id.decode(), app=self.celery_app)
                if async_result.ready():
                    result = async_result.get()
                    task.status = TaskStatus.COMPLETED if result["success"] else TaskStatus.FAILED
                    task.result = result.get("result")
                    task.error = result.get("error")
                    task.execution_time = result.get("execution_time")
                    
                    # Move to completed tasks
                    self.completed_tasks[task_id] = task
                    del self.active_tasks[task_id]
                    
                    # Update agent
                    for agent in self.agents.values():
                        if agent.current_task == task_id:
                            agent.current_task = None
                            agent.task_history.append(task_id)
                            self._update_agent_metrics(agent, result["success"])
                            break
            
            return {
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "agent_type": task.agent_type,
                "created_at": task.created_at.isoformat(),
                "result": task.result,
                "error": task.error,
                "execution_time": task.execution_time
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "agent_type": task.agent_type,
                "created_at": task.created_at.isoformat(),
                "result": task.result,
                "error": task.error,
                "execution_time": task.execution_time
            }
        
        return None
    
    def _update_agent_metrics(self, agent: Agent, success: bool):
        """Update agent performance metrics"""
        metrics = agent.performance_metrics
        
        if success:
            metrics["tasks_completed"] += 1
        else:
            metrics["tasks_failed"] += 1
        
        total_tasks = metrics["tasks_completed"] + metrics["tasks_failed"]
        if total_tasks > 0:
            metrics["success_rate"] = metrics["tasks_completed"] / total_tasks
        
        self._save_agents()
    
    def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get status of all agents"""
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "current_task": agent.current_task,
                "capabilities": agent.capabilities,
                "performance_metrics": agent.performance_metrics
            }
            for agent in self.agents.values()
        ]
    
    def get_task_chain_status(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get task chain status"""
        chain = self.task_chains.get(chain_id)
        if not chain:
            return None
        
        # Get status of all tasks in chain
        task_statuses = []
        for task in chain.tasks:
            status = self.get_task_status(task.id)
            if status:
                task_statuses.append(status)
        
        return {
            "id": chain.id,
            "name": chain.name,
            "description": chain.description,
            "status": chain.status.value,
            "created_at": chain.created_at.isoformat(),
            "keeper_id": chain.keeper_id,
            "tasks": task_statuses,
            "completed_tasks": chain.completed_tasks,
            "failed_tasks": chain.failed_tasks
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            # Revoke Celery task
            async_result_id = self.redis_client.get(f"task_result:{task_id}")
            if async_result_id:
                async_result = AsyncResult(async_result_id.decode(), app=self.celery_app)
                async_result.revoke(terminate=True)
            
            # Update agent
            for agent in self.agents.values():
                if agent.current_task == task_id:
                    agent.current_task = None
                    break
            
            del self.active_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            logging.info(f"🜂 Task cancelled: {task_id}")
            return True
        
        return False
    
    def _execute_nlp_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NLP agent task"""
        # This would integrate with the NLP plugin
        return {"type": "nlp", "parameters": parameters, "status": "executed"}
    
    def _execute_math_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute math agent task"""
        # This would integrate with the math plugin
        return {"type": "math", "parameters": parameters, "status": "executed"}
    
    def _execute_research_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research agent task"""
        # This would integrate with knowledge retrieval
        return {"type": "research", "parameters": parameters, "status": "executed"}
    
    def _execute_creative_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute creative agent task"""
        # This would integrate with creative generation
        return {"type": "creative", "parameters": parameters, "status": "executed"}
    
    def _execute_analysis_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis agent task"""
        # This would integrate with data analysis
        return {"type": "analysis", "parameters": parameters, "status": "executed"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status == "active"]),
            "queued_tasks": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_task_chains": len(self.task_chains),
            "running_chains": len([c for c in self.task_chains.values() if c.status == TaskStatus.RUNNING])
        } 