"""
Standalone Worker Process - Concurrent Version
Polls database for pending user messages and processes them concurrently.
Run with: python worker.py
"""

import asyncio
import json
import uuid
import signal
import sys
from datetime import datetime, timezone
from typing import Optional, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WORKER] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import your existing modules
from models import Message, Conversation, Base, engine, SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import and_

# Worker configuration
POLL_INTERVAL_SECONDS = 2  # How often to check for new jobs
TASK_TIMEOUT_SECONDS = 15 * 60  # 15 minutes max per task
MAX_CONCURRENT_TASKS = 5  # Max simultaneous jobs
SHUTDOWN_REQUESTED = False

# Track active tasks
active_tasks: Dict[str, asyncio.Task] = {}


def get_db():
    """Get database session"""
    return SessionLocal()


def claim_pending_message(db: Session, exclude_ids: list) -> Optional[Message]:
    """
    Find and claim the oldest pending user message.
    Excludes messages already being processed.
    Returns None if no pending messages.
    """
    try:
        # Find oldest pending user message not already being processed
        query = db.query(Message).filter(
            and_(
                Message.role == "user",
                Message.status == "pending"
            )
        )
        
        # Exclude messages already being processed
        if exclude_ids:
            query = query.filter(~Message.id.in_(exclude_ids))
        
        pending_msg = query.order_by(Message.created_at.asc()).first()
        
        if not pending_msg:
            return None
        
        # Claim it by setting status to processing
        pending_msg.status = "processing"
        db.commit()
        db.refresh(pending_msg)
        
        logger.info(f"Claimed message {pending_msg.id} for processing")
        return pending_msg
        
    except Exception as e:
        logger.error(f"Error claiming message: {e}")
        db.rollback()
        return None


def get_conversation_context(db: Session, conversation_id: uuid.UUID, exclude_msg_id: uuid.UUID):
    """Get conversation history for context"""
    messages = db.query(Message).filter(
        and_(
            Message.conversation_id == conversation_id,
            Message.id != exclude_msg_id,
            Message.status == "complete"
        )
    ).order_by(Message.created_at.asc()).all()
    
    messages_list = [
        {"role": msg.role, "content": msg.content} 
        for msg in messages 
        if msg.content and len(msg.content) > 0
    ]
    
    # Get previous app if exists
    apps = [msg.app for msg in messages if msg.app and len(msg.app) > 0]
    app_prev = apps[-1] if apps else None
    
    return messages_list, app_prev


async def process_message(user_msg_id: uuid.UUID):
    """
    Process a single user message - runs deep_search or lab mode.
    Uses its own DB session for isolation.
    """
    
    db = get_db()
    
    try:
        # Re-fetch message in this session
        user_msg = db.query(Message).filter(Message.id == user_msg_id).first()
        
        if not user_msg:
            logger.error(f"Message {user_msg_id} not found")
            return
        
        conversation_id = user_msg.conversation_id
        user_prompt = user_msg.content
        mode = user_msg.mode or "normal"
        
        is_deep_search = mode == "deep_search"
        is_lab_mode = mode == "lab"
        
        logger.info(f"Processing message {user_msg.id} in {mode} mode")
        
        # Get conversation context
        messages_list, app_prev = get_conversation_context(db, conversation_id, user_msg.id)
        
        # Get conversation for updates
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        
        if not conv:
            logger.error(f"Conversation {conversation_id} not found")
            user_msg.status = "failed"
            db.commit()
            return
        
        # Find existing assistant message (created by API) or create new one
        assistant_msg = db.query(Message).filter(
            and_(
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
                Message.status == "streaming"
            )
        ).order_by(Message.created_at.desc()).first()
        
        if not assistant_msg:
            # Create assistant message if not exists
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content="",
                sources=None,
                reasoning_steps=None,
                assets=None,
                lab_mode=is_lab_mode,
                mode=mode,
                status="streaming",
                app=None
            )
            db.add(assistant_msg)
            db.commit()
            db.refresh(assistant_msg)
        
        logger.info(f"Using assistant message {assistant_msg.id}")
        
        # Processing state
        reasoning_steps = []
        full_response = ""
        final_sources = []
        app = None
        
        # Helper functions to save state
        async def save_content(text):
            nonlocal full_response
            full_response += text
            assistant_msg.content = full_response
            db.commit()
        
        async def save_reasoning_step(step):
            reasoning_steps.append(step)
            assistant_msg.reasoning_steps = json.dumps(reasoning_steps)
            db.commit()
        
        try:
            # Import Claude client
            from simple_search_claude_streaming_with_web_search import ClaudeConversation
            claude_client = ClaudeConversation(messages=messages_list)
            
            if is_deep_search:
                from deep_search_with_claude import MarkdownResearch
                deep_research = MarkdownResearch(claude_client)
                
                async for chunk in deep_research.research(user_prompt, files=None, existing_markdown=app_prev):
                    if chunk["type"] == "thinking":
                        logger.debug(f"Thinking...")
                        
                    elif chunk["type"] == "content":
                        print(chunk["text"], end="", flush=True)
                        await save_content(chunk["text"])
                        
                    elif chunk["type"] == "reasoning":
                        step = {
                            "type": "reasoning",
                            "step": "Reasoning...",
                            "content": chunk["text"],
                            "found_sources": None,
                            "sources": None,
                            "query": chunk["text"],
                            "category": "Reasoning",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        await save_reasoning_step(step)
                        
                    elif chunk["type"] == "search_query":
                        query = chunk['text']
                        data = await claude_client.google_search(query)
                        urls = [item["url"] for item in data]
                        
                        step = {
                            "type": "reasoning",
                            "step": "Sources Found",
                            "content": query,
                            "found_sources": len(urls),
                            "sources": urls,
                            "query": query,
                            "category": "Web Search",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        await save_reasoning_step(step)
                        final_sources.append(urls)
                        
                    elif chunk["type"] == "markdown_report":
                        app = chunk["content"]
                        assistant_msg.app = app
                        db.commit()
                        
                    elif chunk["type"] == "html_app":
                        app = chunk["content"]
                        assistant_msg.app = app
                        db.commit()
                        
                    elif chunk["type"] == "research_summary":
                        await save_content(chunk["content"])
                        
            elif is_lab_mode:
                from lab_with_claude import DeepResearch
                deep_research = DeepResearch(claude_client)
                
                async for chunk in deep_research.research(user_prompt, files=None, existing_html=app_prev):
                    if chunk["type"] == "thinking":
                        logger.debug(f"Thinking...")
                        
                    elif chunk["type"] == "content":
                        print(chunk["text"], end="", flush=True)
                        await save_content(chunk["text"])
                        
                    elif chunk["type"] == "reasoning":
                        step = {
                            "type": "reasoning",
                            "step": "Reasoning...",
                            "content": chunk["text"],
                            "found_sources": None,
                            "sources": None,
                            "query": chunk["text"],
                            "category": "Reasoning",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        await save_reasoning_step(step)
                        
                    elif chunk["type"] == "search_query":
                        query = chunk['text']
                        data = await claude_client.google_search(query)
                        urls = [item["url"] for item in data]
                        
                        step = {
                            "type": "reasoning",
                            "step": "Sources Found",
                            "content": query,
                            "found_sources": len(urls),
                            "sources": urls,
                            "query": query,
                            "category": "Web Search",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        await save_reasoning_step(step)
                        final_sources.append(urls)
                        
                    elif chunk["type"] == "report":
                        app = chunk["content"]
                        assistant_msg.app = app
                        db.commit()
                        
                    elif chunk["type"] == "html_app":
                        app = chunk["content"]
                        assistant_msg.app = app
                        db.commit()
                        
                    elif chunk["type"] == "research_summary":
                        await save_content(chunk["content"])
            
            # Finalize assistant message
            assistant_msg.content = full_response
            assistant_msg.sources = json.dumps(final_sources) if final_sources else None
            assistant_msg.reasoning_steps = json.dumps(reasoning_steps) if reasoning_steps else None
            assistant_msg.app = app
            assistant_msg.status = "complete"
            
            # Update user message as processed
            user_msg.status = "complete"
            
            # Update conversation
            conv.updated_at = datetime.now(timezone.utc)
            conv.message_count = (conv.message_count or 0) + 2
            if conv.title == "New Conversation":
                conv.title = user_prompt[:50] + ("..." if len(user_prompt) > 50 else "")
            
            db.commit()
            
            logger.info(f"Successfully processed message {user_msg.id}")
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing message {user_msg.id}")
            assistant_msg.status = "timeout"
            assistant_msg.content = full_response or "Processing timed out after 15 minutes"
            user_msg.status = "complete"
            db.commit()
            
        except Exception as e:
            logger.error(f"Error processing message {user_msg.id}: {e}", exc_info=True)
            assistant_msg.status = "failed"
            assistant_msg.content = full_response or "Processing failed"
            user_msg.status = "failed"
            db.commit()
            
    finally:
        db.close()


async def task_wrapper(user_msg_id: uuid.UUID, task_key: str):
    """Wrapper to handle task completion and cleanup"""
    try:
        await asyncio.wait_for(
            process_message(user_msg_id),
            timeout=TASK_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error(f"Task timed out for message {user_msg_id}")
        # Update status in DB
        db = get_db()
        try:
            user_msg = db.query(Message).filter(Message.id == user_msg_id).first()
            if user_msg:
                user_msg.status = "failed"
                db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Task failed for message {user_msg_id}: {e}", exc_info=True)
    finally:
        # Remove from active tasks
        if task_key in active_tasks:
            del active_tasks[task_key]
            logger.info(f"Task {task_key} completed. Active tasks: {len(active_tasks)}")


async def worker_loop():
    """Main worker loop - polls for and processes pending messages concurrently"""
    
    logger.info("Worker started, polling for pending messages...")
    logger.info(f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}")
    
    while not SHUTDOWN_REQUESTED:
        try:
            # Check if we can accept more tasks
            if len(active_tasks) >= MAX_CONCURRENT_TASKS:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue
            
            # Get IDs of messages already being processed
            processing_ids = list(active_tasks.keys())
            processing_uuids = [uuid.UUID(id) for id in processing_ids]
            
            # Try to claim a pending message
            db = get_db()
            try:
                user_msg = claim_pending_message(db, processing_uuids)
                
                if user_msg:
                    task_key = str(user_msg.id)
                    
                    # Spawn task
                    task = asyncio.create_task(
                        task_wrapper(user_msg.id, task_key)
                    )
                    active_tasks[task_key] = task
                    
                    logger.info(f"Spawned task for message {user_msg.id}. Active tasks: {len(active_tasks)}")
                else:
                    # No pending messages, wait before polling again
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in worker loop: {e}", exc_info=True)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    
    # Graceful shutdown - wait for active tasks
    if active_tasks:
        logger.info(f"Waiting for {len(active_tasks)} active tasks to complete...")
        await asyncio.gather(*active_tasks.values(), return_exceptions=True)
    
    logger.info("Worker shutdown complete")


def handle_shutdown(signum, frame):
    """Handle graceful shutdown"""
    global SHUTDOWN_REQUESTED
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    SHUTDOWN_REQUESTED = True


def main():
    """Entry point"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    logger.info("=" * 50)
    logger.info("DEEPSHIP WORKER STARTING (CONCURRENT)")
    logger.info(f"Poll interval: {POLL_INTERVAL_SECONDS}s")
    logger.info(f"Task timeout: {TASK_TIMEOUT_SECONDS // 60} minutes")
    logger.info(f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}")
    logger.info("=" * 50)
    
    # Run the worker loop
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()