from sqlalchemy import Column, String, Text, DateTime, Enum as SQLEnum, Numeric
from sqlalchemy.sql import func
from database import Base
import enum
import uuid

class JobStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SearchJob(Base):
    __tablename__ = "search_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text, nullable=False)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    result = Column(Text, nullable=True)  # JSON string of final result
    error = Column(Text, nullable=True)
    celery_task_id = Column(String, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "celery_task_id": self.celery_task_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

from sqlalchemy import create_engine, Column, String, Text, Boolean, DateTime, Integer, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime, timezone

# Create base
Base = declarative_base()


# Database Models
class User(Base):
    __tablename__ = "users"
    
    # Core Identity
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    
    # Authentication
    hashed_password = Column(String, nullable=True)
    
    # OAuth Data
    oauth_provider = Column(String, nullable=True)
    oauth_id = Column(String, nullable=True, index=True)
    
    # Google Profile
    google_name = Column(String, nullable=True)
    google_picture = Column(Text, nullable=True)
    
    # User Management
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")

    message_credits = Column(Integer, default=0, nullable=False)
    total_credits_purchased = Column(Integer, default=0, nullable=False)
    total_spent = Column(Numeric(10, 2), default=0.00, nullable=False)
    is_premium = Column(Boolean, default=False, nullable=False)
    last_purchase_date = Column(DateTime, nullable=True)
    
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    is_anonymous = Column(Boolean, default=False)
    message_count = Column(Integer, default=0)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    user = relationship("User", back_populates="conversations")

# class Message(Base):
#     __tablename__ = "messages"
    
#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
#     role = Column(String, nullable=False)
#     content = Column(Text, nullable=False)
#     has_file = Column(Boolean, default=False)
#     file_type = Column(String, nullable=True)
#     file_data = Column(Text, nullable=True)
#     created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
#     # Extended fields
#     sources = Column(Text, nullable=True)
#     reasoning_steps = Column(Text, nullable=True)
#     assets = Column(Text, nullable=True)
#     lab_mode = Column(Boolean, default=False)
#     app = Column(Text, nullable=True)
#     celery_task_id = Column(String, nullable=True, index=True)  # ✅ NEW: Track Celery tasks
    
#     # Relationships
#     conversation = relationship("Conversation", back_populates="messages")
#     reactions = relationship("Reaction", back_populates="message", cascade="all, delete-orphan")

class ReasoningStep(Base):
    __tablename__ = "reasoning_steps"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    query = Column(Text, nullable=True)
    category = Column(String, nullable=True)
    sources = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to Message
    message = relationship("Message", back_populates="reasoning_steps_rel")
    
    # Relationship back to Message
    message = relationship("Message", back_populates="reasoning_steps_rel")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    has_file = Column(Boolean, default=False)
    file_type = Column(String, nullable=True)
    file_data = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Metadata
    sources = Column(Text, nullable=True)  # JSON string
    reasoning_steps = Column(Text, nullable=True)  # JSON string (deprecated, use reasoning_steps_rel)
    assets = Column(Text, nullable=True)  # JSON string
    app = Column(Text, nullable=True)  # HTML/code content
    
    mode = Column(Text, default="normal")
    # Status tracking
    lab_mode = Column(Boolean, default=False)
    celery_task_id = Column(String, nullable=True, index=True)
    status = Column(String, default="complete")  # "streaming", "complete", "failed"
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    reactions = relationship("Reaction", back_populates="message", cascade="all, delete-orphan")
    reasoning_steps_rel = relationship(
        "ReasoningStep", 
        back_populates="message", 
        cascade="all, delete-orphan",
        order_by="ReasoningStep.step_number"
    )
    
class Reaction(Base):
    __tablename__ = "reactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    reaction_type = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    assets = Column(Text, nullable=True)
    lab_mode = Column(Boolean, default=False)
    
    # Relationships
    message = relationship("Message", back_populates="reactions")

class MagicLink(Base):
    __tablename__ = "magic_links"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, nullable=False, index=True)
    token = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    used_at = Column(DateTime(timezone=True), nullable=True)
    ip_address = Column(String, nullable=True)

class UserProject(Base):
    """
    Stores user projects with complete workflow state as JSONB
    """
    __tablename__ = 'user_projects'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Project metadata
    name = Column(String(255), nullable=False, default='Untitled Project')
    description = Column(Text, nullable=True)
    
    # Complete workflow state stored as JSONB
    workflow_data = Column(JSONB, nullable=False, server_default='{}')
    
    # Current workflow step
    current_step = Column(String(50), default='spec_parsing', server_default='spec_parsing')
    
    # Status tracking
    is_completed = Column(Boolean, default=False, server_default='false')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index('idx_user_projects_created_at', 'created_at'),
        Index('idx_user_projects_updated_at', 'updated_at'),
        Index('idx_user_projects_current_step', 'current_step'),
        Index('idx_user_projects_workflow_data', 'workflow_data', postgresql_using='gin'),
    )
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'workflow_data': self.workflow_data,
            'current_step': self.current_step,
            'is_completed': self.is_completed,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


from typing import List, Dict, Optional
from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"

class MessageCreate(BaseModel):
    content: str

class MessageSend(BaseModel):
    content: str
    conversation_id: Optional[str] = None
    deep_search: Optional[bool] = False    
    lab_mode: Optional[bool] = False  # NEW

class ReactionCreate(BaseModel):
    reaction_type: str

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    has_file: bool
    created_at: datetime
    reactions: List[Dict]

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

class NewsArticle(BaseModel):
    title: str
    description: Optional[str] = None
    link: str
    pubDate: str
    thumbnail: Optional[str] = None
    source: str
    author: Optional[str] = None

class NewsResponse(BaseModel):
    status: str
    category: str
    country: str
    articles: List[NewsArticle]
    cached_at: str
    next_update: str
    total_results: int
    
# At the top of your models.py or database file
from sqlalchemy import Column, Integer, String, DateTime, Date, UniqueConstraint
from datetime import datetime, date

class AnonymousUsage(Base):
    __tablename__ = "anonymous_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True, nullable=False)
    usage_date = Column(Date, index=True, nullable=False, default=date.today)
    message_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        # Unique constraint on IP + date combination
        UniqueConstraint('ip_address', 'usage_date', name='uix_ip_date'),
    )

class UserDailyUsage(Base):
    __tablename__ = "user_daily_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    usage_date = Column(Date, nullable=False, default=date.today, index=True)
    free_message_count = Column(Integer, default=0)  # Only for free tier
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('user_id', 'usage_date', name='uix_user_date'),
    )


class CreditPurchase(Base):
    __tablename__ = "credit_purchases"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Razorpay IDs
    razorpay_order_id = Column(String, unique=True, nullable=False, index=True)
    razorpay_payment_id = Column(String, unique=True, nullable=True, index=True)
    razorpay_signature = Column(String, nullable=True)
    
    # Purchase details
    package = Column(String, nullable=False)  # "starter", "popular", "pro"
    amount = Column(Numeric(10, 2), nullable=False)
    credits_purchased = Column(Integer, nullable=False)
    
    # Status
    payment_status = Column(String, default="created")  # created, paid, failed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    paid_at = Column(DateTime, nullable=True)
    
    # Metadata
    currency = Column(String, default="USD")
    receipt = Column(String, nullable=True)

class CreateOrderRequest(BaseModel):
    package: str

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    
# Database connection
DATABASE_URL = "postgresql://postgres:ciZHXAGUToWMnHlQEEogymCAZViEZnYE@hopper.proxy.rlwy.net:45046/railway"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

if __name__ == "__main__":
    print("🗑️  Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ Tables dropped")
    
    print("📝 Creating all tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created")
    
    print("\n✅ Migration complete!")
    print("\nCreated tables:")
    print("  - users")
    print("  - conversations")
    print("  - messages (with celery_task_id)")  # ✅ NEW
    print("  - reactions")
    print("  - magic_links")
    print("  - user_projects")