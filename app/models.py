from __future__ import annotations

import time

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database import Base


def now_ts() -> float:
    return time.time()


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(32), unique=True, nullable=False, index=True)


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(160), nullable=False)
    rut = Column(String(64), nullable=True)
    codigo = Column(String(16), unique=True, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(Float, default=now_ts, nullable=False)

    users = relationship("User", back_populates="company")
    cameras = relationship("Camera", back_populates="company")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(120), nullable=False)
    apellido = Column(String(120), nullable=False)
    username = Column(String(255), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    salt = Column(String(64), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(Float, default=now_ts, nullable=False)

    role = relationship("Role")
    company = relationship("Company", back_populates="users")


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (UniqueConstraint("created_by_user_id", "serial_number", name="uq_camera_user_serial"),)

    id = Column(Integer, primary_key=True, index=True)
    serial_number = Column(String(120), nullable=False, index=True)
    src = Column(Text, nullable=False)
    location_description = Column(String(255), nullable=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    status = Column(String(32), default="offline", nullable=False)
    status_detail = Column(Text, nullable=True)
    created_at = Column(Float, default=now_ts, nullable=False)
    updated_at = Column(Float, default=now_ts, nullable=False)

    company = relationship("Company", back_populates="cameras")
    created_by = relationship("User")
    inference_config = relationship(
        "CameraInferenceConfig",
        back_populates="camera",
        uselist=False,
        cascade="all, delete-orphan",
    )


class CameraInferenceConfig(Base):
    __tablename__ = "camera_inference_configs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), unique=True, nullable=False, index=True)
    model_profile = Column(String(64), default="hnm_balanced", nullable=False)
    threshold = Column(Float, default=0.49, nullable=False)
    fps = Column(Float, default=10.0, nullable=False)
    imgsz = Column(Integer, default=960, nullable=False)
    pose_model = Column(String(255), default="yolo11s-pose.pt", nullable=False)
    alert_windows = Column(Integer, default=2, nullable=False)
    cooldown_seconds = Column(Float, default=30.0, nullable=False)
    overlay = Column(Boolean, default=False, nullable=False)
    updated_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_at = Column(Float, default=now_ts, nullable=False)

    camera = relationship("Camera", back_populates="inference_config")
    updated_by = relationship("User")


class OperatorInferenceConfig(Base):
    __tablename__ = "operator_inference_configs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    model_profile = Column(String(64), default="hnm_balanced", nullable=False)
    fps = Column(Float, default=10.0, nullable=False)
    imgsz = Column(Integer, default=960, nullable=False)
    pose_model = Column(String(255), default="yolo11s-pose.pt", nullable=False)
    alert_windows = Column(Integer, default=2, nullable=False)
    cooldown_seconds = Column(Float, default=30.0, nullable=False)
    updated_at = Column(Float, default=now_ts, nullable=False)

    user = relationship("User")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(128), unique=True, nullable=False, index=True)
    expires_at = Column(Float, nullable=False)
    used_at = Column(Float, nullable=True)
    created_at = Column(Float, default=now_ts, nullable=False)

    user = relationship("User")


class InferenceConfigAudit(Base):
    __tablename__ = "inference_config_audit"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    old_config = Column(JSON, nullable=True)
    new_config = Column(JSON, nullable=False)
    timestamp = Column(Float, default=now_ts, nullable=False)

    camera = relationship("Camera")
    user = relationship("User")


class AccessLog(Base):
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(Float, default=now_ts, nullable=False)

    user = relationship("User")


class CameraAction(Base):
    __tablename__ = "camera_actions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    action = Column(String(64), nullable=False)
    timestamp = Column(Float, default=now_ts, nullable=False)

    user = relationship("User")
    camera = relationship("Camera")


class AlertLog(Base):
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    prob = Column(Float, default=0.0, nullable=False)
    timestamp = Column(Float, default=now_ts, nullable=False)
    status = Column(String(32), default="pending", nullable=False)
    reviewed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_by_email = Column(String(255), nullable=True)
    review_timestamp = Column(Float, nullable=True)
    model_profile = Column(String(64), nullable=True)
    threshold = Column(Float, nullable=True)

    camera = relationship("Camera")
    reviewed_by = relationship("User")
