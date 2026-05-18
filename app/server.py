from __future__ import annotations

import hashlib
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.email_notifications import send_alert_email, send_email, unique_recipients
from app.inference_runtime import SESSION_STATUS, stream_camera
from app.model_profiles import MODEL_PROFILES, config_from_profile, default_config, list_profiles, profile_available
from app.models import (
    AccessLog,
    AlertLog,
    Camera,
    CameraAction,
    CameraInferenceConfig,
    Company,
    InferenceConfigAudit,
    OperatorInferenceConfig,
    PasswordResetToken,
    Role,
    User,
    now_ts,
)
from app.utils import hash_password, make_salt, verify_password


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"
GLOBAL_STATE = {"overlay": False, "threshold": 0.49}

app = FastAPI(title="Sicher Aggression Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RegisterRequest(BaseModel):
    nombre: str
    apellido: str
    email: str
    password: str
    charge: str = "operador"
    codigo: str | None = None
    empresa_nombre: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirm(BaseModel):
    token: str
    password: str = Field(min_length=6)


class CameraRequest(BaseModel):
    serial_number: str
    src: str
    location_description: str | None = None


class CameraStatusRequest(BaseModel):
    status: str
    detail: str | None = None


class AlertRequest(BaseModel):
    serial_number: str
    prob: float = 0.0
    timestamp: float | None = None
    status: str = "pending"
    reviewer_email: str | None = None
    model_profile: str | None = None
    threshold: float | None = None


class CompanyRequest(BaseModel):
    name: str
    rut: str | None = None


class UserActivePatch(BaseModel):
    is_active: bool


class InferenceConfigPatch(BaseModel):
    model_profile: str | None = None
    threshold: float | None = Field(default=None, ge=0.10, le=0.95)
    fps: float | None = Field(default=None, ge=1.0, le=30.0)
    imgsz: int | None = Field(default=None, ge=320, le=1920)
    pose_model: str | None = None
    alert_windows: int | None = Field(default=None, ge=1, le=10)
    cooldown_seconds: float | None = Field(default=None, ge=0.0, le=3600.0)
    overlay: bool | None = None


class OperatorInferenceConfigPatch(BaseModel):
    model_profile: str


class OverlayRequest(BaseModel):
    overlay: bool


class ThresholdRequest(BaseModel):
    thr_on: float = Field(ge=0.10, le=0.95)
    thr_off: float | None = None


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_roles(db: Session) -> None:
    current = {role.name for role in db.query(Role).all()}
    for role_name in ("jefe", "operador"):
        if role_name not in current:
            db.add(Role(name=role_name))
    db.commit()


def migrate_sqlite_camera_uniqueness() -> None:
    """Move SQLite camera uniqueness from company+serial to user+serial.

    Operators own their camera list. A jefe can still inspect company cameras,
    but one operator should not inherit another operator's cameras.
    """
    if engine.url.get_backend_name() != "sqlite":
        return

    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cameras'")
        if not cur.fetchone():
            return

        def unique_index_columns() -> list[list[str]]:
            indexes: list[list[str]] = []
            cur.execute("PRAGMA index_list('cameras')")
            for row in cur.fetchall():
                index_name = str(row[1])
                is_unique = int(row[2]) == 1
                if not is_unique:
                    continue
                safe_index = index_name.replace("'", "''")
                cur.execute(f"PRAGMA index_info('{safe_index}')")
                indexes.append([str(info[2]) for info in cur.fetchall()])
            return indexes

        unique_columns = unique_index_columns()
        has_user_serial = ["created_by_user_id", "serial_number"] in unique_columns
        has_company_serial = ["company_id", "serial_number"] in unique_columns
        if has_user_serial or not has_company_serial:
            return

        cur.execute("PRAGMA foreign_keys=OFF")
        cur.execute(
            """
            CREATE TABLE cameras_new (
                id INTEGER NOT NULL,
                serial_number VARCHAR(120) NOT NULL,
                src TEXT NOT NULL,
                location_description VARCHAR(255),
                company_id INTEGER,
                created_by_user_id INTEGER,
                is_active BOOLEAN NOT NULL,
                status VARCHAR(32) NOT NULL,
                status_detail TEXT,
                created_at FLOAT NOT NULL,
                updated_at FLOAT NOT NULL,
                PRIMARY KEY (id),
                CONSTRAINT uq_camera_user_serial UNIQUE (created_by_user_id, serial_number),
                FOREIGN KEY(company_id) REFERENCES companies (id),
                FOREIGN KEY(created_by_user_id) REFERENCES users (id)
            )
            """
        )
        cur.execute(
            """
            INSERT INTO cameras_new (
                id, serial_number, src, location_description, company_id,
                created_by_user_id, is_active, status, status_detail, created_at, updated_at
            )
            SELECT
                id, serial_number, src, location_description, company_id,
                created_by_user_id, is_active, status, status_detail, created_at, updated_at
            FROM cameras
            """
        )
        cur.execute("DROP TABLE cameras")
        cur.execute("ALTER TABLE cameras_new RENAME TO cameras")
        cur.execute("CREATE INDEX ix_cameras_id ON cameras (id)")
        cur.execute("CREATE INDEX ix_cameras_serial_number ON cameras (serial_number)")
        cur.execute("CREATE INDEX ix_cameras_company_id ON cameras (company_id)")
        raw.commit()
        cur.execute("PRAGMA foreign_keys=ON")
        raw.commit()
    finally:
        raw.close()


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    migrate_sqlite_camera_uniqueness()
    db = SessionLocal()
    try:
        ensure_roles(db)
    finally:
        db.close()


def user_role(user: User | None) -> str:
    if not user or not user.role:
        return ""
    return str(user.role.name)


def current_user_optional(
    db: Session,
    x_user_email: str | None = Header(default=None, alias="X-User-Email"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> User | None:
    if x_user_id and str(x_user_id).isdigit():
        user = db.query(User).filter(User.id == int(x_user_id), User.is_active.is_(True)).first()
        if user:
            return user
    if x_user_email:
        return db.query(User).filter(User.username == x_user_email, User.is_active.is_(True)).first()
    return None


def current_user(
    db: Session = Depends(get_db),
    x_user_email: str | None = Header(default=None, alias="X-User-Email"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> User:
    user = current_user_optional(db, x_user_email, x_user_id)
    if not user:
        raise HTTPException(status_code=401, detail={"error": "No autenticado"})
    return user


def camera_query_for_user(db: Session, user: User | None, include_inactive: bool = False):
    query = db.query(Camera)
    if user:
        role_name = user_role(user)
        if role_name == "operador":
            query = query.filter(Camera.created_by_user_id == user.id)
        elif user.company_id is not None:
            query = query.filter(Camera.company_id == user.company_id)
    if not include_inactive:
        query = query.filter(Camera.is_active.is_(True))
    return query


def get_camera_by_serial(db: Session, serial_number: str, user: User | None = None, include_inactive: bool = False) -> Camera | None:
    query = camera_query_for_user(db, user, include_inactive=include_inactive)
    return query.filter(Camera.serial_number == serial_number).first()


def config_to_dict(config: CameraInferenceConfig) -> dict[str, Any]:
    profile_id = str(config.model_profile or "hnm_balanced")
    if profile_id not in MODEL_PROFILES:
        profile_id = "hnm_balanced"
    return {
        "model_profile": profile_id,
        "threshold": float(config.threshold),
        "fps": float(config.fps),
        "imgsz": int(config.imgsz),
        "pose_model": config.pose_model,
        "feature_version": MODEL_PROFILES[profile_id].get("feature_version", "v1"),
        "smoothing_windows": int(MODEL_PROFILES[profile_id].get("smoothing_windows", 3)),
        "alert_windows": int(config.alert_windows),
        "cooldown_seconds": float(config.cooldown_seconds),
        "overlay": bool(config.overlay),
        "updated_at": float(config.updated_at),
    }


def apply_config_dict(config: CameraInferenceConfig, values: dict[str, Any], user: User | None = None) -> None:
    config.model_profile = str(values["model_profile"])
    config.threshold = float(values["threshold"])
    config.fps = float(values["fps"])
    config.imgsz = int(values["imgsz"])
    config.pose_model = str(values["pose_model"])
    config.alert_windows = int(values["alert_windows"])
    config.cooldown_seconds = float(values["cooldown_seconds"])
    config.overlay = bool(values["overlay"])
    config.updated_by_user_id = user.id if user else None
    config.updated_at = now_ts()


def operator_config_to_dict(config: OperatorInferenceConfig) -> dict[str, Any]:
    profile_id = str(config.model_profile or "hnm_balanced")
    if profile_id not in MODEL_PROFILES:
        profile_id = "hnm_balanced"
    return {
        "model_profile": profile_id,
        "fps": float(config.fps),
        "imgsz": int(config.imgsz),
        "pose_model": config.pose_model,
        "feature_version": MODEL_PROFILES[profile_id].get("feature_version", "v1"),
        "smoothing_windows": int(MODEL_PROFILES[profile_id].get("smoothing_windows", 3)),
        "alert_windows": int(config.alert_windows),
        "cooldown_seconds": float(config.cooldown_seconds),
        "updated_at": float(config.updated_at),
    }


def operator_values_from_profile(profile_id: str) -> dict[str, Any]:
    profile_values = config_from_profile(profile_id)
    return {
        "model_profile": profile_values["model_profile"],
        "fps": profile_values["fps"],
        "imgsz": profile_values["imgsz"],
        "pose_model": profile_values["pose_model"],
        "feature_version": profile_values.get("feature_version", "v1"),
        "smoothing_windows": profile_values.get("smoothing_windows", 3),
        "alert_windows": profile_values["alert_windows"],
        "cooldown_seconds": profile_values["cooldown_seconds"],
    }


def apply_operator_config_dict(config: OperatorInferenceConfig, values: dict[str, Any]) -> None:
    config.model_profile = str(values["model_profile"])
    config.fps = float(values["fps"])
    config.imgsz = int(values["imgsz"])
    config.pose_model = str(values["pose_model"])
    config.alert_windows = int(values["alert_windows"])
    config.cooldown_seconds = float(values["cooldown_seconds"])
    config.updated_at = now_ts()


def ensure_operator_config(db: Session, user: User) -> OperatorInferenceConfig:
    config = db.query(OperatorInferenceConfig).filter(OperatorInferenceConfig.user_id == user.id).first()
    if config:
        if str(config.model_profile) not in MODEL_PROFILES:
            values = operator_values_from_profile("hnm_balanced")
            apply_operator_config_dict(config, values)
            db.add(config)
            db.commit()
            db.refresh(config)
        return config
    values = operator_values_from_profile("hnm_balanced")
    config = OperatorInferenceConfig(user_id=user.id)
    apply_operator_config_dict(config, values)
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def apply_operator_profile_to_camera_config(
    camera_config: CameraInferenceConfig,
    operator_values: dict[str, Any],
    user: User | None = None,
) -> None:
    current = config_to_dict(camera_config)
    current.update(operator_values)
    apply_config_dict(camera_config, current, user=user)


def ensure_camera_config(db: Session, camera: Camera, user: User | None = None) -> CameraInferenceConfig:
    if camera.inference_config:
        if str(camera.inference_config.model_profile) not in MODEL_PROFILES:
            threshold = float(camera.inference_config.threshold)
            overlay = bool(camera.inference_config.overlay)
            values = config_from_profile("hnm_balanced", overlay=overlay)
            values["threshold"] = threshold
            apply_config_dict(camera.inference_config, values)
            db.add(camera.inference_config)
            db.commit()
            db.refresh(camera.inference_config)
        return camera.inference_config
    values = default_config()
    owner = user if user and user.id == camera.created_by_user_id else None
    if owner is None and camera.created_by_user_id:
        owner = db.query(User).filter(User.id == camera.created_by_user_id).first()
    if owner and user_role(owner) == "operador":
        operator_config = ensure_operator_config(db, owner)
        values.update(operator_config_to_dict(operator_config))
    config = CameraInferenceConfig(camera_id=camera.id)
    apply_config_dict(config, values)
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def serialize_camera(db: Session, camera: Camera) -> dict[str, Any]:
    config = ensure_camera_config(db, camera)
    return {
        "id": camera.id,
        "serial_number": camera.serial_number,
        "cam_id": camera.serial_number,
        "src": camera.src,
        "location_description": camera.location_description,
        "company_id": camera.company_id,
        "company_name": camera.company.name if camera.company else None,
        "is_active": camera.is_active,
        "status": camera.status,
        "status_detail": camera.status_detail,
        "created_by_user_id": camera.created_by_user_id,
        "created_by_email": camera.created_by.username if camera.created_by else None,
        "created_by_name": f"{camera.created_by.nombre} {camera.created_by.apellido}" if camera.created_by else None,
        "inference_config": config_to_dict(config),
    }


def alert_email_recipients(db: Session, camera: Camera) -> list[str]:
    query = db.query(User).join(Role, User.role_id == Role.id).filter(
        User.is_active.is_(True),
        Role.name.in_(("jefe", "operador")),
    )
    if camera.company_id is not None:
        query = query.filter(User.company_id == camera.company_id)
    return unique_recipients(user.username for user in query.all())


def build_alert_email(camera: Camera, payload: AlertRequest, alert: AlertLog) -> tuple[str, str]:
    probability = float(payload.prob or alert.prob or 0.0)
    threshold = payload.threshold
    profile = payload.model_profile or "modelo activo"
    location = camera.location_description or "Sin ubicacion"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(alert.timestamp or now_ts())))

    subject = f"Alerta de agresion - Camara {camera.serial_number}"
    body = "\n".join(
        [
            "Se detecto una posible conducta agresiva.",
            "",
            f"Camara: {camera.serial_number}",
            f"Ubicacion: {location}",
            f"Probabilidad: {probability:.3f}",
            f"Threshold: {float(threshold):.2f}" if threshold is not None else "Threshold: no informado",
            f"Perfil: {profile}",
            f"Fecha/Hora: {timestamp}",
            "",
            "Revise la alerta desde el panel de monitoreo Sicher.",
        ]
    )
    return subject, body


def validate_profile(profile_id: str) -> dict[str, Any]:
    if profile_id not in MODEL_PROFILES:
        raise HTTPException(status_code=400, detail={"error": "Perfil de modelo no existe"})
    profile = dict(MODEL_PROFILES[profile_id])
    available, missing = profile_available(profile)
    if not available:
        raise HTTPException(
            status_code=400,
            detail={"error": "Perfil no disponible en este servidor", "missing": missing},
        )
    return profile


def model_payload(payload: BaseModel, *, exclude_unset: bool = False) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_unset=exclude_unset)
    return payload.dict(exclude_unset=exclude_unset)


def password_reset_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def build_reset_link(request: Request, token: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/reset-password?token={token}"


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/login_fail")
def login_fail() -> FileResponse:
    return FileResponse(STATIC_DIR / "login_fail.html")


@app.get("/register")
def register_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "register.html")


@app.get("/reset-password")
def reset_password_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "reset_password.html")


@app.get("/config.js")
def config_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "config.js")


@app.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    ensure_roles(db)
    email = payload.email.strip().lower()
    if db.query(User).filter(User.username == email).first():
        raise HTTPException(status_code=400, detail={"error": "El email ya existe"})

    role_name = payload.charge.strip().lower()
    role = db.query(Role).filter(Role.name == role_name).first()
    if not role:
        raise HTTPException(status_code=400, detail={"error": "Rol invalido"})

    company: Company | None = None
    if role_name == "jefe":
        company_name = (payload.empresa_nombre or "").strip() or "Empresa"
        code = (payload.codigo or "").strip().upper()
        if not code:
            import random
            import string

            while True:
                code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                if not db.query(Company).filter(Company.codigo == code).first():
                    break
        if db.query(Company).filter(Company.codigo == code).first():
            raise HTTPException(status_code=400, detail={"error": "El codigo de empresa ya existe"})
        company = Company(name=company_name, codigo=code, is_active=True)
        db.add(company)
        db.commit()
        db.refresh(company)
    else:
        code = (payload.codigo or "").strip().upper()
        company = db.query(Company).filter(Company.codigo == code, Company.is_active.is_(True)).first()
        if not company:
            raise HTTPException(status_code=400, detail={"error": "Codigo de empresa invalido"})

    salt = make_salt()
    user = User(
        nombre=payload.nombre.strip(),
        apellido=payload.apellido.strip(),
        username=email,
        password=hash_password(payload.password, salt),
        salt=salt,
        role_id=role.id,
        company_id=company.id if company else None,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"ok": True, "company_code": company.codigo if company else None}


@app.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> Any:
    user = db.query(User).filter(User.username == payload.email.strip().lower()).first()
    if not user or not verify_password(payload.password, user.salt, user.password):
        raise HTTPException(status_code=401, detail={"error": "Credenciales invalidas"})
    if not user.is_active:
        return JSONResponse(
            status_code=403,
            content={
                "error": "Usted no tiene permisos. Pida activacion a su jefe.",
                "blocked": True,
            },
        )

    db.add(AccessLog(user_id=user.id))
    db.commit()
    role_name = user_role(user)
    return {
        "access_token": f"local-{user.id}-{int(time.time())}",
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.username,
        "nombre": user.nombre,
        "apellido": user.apellido,
        "role": role_name,
        "charge": role_name,
        "company_id": user.company_id,
        "company_name": user.company.name if user.company else None,
    }


@app.post("/password-reset/request")
def request_password_reset(
    payload: PasswordResetRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.username == email).first()
    response: dict[str, Any] = {
        "ok": True,
        "message": "Si el correo existe, enviaremos un enlace para crear una nueva contrasena.",
    }
    if not user:
        return response

    token = secrets.token_urlsafe(32)
    reset = PasswordResetToken(
        user_id=user.id,
        token_hash=password_reset_token_hash(token),
        expires_at=now_ts() + 3600,
    )
    db.add(reset)
    db.commit()

    link = build_reset_link(request, token)
    subject = "Restablecer contrasena - Sicher"
    body = "\n".join(
        [
            "Se solicito crear una nueva contrasena para su cuenta.",
            "",
            f"Correo: {user.username}",
            f"Enlace: {link}",
            "",
            "Este enlace vence en 1 hora.",
            "Si usted no solicito este cambio, ignore este mensaje.",
        ]
    )
    background_tasks.add_task(send_email, [user.username], subject, body)
    return response


@app.post("/password-reset/confirm")
def confirm_password_reset(payload: PasswordResetConfirm, db: Session = Depends(get_db)) -> dict[str, Any]:
    token_hash = password_reset_token_hash(payload.token.strip())
    reset = (
        db.query(PasswordResetToken)
        .filter(PasswordResetToken.token_hash == token_hash)
        .order_by(PasswordResetToken.id.desc())
        .first()
    )
    if not reset or reset.used_at is not None or reset.expires_at < now_ts():
        raise HTTPException(status_code=400, detail={"error": "Enlace invalido o vencido"})

    user = db.query(User).filter(User.id == reset.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail={"error": "Usuario no encontrado"})

    salt = make_salt()
    user.salt = salt
    user.password = hash_password(payload.password, salt)
    reset.used_at = now_ts()
    db.commit()
    return {"ok": True, "message": "Contrasena actualizada. Ahora puede iniciar sesion."}


@app.get("/session/status")
def session_status(
    db: Session = Depends(get_db),
    x_user_email: str | None = Header(default=None, alias="X-User-Email"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> dict[str, Any]:
    user: User | None = None
    if x_user_id and str(x_user_id).isdigit():
        user = db.query(User).filter(User.id == int(x_user_id)).first()
    if not user and x_user_email:
        user = db.query(User).filter(User.username == x_user_email).first()
    if not user:
        return {
            "authenticated": False,
            "is_active": False,
            "message": "Sesion no valida. Inicie sesion nuevamente.",
        }
    role_name = user_role(user)
    return {
        "authenticated": True,
        "is_active": bool(user.is_active),
        "user_id": user.id,
        "email": user.username,
        "role": role_name,
        "charge": role_name,
        "message": None if user.is_active else "Usted no tiene permisos. Pida activacion a su jefe.",
    }


@app.get("/model-profiles")
def model_profiles() -> list[dict[str, Any]]:
    return list_profiles()


@app.get("/operator/inference-config")
def get_operator_inference_config(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    if user_role(user) != "operador":
        raise HTTPException(status_code=403, detail={"error": "Solo operador"})
    config = ensure_operator_config(db, user)
    return operator_config_to_dict(config)


@app.patch("/operator/inference-config")
def patch_operator_inference_config(
    payload: OperatorInferenceConfigPatch,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    if user_role(user) != "operador":
        raise HTTPException(status_code=403, detail={"error": "Solo operador"})

    profile = validate_profile(payload.model_profile)
    operator_values = operator_values_from_profile(profile["id"])
    operator_config = ensure_operator_config(db, user)
    apply_operator_config_dict(operator_config, operator_values)

    updated_cameras = 0
    cameras = db.query(Camera).filter(Camera.created_by_user_id == user.id).all()
    for camera in cameras:
        camera_config = ensure_camera_config(db, camera, user=user)
        old_config = config_to_dict(camera_config)
        apply_operator_profile_to_camera_config(camera_config, operator_values, user=user)
        new_config = config_to_dict(camera_config)
        if old_config != new_config:
            updated_cameras += 1
            db.add(
                InferenceConfigAudit(
                    camera_id=camera.id,
                    user_id=user.id,
                    old_config=old_config,
                    new_config=new_config,
                )
            )
            db.add(CameraAction(user_id=user.id, camera_id=camera.id, action="modelo_operador"))

    db.commit()
    db.refresh(operator_config)
    return {
        **operator_config_to_dict(operator_config),
        "updated_cameras": updated_cameras,
    }


@app.get("/cameras")
def list_cameras(
    include_inactive: bool = Query(default=False),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> list[dict[str, Any]]:
    cameras = camera_query_for_user(db, user, include_inactive=include_inactive).order_by(Camera.id.desc()).all()
    return [serialize_camera(db, camera) for camera in cameras]


@app.post("/cameras")
def upsert_camera(payload: CameraRequest, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    serial = payload.serial_number.strip()
    camera = get_camera_by_serial(db, serial, user, include_inactive=True)
    if camera:
        camera.src = payload.src.strip()
        camera.location_description = payload.location_description
        camera.is_active = True
        camera.updated_at = now_ts()
        action = "actualizar"
    else:
        camera = Camera(
            serial_number=serial,
            src=payload.src.strip(),
            location_description=payload.location_description,
            company_id=user.company_id,
            created_by_user_id=user.id,
            is_active=True,
        )
        db.add(camera)
        db.flush()
        action = "agregar"
    db.commit()
    db.refresh(camera)
    ensure_camera_config(db, camera, user=user)
    db.add(CameraAction(user_id=user.id, camera_id=camera.id, action=action))
    db.commit()
    return serialize_camera(db, camera)


@app.delete("/cameras/{serial_number}")
def delete_camera(serial_number: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    camera = get_camera_by_serial(db, serial_number, user, include_inactive=True)
    if not camera:
        raise HTTPException(status_code=404, detail={"error": "Camara no encontrada"})
    camera.is_active = False
    camera.status = "offline"
    camera.updated_at = now_ts()
    db.add(CameraAction(user_id=user.id, camera_id=camera.id, action="eliminar"))
    db.commit()
    return {"ok": True}


@app.post("/cameras/{serial_number}/status")
def camera_status(serial_number: str, payload: CameraStatusRequest, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    camera = get_camera_by_serial(db, serial_number, user, include_inactive=True)
    if not camera:
        raise HTTPException(status_code=404, detail={"error": "Camara no encontrada"})
    camera.status = payload.status
    camera.status_detail = payload.detail
    camera.updated_at = now_ts()
    detail = (payload.detail or "").lower()
    action = f"estado:{payload.status}"
    if payload.status == "online" and "conexion abierta" in detail:
        action = "abrir"
    elif payload.status == "offline" and "cerrado por usuario" in detail:
        action = "cerrar"
    db.add(CameraAction(user_id=user.id, camera_id=camera.id, action=action))
    db.commit()
    return {"ok": True}


@app.get("/cameras/{serial_number}/inference-config")
def get_inference_config(serial_number: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    camera = get_camera_by_serial(db, serial_number, user, include_inactive=True)
    if not camera:
        raise HTTPException(status_code=404, detail={"error": "Camara no encontrada"})
    return config_to_dict(ensure_camera_config(db, camera))


@app.patch("/cameras/{serial_number}/inference-config")
def patch_inference_config(
    serial_number: str,
    payload: InferenceConfigPatch,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    camera = get_camera_by_serial(db, serial_number, user, include_inactive=True)
    if not camera:
        raise HTTPException(status_code=404, detail={"error": "Camara no encontrada"})

    config = ensure_camera_config(db, camera)
    old_config = config_to_dict(config)
    values = dict(old_config)

    if payload.model_profile:
        profile = validate_profile(payload.model_profile)
        values.update(config_from_profile(profile["id"], overlay=old_config["overlay"]))

    patch = model_payload(payload, exclude_unset=True)
    for key in ("threshold", "fps", "imgsz", "pose_model", "alert_windows", "cooldown_seconds", "overlay"):
        if key in patch and patch[key] is not None:
            values[key] = patch[key]

    validate_profile(str(values["model_profile"]))
    apply_config_dict(config, values, user=user)
    db.add(
        InferenceConfigAudit(
            camera_id=camera.id,
            user_id=user.id,
            old_config=old_config,
            new_config=config_to_dict(config),
        )
    )
    db.add(CameraAction(user_id=user.id, camera_id=camera.id, action="configuracion_inferencia"))
    db.commit()
    db.refresh(config)
    return config_to_dict(config)


@app.get("/cameras/{serial_number}/inference-status")
def get_inference_status(serial_number: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    camera = get_camera_by_serial(db, serial_number, user, include_inactive=True)
    if not camera:
        raise HTTPException(status_code=404, detail={"error": "Camara no encontrada"})
    return {
        "serial_number": serial_number,
        **SESSION_STATUS.get(serial_number, {"running": False, "last_score": None, "last_error": None}),
    }


@app.get("/overlay/get")
def overlay_get() -> dict[str, Any]:
    return {"overlay": bool(GLOBAL_STATE["overlay"])}


@app.post("/overlay/set")
def overlay_set(payload: OverlayRequest) -> dict[str, Any]:
    GLOBAL_STATE["overlay"] = bool(payload.overlay)
    return {"ok": True, "overlay": bool(GLOBAL_STATE["overlay"])}


@app.get("/threshold/get")
def threshold_get() -> dict[str, Any]:
    return {"thr_on_override": float(GLOBAL_STATE["threshold"]), "thr_on_model": 0.49}


@app.post("/threshold/set")
def threshold_set(payload: ThresholdRequest) -> dict[str, Any]:
    GLOBAL_STATE["threshold"] = float(payload.thr_on)
    return {"ok": True, "thr_on": float(payload.thr_on), "thr_off": payload.thr_off}


@app.post("/alerts/save")
def save_alert(
    payload: AlertRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    camera = get_camera_by_serial(db, payload.serial_number, user, include_inactive=True)
    if not camera:
        raise HTTPException(status_code=404, detail={"error": "Camara no encontrada"})

    reviewed_by: User | None = None
    if payload.reviewer_email:
        reviewed_by = db.query(User).filter(User.username == payload.reviewer_email).first()

    existing = (
        db.query(AlertLog)
        .filter(AlertLog.camera_id == camera.id, AlertLog.timestamp == float(payload.timestamp or 0))
        .first()
    )
    is_new_alert = existing is None
    alert = existing or AlertLog(camera_id=camera.id, timestamp=float(payload.timestamp or now_ts()))
    alert.prob = float(payload.prob)
    alert.status = payload.status
    alert.reviewed_by_user_id = reviewed_by.id if reviewed_by else (user.id if payload.status != "pending" else None)
    alert.reviewed_by_email = payload.reviewer_email
    alert.review_timestamp = now_ts() if payload.status != "pending" else alert.review_timestamp
    alert.model_profile = payload.model_profile
    alert.threshold = payload.threshold
    db.add(alert)
    db.commit()
    db.refresh(alert)

    email_result: dict[str, Any] | None = None
    if is_new_alert and payload.status == "pending":
        recipients = alert_email_recipients(db, camera)
        subject, body = build_alert_email(camera, payload, alert)
        if recipients:
            background_tasks.add_task(send_alert_email, recipients, subject, body)
            email_result = {"queued": True, "recipients": recipients}
        else:
            email_result = {"queued": False, "reason": "no_recipients"}

    return {"ok": True, "email": email_result}


@app.get("/companies")
def list_companies(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[dict[str, Any]]:
    query = db.query(Company)
    if user_role(user) != "jefe" and user.company_id is not None:
        query = query.filter(Company.id == user.company_id)
    companies = query.order_by(Company.id.desc()).all()
    rows: list[dict[str, Any]] = []
    for company in companies:
        rows.append(
            {
                "id": company.id,
                "name": company.name,
                "rut": company.rut,
                "codigo": company.codigo,
                "is_active": company.is_active,
                "user_count": len(company.users),
                "camera_count": len(company.cameras),
            }
        )
    return rows


@app.post("/companies")
def create_company(payload: CompanyRequest, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe puede crear empresas"})
    import random
    import string

    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if not db.query(Company).filter(Company.codigo == code).first():
            break
    company = Company(name=payload.name.strip(), rut=payload.rut, codigo=code, is_active=True)
    db.add(company)
    db.commit()
    db.refresh(company)
    return {"id": company.id, "name": company.name, "rut": company.rut, "codigo": company.codigo, "is_active": company.is_active}


@app.get("/admin/users")
def admin_users(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[dict[str, Any]]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    query = db.query(User)
    if user.company_id is not None:
        query = query.filter(User.company_id == user.company_id)
    return [
        {
            "id": item.id,
            "nombre": item.nombre,
            "apellido": item.apellido,
            "email": item.username,
            "role": user_role(item),
            "charge": user_role(item),
            "company_name": item.company.name if item.company else "-",
            "is_active": item.is_active,
        }
        for item in query.order_by(User.id.desc()).all()
    ]


@app.patch("/admin/users/{target_user_id}/active")
def admin_set_user_active(
    target_user_id: int,
    payload: UserActivePatch,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})

    target = db.query(User).filter(User.id == target_user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail={"error": "Usuario no encontrado"})
    if user.company_id is not None and target.company_id != user.company_id:
        raise HTTPException(status_code=403, detail={"error": "Usuario fuera de la empresa"})
    if user_role(target) != "operador":
        raise HTTPException(status_code=400, detail={"error": "Solo se puede bloquear o activar operadores"})

    target.is_active = bool(payload.is_active)
    db.commit()
    db.refresh(target)
    return {
        "ok": True,
        "id": target.id,
        "email": target.username,
        "is_active": target.is_active,
    }


@app.get("/admin/stats")
def admin_stats(
    operator_id: int | None = Query(default=None),
    days: int | None = Query(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})

    operator_query = db.query(User).join(Role, User.role_id == Role.id).filter(
        Role.name == "operador",
    )
    if user.company_id is not None:
        operator_query = operator_query.filter(User.company_id == user.company_id)
    operators = operator_query.order_by(User.nombre.asc(), User.apellido.asc()).all()
    operator_ids = {operator.id for operator in operators}

    if operator_id is not None and operator_id not in operator_ids:
        raise HTTPException(status_code=404, detail={"error": "Operador no encontrado en esta empresa"})
    if days is not None and days not in {7, 15}:
        raise HTTPException(status_code=400, detail={"error": "Periodo invalido"})

    since_ts = now_ts() - (days * 86400) if days is not None else None

    def operator_row(operator: User) -> dict[str, Any]:
        return {
            "user_id": operator.id,
            "operator": f"{operator.nombre} {operator.apellido}".strip(),
            "email": operator.username,
        }

    selected_ids = {operator_id} if operator_id is not None else operator_ids
    fallback_operator_id = next(iter(operator_ids)) if len(operator_ids) == 1 else None
    generated_counts = {operator.id: 0 for operator in operators if operator.id in selected_ids}
    accepted_counts = {operator.id: 0 for operator in operators if operator.id in selected_ids}
    reviewed_cameras = {operator.id: set() for operator in operators if operator.id in selected_ids}

    alert_query = db.query(AlertLog).join(Camera, AlertLog.camera_id == Camera.id)
    if user.company_id is not None:
        alert_query = alert_query.filter(Camera.company_id == user.company_id)
    if since_ts is not None:
        alert_query = alert_query.filter(AlertLog.timestamp >= since_ts)
    alerts = alert_query.all()

    for alert in alerts:
        creator_id = alert.camera.created_by_user_id if alert.camera else None
        if creator_id not in generated_counts:
            creator_id = alert.reviewed_by_user_id if alert.reviewed_by_user_id in generated_counts else fallback_operator_id
        if creator_id in generated_counts:
            generated_counts[creator_id] += 1

        reviewer_id = alert.reviewed_by_user_id
        if alert.status == "true_positive" and reviewer_id in accepted_counts:
            accepted_counts[reviewer_id] += 1

    action_query = db.query(CameraAction).outerjoin(Camera, CameraAction.camera_id == Camera.id).outerjoin(User, CameraAction.user_id == User.id)
    if user.company_id is not None:
        action_query = action_query.filter((Camera.company_id == user.company_id) | (User.company_id == user.company_id))
    if since_ts is not None:
        action_query = action_query.filter(CameraAction.timestamp >= since_ts)
    for action in action_query.all():
        if action.user_id in reviewed_cameras and action.camera_id is not None:
            reviewed_cameras[action.user_id].add(action.camera_id)

    zones: dict[str, dict[str, Any]] = {}
    for alert in alerts:
        camera = alert.camera
        if not camera:
            continue
        if operator_id is not None and camera.created_by_user_id != operator_id and alert.reviewed_by_user_id != operator_id:
            continue
        zone = (camera.location_description or "Sin ubicacion").strip() or "Sin ubicacion"
        row = zones.setdefault(
            zone,
            {
                "location": zone,
                "alert_count": 0,
                "accepted_count": 0,
                "false_positive_count": 0,
                "pending_count": 0,
            },
        )
        row["alert_count"] += 1
        if alert.status == "true_positive":
            row["accepted_count"] += 1
        elif alert.status == "false_positive":
            row["false_positive_count"] += 1
        else:
            row["pending_count"] += 1

    generated_rows = []
    accepted_rows = []
    reviewed_rows = []
    for operator in operators:
        if operator.id not in selected_ids:
            continue
        base = operator_row(operator)
        generated_rows.append({**base, "generated_alerts": generated_counts.get(operator.id, 0)})
        accepted_rows.append({**base, "accepted_alerts": accepted_counts.get(operator.id, 0)})
        reviewed_rows.append({**base, "reviewed_cameras": len(reviewed_cameras.get(operator.id, set()))})

    generated_rows.sort(key=lambda row: row["generated_alerts"], reverse=True)
    accepted_rows.sort(key=lambda row: row["accepted_alerts"], reverse=True)
    reviewed_rows.sort(key=lambda row: row["reviewed_cameras"], reverse=True)
    zone_rows = sorted(zones.values(), key=lambda row: row["alert_count"], reverse=True)

    return {
        "operator_id": operator_id,
        "days": days,
        "since": since_ts,
        "generated_alerts_by_operator": generated_rows,
        "accepted_alerts_by_operator": accepted_rows,
        "reviewed_cameras_by_operator": reviewed_rows,
        "alerts_by_zone": zone_rows,
        "summary": {
            "generated_alerts": sum(row["generated_alerts"] for row in generated_rows),
            "accepted_alerts": sum(row["accepted_alerts"] for row in accepted_rows),
            "reviewed_cameras": sum(row["reviewed_cameras"] for row in reviewed_rows),
            "zones_with_alerts": len(zone_rows),
            "total_alerts": sum(row["alert_count"] for row in zone_rows),
        },
    }


@app.get("/admin/logs/alerts")
def admin_alert_logs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[dict[str, Any]]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    query = db.query(AlertLog).join(Camera, AlertLog.camera_id == Camera.id)
    if user.company_id is not None:
        query = query.filter(Camera.company_id == user.company_id)
    rows = []
    for alert in query.order_by(AlertLog.timestamp.desc()).limit(500).all():
        rows.append(
            {
                "id": alert.id,
                "camera_id": alert.camera_id,
                "serial_number": alert.camera.serial_number if alert.camera else None,
                "prob": alert.prob,
                "status": alert.status,
                "timestamp": alert.timestamp,
                "review_timestamp": alert.review_timestamp,
                "reviewed_by_user_id": alert.reviewed_by_user_id,
                "reviewed_by_email": alert.reviewed_by_email,
                "reviewed_by_name": f"{alert.reviewed_by.nombre} {alert.reviewed_by.apellido}" if alert.reviewed_by else "-",
                "model_profile": alert.model_profile,
                "threshold": alert.threshold,
            }
        )
    return rows


@app.delete("/admin/logs/alerts/clear")
def clear_alert_logs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    db.query(AlertLog).delete()
    db.commit()
    return {"ok": True}


@app.get("/admin/logs/access")
def admin_access_logs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[dict[str, Any]]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    query = db.query(AccessLog).join(User, AccessLog.user_id == User.id)
    if user.company_id is not None:
        query = query.filter(User.company_id == user.company_id)
    return [
        {
            "user_id": log.user_id,
            "email": log.user.username if log.user else None,
            "usuario": f"{log.user.nombre} {log.user.apellido}" if log.user else "-",
            "cargo": user_role(log.user),
            "timestamp": log.timestamp,
        }
        for log in query.order_by(AccessLog.timestamp.desc()).limit(500).all()
    ]


@app.delete("/admin/logs/access/clear")
def clear_access_logs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    db.query(AccessLog).delete()
    db.commit()
    return {"ok": True}


@app.get("/admin/logs/cameras")
def admin_camera_logs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[dict[str, Any]]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    query = db.query(CameraAction).outerjoin(Camera, CameraAction.camera_id == Camera.id).outerjoin(User, CameraAction.user_id == User.id)
    if user.company_id is not None:
        query = query.filter((Camera.company_id == user.company_id) | (User.company_id == user.company_id))
    allowed_actions = {"abrir", "cerrar", "eliminar", "status:online", "status:offline"}
    action_labels = {
        "abrir": "Abrir",
        "cerrar": "Cerrar",
        "eliminar": "Eliminar",
        "status:online": "Abrir",
        "status:offline": "Cerrar",
    }
    rows = []
    for log in query.order_by(CameraAction.timestamp.desc()).limit(500).all():
        if log.action not in allowed_actions:
            continue
        rows.append(
            {
                "user_id": log.user_id,
                "email": log.user.username if log.user else None,
                "usuario": f"{log.user.nombre} {log.user.apellido}" if log.user else "-",
                "serial_number": log.camera.serial_number if log.camera else "-",
                "camera_id": log.camera_id,
                "action": action_labels.get(log.action, log.action),
                "timestamp": log.timestamp,
            }
        )
    return rows


@app.get("/admin/logs/inference-config")
def admin_inference_config_logs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[dict[str, Any]]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    query = db.query(InferenceConfigAudit).join(Camera, InferenceConfigAudit.camera_id == Camera.id)
    if user.company_id is not None:
        query = query.filter(Camera.company_id == user.company_id)
    rows = []
    for audit in query.order_by(InferenceConfigAudit.timestamp.desc()).limit(500).all():
        rows.append(
            {
                "id": audit.id,
                "serial_number": audit.camera.serial_number if audit.camera else "-",
                "user_id": audit.user_id,
                "email": audit.user.username if audit.user else None,
                "usuario": f"{audit.user.nombre} {audit.user.apellido}" if audit.user else "-",
                "old_config": audit.old_config,
                "new_config": audit.new_config,
                "timestamp": audit.timestamp,
            }
        )
    return rows


@app.delete("/admin/logs/inference-config/clear")
def clear_inference_config_logs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    query = db.query(InferenceConfigAudit)
    if user.company_id is not None:
        camera_ids = select(Camera.id).where(Camera.company_id == user.company_id)
        query = query.filter(InferenceConfigAudit.camera_id.in_(camera_ids))
    deleted = query.delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": deleted}


@app.delete("/admin/logs/actions/clear")
def clear_camera_logs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict[str, Any]:
    if user_role(user) != "jefe":
        raise HTTPException(status_code=403, detail={"error": "Solo jefe"})
    db.query(CameraAction).delete()
    db.commit()
    return {"ok": True}


@app.websocket("/ws/stream/{serial_number}")
async def ws_stream(
    websocket: WebSocket,
    serial_number: str,
    src: str | None = Query(default=None),
    user_id: int | None = Query(default=None),
) -> None:
    await websocket.accept()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first() if user_id else None
        if user_id and not user:
            await websocket.send_json({"type": "error", "serial_number": serial_number, "error": "Usuario bloqueado o sin acceso"})
            return
        camera = get_camera_by_serial(db, serial_number, user, include_inactive=True)
        if not camera and not src:
            await websocket.send_json({"type": "error", "serial_number": serial_number, "error": "Camara no encontrada"})
            return
        stream_src = src or (camera.src if camera else "")

        def config_loader() -> dict[str, Any]:
            local_db = SessionLocal()
            try:
                local_user = local_db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first() if user_id else None
                if user_id and not local_user:
                    raise PermissionError("Usuario bloqueado o sin acceso")
                local_camera = get_camera_by_serial(local_db, serial_number, local_user, include_inactive=True)
                if local_camera:
                    config = config_to_dict(ensure_camera_config(local_db, local_camera))
                else:
                    config = default_config()
                if GLOBAL_STATE["overlay"]:
                    config["overlay"] = True
                return config
            finally:
                local_db.close()

        await stream_camera(websocket, serial_number, stream_src, config_loader)
    except WebSocketDisconnect:
        return
    finally:
        db.close()
