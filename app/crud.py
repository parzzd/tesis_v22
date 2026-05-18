# app/crud.py  –  Operaciones CRUD sobre la base de datos
from sqlalchemy.orm import Session
from app.models import User, AccessLog, CameraAction, AlertLog, Role, Company
from app.utils import make_salt, hash_password


# ── Empresas ──────────────────────────────────────────────
def _generate_company_code(db: Session) -> str:
    """Generate unique 6-digit company code."""
    import random
    import string
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if not db.query(Company).filter(Company.codigo == code).first():
            return code


def create_company(db: Session, name: str, rut: str | None = None, codigo: str | None = None) -> Company:
    if not codigo:
        codigo = _generate_company_code(db)
    company = Company(name=name, rut=rut or None, codigo=codigo, is_active=True)
    db.add(company)
    db.commit()
    db.refresh(company)
    return company


def get_company_by_id(db: Session, company_id: int) -> Company | None:
    return db.query(Company).filter(Company.id == company_id).first()


def get_company_by_codigo(db: Session, codigo: str) -> Company | None:
    return db.query(Company).filter(Company.codigo == codigo, Company.is_active.is_(True)).first()


# ── Usuarios ──────────────────────────────────────────────
def create_user(db: Session, nombre: str, apellido: str, email: str,
                password: str, role_id: int, company_id: int | None = None) -> User:
    salt = make_salt()
    user = User(
        nombre=nombre,
        apellido=apellido,
        username=email,
        password=hash_password(password, salt),
        salt=salt,
        role_id=role_id,
        company_id=company_id,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.username == email).first()


def get_role_by_name(db: Session, role_name: str) -> Role | None:
    return db.query(Role).filter(Role.name == role_name).first()


def ensure_roles(db: Session, role_names: list[str]):
    current = {r.name for r in db.query(Role).all()}
    changed = False
    for role_name in role_names:
        if role_name not in current:
            db.add(Role(name=role_name))
            changed = True
    if changed:
        db.commit()


# ── Logs ──────────────────────────────────────────────────
def add_access_log(db: Session, user_id: int):
    db.add(AccessLog(user_id=user_id))
    db.commit()


def add_camera_action(db: Session, user_id: int, camera_id: int, action: str):
    db.add(CameraAction(user_id=user_id, camera_id=camera_id, action=action))
    db.commit()


def add_alert_log(db: Session, camera_id: int, prob: float):
    db.add(AlertLog(camera_id=camera_id, prob=prob))
    db.commit()
