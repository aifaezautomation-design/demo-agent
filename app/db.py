"""Supabase access. The backend always uses the service role and scopes every
query by client_id itself; RLS protects the dashboard, not this process."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from supabase import AsyncClient, acreate_client

from app import config

_client: AsyncClient | None = None


async def get_db() -> AsyncClient:
    global _client
    if _client is None:
        _client = await acreate_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
    return _client


@dataclass
class Tenant:
    id: str
    name: str
    phone_number: str
    timezone: str
    cal_com_event_type_id: int | None
    emergency_event_type_id: int | None
    cal_com_api_key: str | None
    knowledge_base: dict[str, Any] = field(default_factory=dict)
    demo_mode: bool = False
    record_calls: bool = False
    total_minute_limit: int = 2000
    used_minutes: float = 0.0
    status: str = "active"

    @property
    def remaining_minutes(self) -> float:
        return max(self.total_minute_limit - self.used_minutes, 0.0)

    @property
    def can_take_calls(self) -> bool:
        return self.status == "active" and self.remaining_minutes > 0


# Tenant config is read on every inbound call; a short TTL cache keeps that
# lookup off the hot path without serving stale balances for long.
_TENANT_TTL_SECS = 30.0
_tenant_cache: dict[str, tuple[float, Tenant]] = {}


async def get_tenant_by_number(phone_number: str) -> Tenant | None:
    cached = _tenant_cache.get(phone_number)
    if cached and time.monotonic() - cached[0] < _TENANT_TTL_SECS:
        return cached[1]

    db = await get_db()
    res = (
        await db.table("clients")
        .select("*, client_secrets(cal_com_api_key)")
        .eq("phone_number", phone_number)
        .maybe_single()
        .execute()
    )
    row = res.data if res else None
    if not row:
        return None

    secrets = row.get("client_secrets") or {}
    tenant = Tenant(
        id=row["id"],
        name=row["name"],
        phone_number=row["phone_number"],
        timezone=row["timezone"],
        cal_com_event_type_id=row.get("cal_com_event_type_id"),
        emergency_event_type_id=row.get("emergency_event_type_id"),
        cal_com_api_key=secrets.get("cal_com_api_key"),
        knowledge_base=row.get("knowledge_base") or {},
        demo_mode=row.get("demo_mode", False),
        record_calls=row.get("record_calls", False),
        total_minute_limit=row["total_minute_limit"],
        used_minutes=float(row["used_minutes"]),
        status=row["status"],
    )
    _tenant_cache[phone_number] = (time.monotonic(), tenant)
    return tenant


async def create_call_log(
    client_id: str, caller_phone: str, call_control_id: str | None, call_sid: str | None
) -> str:
    db = await get_db()
    res = (
        await db.table("call_logs")
        .insert(
            {
                "client_id": client_id,
                "caller_phone": caller_phone,
                "call_control_id": call_control_id,
                "call_sid": call_sid,
                "status": "in_progress",
            }
        )
        .execute()
    )
    return res.data[0]["id"]


async def finalize_call(
    call_id: str,
    duration_seconds: int,
    transcript: list[dict],
    summary: str | None,
    appointment_booked: bool,
) -> float:
    """Atomically closes the call log and deducts minutes (see schema.sql)."""
    db = await get_db()
    res = await db.rpc(
        "finalize_call",
        {
            "p_call_id": call_id,
            "p_duration_seconds": duration_seconds,
            "p_transcript": transcript,
            "p_summary": summary,
            "p_appointment_booked": appointment_booked,
        },
    ).execute()
    return float(res.data or 0)


async def flag_urgent_callback(call_id: str, reason: str) -> None:
    db = await get_db()
    await (
        db.table("call_logs")
        .update({"urgent_callback": True, "urgent_reason": reason})
        .eq("id", call_id)
        .execute()
    )


async def insert_appointment(row: dict) -> dict:
    db = await get_db()
    res = await db.table("appointments").insert(row).execute()
    return res.data[0]


async def find_upcoming_appointments(
    client_id: str, phone: str | None, name: str | None, now_iso: str
) -> list[dict]:
    db = await get_db()
    query = (
        db.table("appointments")
        .select("id, cal_booking_uid, patient_name, patient_phone, visit_reason, start_time, is_emergency")
        .eq("client_id", client_id)
        .neq("status", "cancelled")
        .gte("start_time", now_iso)
        .order("start_time")
        .limit(5)
    )
    if phone:
        query = query.eq("patient_phone", phone)
    elif name:
        query = query.ilike("patient_name", f"%{name}%")
    else:
        return []
    res = await query.execute()
    return res.data or []


async def get_appointment(client_id: str, appointment_id: str) -> dict | None:
    db = await get_db()
    res = (
        await db.table("appointments")
        .select("*")
        .eq("client_id", client_id)
        .eq("id", appointment_id)
        .maybe_single()
        .execute()
    )
    return res.data if res else None


async def update_appointment(client_id: str, appointment_id: str, fields: dict) -> None:
    db = await get_db()
    await (
        db.table("appointments")
        .update({**fields, "updated_at": datetime.now(timezone.utc).isoformat()})
        .eq("client_id", client_id)
        .eq("id", appointment_id)
        .execute()
    )


async def save_recording(call_sid: str, audio: bytes) -> bool:
    """Stores a recording in the private bucket and links it to its call log."""
    db = await get_db()
    res = (
        await db.table("call_logs")
        .select("id, client_id")
        .eq("call_sid", call_sid)
        .maybe_single()
        .execute()
    )
    row = res.data if res else None
    if not row:
        return False

    path = f"{row['client_id']}/{row['id']}.mp3"
    await db.storage.from_("recordings").upload(
        path, audio, {"content-type": "audio/mpeg", "upsert": "true"}
    )
    await db.table("call_logs").update({"recording_path": path}).eq("id", row["id"]).execute()
    return True


async def get_tenant_by_id(client_id: str) -> Tenant | None:
    db = await get_db()
    res = await db.table("clients").select("phone_number").eq("id", client_id).maybe_single().execute()
    return await get_tenant_by_number(res.data["phone_number"]) if res and res.data else None
