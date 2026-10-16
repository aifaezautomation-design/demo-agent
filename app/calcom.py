"""Minimal async Cal.com API v2 client (slots + bookings)."""

from datetime import datetime
from typing import Any

import httpx

from app import config

# Cal.com versions each endpoint family by header; a wrong value silently
# falls back to an older contract.
_SLOTS_VERSION = "2024-09-04"
_BOOKINGS_VERSION = "2026-02-25"

# One pooled client for the process: keeps TLS sessions warm, which matters
# for tool-call latency mid-conversation.
_http = httpx.AsyncClient(base_url=config.CALCOM_BASE_URL, timeout=httpx.Timeout(6.0))


class CalComError(Exception):
    pass


class CalCom:
    def __init__(self, api_key: str):
        self._auth = {"Authorization": f"Bearer {api_key}"}

    async def _request(self, method: str, path: str, version: str, **kwargs) -> Any:
        headers = {**self._auth, "cal-api-version": version}
        # Reads are safe to repeat, so a slow or failing lookup gets a second, quicker try (a stalled call is worse than
        # a retry). Writes are never repeated: a lost reply must not become a double booking.
        attempts = 2 if method == "GET" else 1
        for attempt in range(1, attempts + 1):
            try:
                if method == "GET":
                    kwargs["timeout"] = httpx.Timeout(3.5)
                resp = await _http.request(method, path, headers=headers, **kwargs)
            except httpx.HTTPError as e:
                if attempt < attempts:
                    continue
                raise CalComError(f"Cal.com unreachable: {e!r}") from e
            if resp.status_code >= 500 and attempt < attempts:
                continue
            break
        if resp.status_code >= 400:
            raise CalComError(f"Cal.com {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        if body.get("status") != "success":
            raise CalComError(f"Cal.com error: {body}")
        return body.get("data")

    async def get_slots(
        self, event_type_id: int, start: datetime, end: datetime, time_zone: str
    ) -> list[datetime]:
        """Returns available slot start times (tz-aware, in `time_zone`)."""
        data = await self._request(
            "GET",
            "/v2/slots",
            _SLOTS_VERSION,
            params={
                "eventTypeId": event_type_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "timeZone": time_zone,
            },
        )
        slots: list[datetime] = []
        for day_slots in (data or {}).values():
            for slot in day_slots:
                # Docs show both bare strings and {"start": ...} objects.
                raw = slot["start"] if isinstance(slot, dict) else slot
                slots.append(datetime.fromisoformat(raw))
        return sorted(slots)

    async def create_booking(
        self,
        event_type_id: int,
        start_utc: datetime,
        name: str,
        phone: str,
        time_zone: str,
        metadata: dict[str, str],
    ) -> dict:
        # No attendee email: the 2026-02-25 contract makes it optional, which
        # keeps Cal.com from sending patient emails.
        return await self._request(
            "POST",
            "/v2/bookings",
            _BOOKINGS_VERSION,
            json={
                "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "eventTypeId": event_type_id,
                "attendee": {"name": name, "phoneNumber": phone, "timeZone": time_zone},
                "metadata": metadata,
            },
        )

    async def reschedule_booking(self, booking_uid: str, new_start_utc: datetime) -> dict:
        return await self._request(
            "POST",
            f"/v2/bookings/{booking_uid}/reschedule",
            _BOOKINGS_VERSION,
            json={
                "start": new_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reschedulingReason": "Rescheduled by patient via phone receptionist",
            },
        )

    async def cancel_booking(self, booking_uid: str, reason: str) -> None:
        await self._request(
            "POST",
            f"/v2/bookings/{booking_uid}/cancel",
            _BOOKINGS_VERSION,
            json={"cancellationReason": reason},
        )
