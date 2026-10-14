-- =============================================================================
-- Boltex AI — Dental Voice Agent: multi-tenant schema (Supabase / Postgres)
-- Run in the Supabase SQL editor. Idempotent where practical.
-- =============================================================================

create extension if not exists pgcrypto;

-- -----------------------------------------------------------------------------
-- Tenants
-- -----------------------------------------------------------------------------
create table if not exists public.clients (
    id                     text primary key,                 -- e.g. 'apex-dental'
    name                   text not null,
    phone_number           text unique not null,             -- E.164 number Telnyx routes to us
    timezone               text not null default 'America/Chicago',
    cal_com_event_type_id  int,
    emergency_event_type_id int,                             -- optional separate Cal.com event for emergencies
    knowledge_base         jsonb not null default '{}'::jsonb, -- address, hours, insurances, services
    demo_mode              boolean not null default false,   -- enables Boltex sales lines in the prompt
    record_calls           boolean not null default false,   -- adds a disclosure to the greeting when on
    total_minute_limit     int not null default 2000,
    used_minutes           numeric(10,2) not null default 0,
    billing_cycle_start    date not null default date_trunc('month', now())::date,
    status                 text not null default 'active' check (status in ('active','paused','suspended')),
    created_at             timestamptz not null default now()
);

-- Secrets live apart from `clients` so dashboard users can never SELECT them.
-- RLS is on with NO policies: only the service role (backend) can read.
create table if not exists public.client_secrets (
    client_id        text primary key references public.clients(id) on delete cascade,
    cal_com_api_key  text not null,
    updated_at       timestamptz not null default now()
);

-- Maps Supabase Auth users to the practice(s) they may see.
create table if not exists public.client_users (
    user_id    uuid not null references auth.users(id) on delete cascade,
    client_id  text not null references public.clients(id) on delete cascade,
    role       text not null default 'staff' check (role in ('owner','staff')),
    created_at timestamptz not null default now(),
    primary key (user_id, client_id)
);

-- Boltex staff who see every tenant in the admin dashboard.
create table if not exists public.platform_admins (
    user_id uuid primary key references auth.users(id) on delete cascade
);

-- -----------------------------------------------------------------------------
-- Calls & appointments
-- -----------------------------------------------------------------------------
create table if not exists public.call_logs (
    id                     uuid primary key default gen_random_uuid(),
    client_id              text not null references public.clients(id) on delete cascade,
    call_control_id        text unique,                      -- Telnyx media-stream call id
    call_sid               text unique,                      -- TeXML CallSid; links recording callbacks
    caller_phone           text not null,
    status                 text not null default 'in_progress' check (status in ('in_progress','completed','failed')),
    call_duration_seconds  int not null default 0,
    billed_minutes         numeric(10,2),                    -- null until billed (idempotency guard)
    recording_path         text,                             -- object path in the private 'recordings' bucket
    transcript             jsonb,
    call_summary           text,
    appointment_booked     boolean not null default false,
    urgent_callback        boolean not null default false,   -- emergency with no slot available
    urgent_reason          text,
    started_at             timestamptz not null default now(),
    ended_at               timestamptz,
    created_at             timestamptz not null default now()
);
create index if not exists call_logs_client_created_idx on public.call_logs (client_id, created_at desc);
create index if not exists call_logs_urgent_idx on public.call_logs (client_id) where urgent_callback;

-- Our own record of bookings. Cal.com v2 cannot search bookings by phone,
-- so reschedule/cancel lookups go through this table.
create table if not exists public.appointments (
    id               uuid primary key default gen_random_uuid(),
    client_id        text not null references public.clients(id) on delete cascade,
    call_log_id      uuid references public.call_logs(id) on delete set null,
    cal_booking_uid  text unique not null,
    patient_name     text not null,
    patient_phone    text not null,
    visit_reason     text,
    is_new_patient   boolean,
    is_emergency     boolean not null default false,
    start_time       timestamptz not null,
    status           text not null default 'booked' check (status in ('booked','rescheduled','cancelled')),
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);
create index if not exists appointments_lookup_idx on public.appointments (client_id, patient_phone, start_time);

-- -----------------------------------------------------------------------------
-- RLS
-- -----------------------------------------------------------------------------
-- SECURITY DEFINER helpers avoid recursive policy evaluation on client_users.
create or replace function public.is_platform_admin()
returns boolean language sql stable security definer set search_path = public as $$
    select exists (select 1 from platform_admins where user_id = auth.uid());
$$;

create or replace function public.can_access_client(p_client_id text)
returns boolean language sql stable security definer set search_path = public as $$
    select public.is_platform_admin()
        or exists (select 1 from client_users where user_id = auth.uid() and client_id = p_client_id);
$$;

alter table public.clients         enable row level security;
alter table public.client_secrets  enable row level security;
alter table public.client_users    enable row level security;
alter table public.platform_admins enable row level security;
alter table public.call_logs       enable row level security;
alter table public.appointments    enable row level security;

drop policy if exists "read own client"        on public.clients;
drop policy if exists "read own memberships"   on public.client_users;
drop policy if exists "read own call logs"     on public.call_logs;
drop policy if exists "read own appointments"  on public.appointments;

create policy "read own client"       on public.clients      for select using (public.can_access_client(id));
create policy "read own memberships"  on public.client_users for select using (user_id = auth.uid() or public.is_platform_admin());
create policy "read own call logs"    on public.call_logs    for select using (public.can_access_client(client_id));
create policy "read own appointments" on public.appointments for select using (public.can_access_client(client_id));
-- No INSERT/UPDATE/DELETE policies: all writes come from the backend's service role.

-- -----------------------------------------------------------------------------
-- Billing: atomic, idempotent call finalization
-- -----------------------------------------------------------------------------
-- Writes the final call record and deducts minutes in one transaction.
-- Safe to call twice for the same call: the second call is a no-op.
create or replace function public.finalize_call(
    p_call_id            uuid,
    p_duration_seconds   int,
    p_transcript         jsonb,
    p_summary            text,
    p_appointment_booked boolean
) returns numeric
language plpgsql security definer set search_path = public as $$
declare
    v_client_id text;
    v_minutes   numeric(10,2) := round(greatest(p_duration_seconds, 0) / 60.0, 2);
begin
    update call_logs
       set status = 'completed',
           ended_at = now(),
           call_duration_seconds = p_duration_seconds,
           transcript = p_transcript,
           call_summary = p_summary,
           appointment_booked = appointment_booked or p_appointment_booked,
           billed_minutes = v_minutes
     where id = p_call_id and billed_minutes is null
    returning client_id into v_client_id;

    if v_client_id is null then
        return 0;  -- unknown call or already billed
    end if;

    update clients set used_minutes = used_minutes + v_minutes where id = v_client_id;
    return v_minutes;
end;
$$;

-- Call from pg_cron daily; resets any tenant whose cycle has rolled over.
--   select cron.schedule('reset-minutes', '5 0 * * *', 'select public.reset_billing_cycles()');
create or replace function public.reset_billing_cycles()
returns int language sql security definer set search_path = public as $$
    with r as (
        update clients
           set used_minutes = 0,
               billing_cycle_start = date_trunc('month', now())::date
         where billing_cycle_start < date_trunc('month', now())::date
        returning 1
    ) select count(*)::int from r;
$$;

revoke execute on function public.finalize_call(uuid,int,jsonb,text,boolean) from public, anon, authenticated;
revoke execute on function public.reset_billing_cycles() from public, anon, authenticated;

-- Dashboard convenience view (inherits RLS from underlying tables).
create or replace view public.client_usage with (security_invoker = true) as
select id as client_id,
       name,
       total_minute_limit,
       used_minutes,
       greatest(total_minute_limit - used_minutes, 0) as remaining_minutes,
       billing_cycle_start,
       status
  from public.clients;

-- Private bucket for call recordings. Telnyx recording URLs expire, so the
-- backend copies each file here. Dashboards fetch short-lived signed URLs;
-- the first path segment is the client_id.
insert into storage.buckets (id, name, public)
values ('recordings', 'recordings', false)
on conflict (id) do nothing;

drop policy if exists "read own recordings" on storage.objects;
create policy "read own recordings" on storage.objects for select
    using (bucket_id = 'recordings' and public.can_access_client((storage.foldername(name))[1]));

-- Realtime feed for the live dashboard.
do $$ begin
    alter publication supabase_realtime add table public.call_logs, public.appointments, public.clients;
exception when duplicate_object then null; end $$;

-- -----------------------------------------------------------------------------
-- Seed: Apex Dental Center (sample tenant)
-- -----------------------------------------------------------------------------
insert into public.clients (id, name, phone_number, timezone, cal_com_event_type_id, demo_mode, knowledge_base)
values (
    'apex-dental',
    'Apex Dental Care',
    '+15125550100',              -- replace with your Telnyx number
    'America/Chicago',
    null,                        -- replace with your Cal.com event type id
    true,
    '{
  "address": "742 Evergreen Terrace, Suite 100, Austin, TX 78701",
  "hours": "Monday to Saturday 8:00 AM to 7:00 PM Central Time (9:00 AM to 8:00 PM Eastern), Sunday closed",
  "hours_schedule": {
    "mon": [
      "08:00",
      "19:00"
    ],
    "tue": [
      "08:00",
      "19:00"
    ],
    "wed": [
      "08:00",
      "19:00"
    ],
    "thu": [
      "08:00",
      "19:00"
    ],
    "fri": [
      "08:00",
      "19:00"
    ],
    "sat": [
      "08:00",
      "19:00"
    ],
    "sun": null
  },
  "appointment_minutes": 60,
  "team": [
    {
      "name": "Dr. Sarah Jenkins",
      "role": "Lead Dentist, specialist in restorative and cosmetic dentistry (crowns, fillings, whitening)"
    },
    {
      "name": "Dr. Michael Vance",
      "role": "Orthodontist and general dentist (braces, alignment, general care)"
    }
  ],
  "insurances": [
    "Delta Dental",
    "Cigna",
    "MetLife",
    "Aetna",
    "Guardian",
    "Premier Access"
  ],
  "services": [
    {
      "name": "Routine cleaning and exam, with X-rays",
      "price": "$150–$250",
      "minutes": 45
    },
    {
      "name": "Fillings",
      "price": "$180–$350",
      "minutes": 45
    },
    {
      "name": "Root canals",
      "price": "$900–$1,400",
      "minutes": 60
    },
    {
      "name": "Crowns",
      "price": "$1,100–$1,800",
      "minutes": 60
    },
    {
      "name": "Teeth whitening",
      "price": null,
      "minutes": null
    },
    {
      "name": "Emergency tooth pain care and triage",
      "price": "$100 diagnostic fee",
      "minutes": 30
    }
  ]
}'::jsonb
)
on conflict (id) do nothing;
