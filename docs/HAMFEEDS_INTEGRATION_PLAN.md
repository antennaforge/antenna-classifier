# AntennaForge Portal Integration Plan

**antenna-classifier ↔ hamfeeds integration**
**Date:** 2026-03-07
**Status:** Draft

---

## 1. Vision

Logged-in hamfeeds users get a **"My Antenna"** menu item that opens the antenna-classifier dashboard enriched with their personal antenna workspace. Users can browse the 1,000+ NEC catalog, build their own antennas (form or AI-from-PDF), simulate, visualise in 3D, and save — all tied to their hamfeeds account. The AI generation feature is **admin-gated**: an admin must explicitly enable it per user from the admin dashboard.

The result: a user's NEC files become personal reference material for exploration and learning, while also letting them compare their real-world antenna performance (from WSPR/PSK/RBN propagation data already in hamfeeds) against simulated radiation patterns and SWR curves.

---

## 2. Architecture Principles

| Principle | Rationale |
|-----------|-----------|
| **Git submodule** | antenna-classifier keeps its own repo, release cadence, and CI. hamfeeds references a pinned commit. |
| **Minimal coupling** | hamfeeds only touches: (a) user↔antenna ownership in Postgres, (b) a thin auth-forwarding proxy header, (c) an admin feature-flag column. No hamfeeds Python code imports antenna-classifier code. |
| **Shared Postgres, separate tables** | antenna-classifier already uses `ac_user_antennas` (prefixed to avoid the existing `user_antennas` table). A new bridge table `ac_user_antenna_links` lives in the same `wspr` database and ties `users.id` → `ac_user_antennas.id`. |
| **Shared Docker network** | Already in place: `hamfeeds_wspr_network` (external). The `ac-dashboard` container is reachable from nginx at `http://ac-dashboard:8501`. |
| **Auth via trusted headers** | nginx already sets `X-Real-IP`, `X-Forwarded-For`, `X-Forwarded-Proto`. We add `X-HF-User-Id`, `X-HF-Callsign`, `X-HF-AI-Enabled` headers on the `/antenna/` proxy path, injected by a small Flask endpoint that validates the session cookie and forwards. |

---

## 3. Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        nginx (port 443)                        │
│                                                                 │
│  /antenna/*  ──► auth-check subrequest ──► ac-dashboard:8501   │
│  /*          ──► wspr_dashboard:8050 (hamfeeds Flask app)       │
└─────────────────────────────────────────────────────────────────┘
         │                          │
         │ X-HF-User-Id             │ Flask session
         │ X-HF-Callsign           │
         │ X-HF-AI-Enabled         │
         ▼                          ▼
┌──────────────────┐      ┌──────────────────────────┐
│  ac-dashboard    │      │  wspr_dashboard (Flask)   │
│  (FastAPI)       │      │                          │
│                  │      │  /api/auth/check ◄──nginx │
│  /api/my-antennas│      │  (returns user headers)  │
│  /api/catalog    │      │                          │
│  /api/simulate   │      │  /admin → AI toggle UI   │
└───────┬──────────┘      └───────────┬──────────────┘
        │                             │
        │  ac_user_antennas           │  users, ac_user_antenna_links,
        │                             │  user_feature_flags
        ▼                             ▼
┌─────────────────────────────────────────────────┐
│           Postgres (wspr database)              │
│                                                 │
│  users                (hamfeeds, existing)       │
│  user_feature_flags   (new, hamfeeds-owned)      │
│  ac_user_antenna_links(new, bridge table)        │
│  ac_user_antennas     (antenna-classifier-owned) │
└─────────────────────────────────────────────────┘
```

---

## 4. Git Submodule Setup

```bash
cd /home/herman/workspace/hamfeeds
git submodule add git@github.com:antennaforge/antenna-classifier.git submodules/antenna-classifier
git submodule update --init
```

hamfeeds `docker-compose.yml` gains an include or a new service definition referencing `submodules/antenna-classifier/docker-compose.yml`, or — simpler — duplicates the two service definitions (`nec-solver`, `dashboard`) directly.

**Recommended approach:** Reference services inline in hamfeeds `docker-compose.yml` using build context `submodules/antenna-classifier/`. This avoids nested-compose complexity while still pinning the submodule to a commit.

---

## 5. Database Schema Changes

### 5.1 User Feature Flags (hamfeeds-owned)

```sql
-- Migration: add_user_feature_flags.sql
CREATE TABLE IF NOT EXISTS user_feature_flags (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    feature     VARCHAR(100) NOT NULL,
    enabled     BOOLEAN DEFAULT FALSE,
    granted_by  INTEGER REFERENCES users(id),
    granted_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, feature)
);

CREATE INDEX IF NOT EXISTS idx_uff_user ON user_feature_flags(user_id);
CREATE INDEX IF NOT EXISTS idx_uff_feature ON user_feature_flags(feature);
```

Feature key for AI antenna generation: `my_antenna_ai`

### 5.2 User-Antenna Bridge Table (hamfeeds-owned)

```sql
-- Migration: add_ac_user_antenna_links.sql
CREATE TABLE IF NOT EXISTS ac_user_antenna_links (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    antenna_id      INTEGER NOT NULL,  -- references ac_user_antennas.id (no FK across ownership boundary)
    nec_filename    VARCHAR(255),      -- user-meaningful filename for reference
    is_primary      BOOLEAN DEFAULT FALSE,
    notes           TEXT DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, antenna_id)
);

CREATE INDEX IF NOT EXISTS idx_acual_user ON ac_user_antenna_links(user_id);
CREATE INDEX IF NOT EXISTS idx_acual_antenna ON ac_user_antenna_links(antenna_id);
```

### 5.3 ac_user_antennas Additions (antenna-classifier-owned)

Add a nullable `owner_user_id` column to `ac_user_antennas` for direct queries:

```sql
ALTER TABLE ac_user_antennas ADD COLUMN IF NOT EXISTS owner_user_id INTEGER;
CREATE INDEX IF NOT EXISTS idx_ac_ua_owner ON ac_user_antennas(owner_user_id);
```

This is denormalized (the canonical link is `ac_user_antenna_links`) but avoids cross-container JOINs for the 90% case.

---

## 6. Authentication Flow

### 6.1 nginx Auth Subrequest

nginx already proxies `/antenna/` to `ac-dashboard:8501`. We enhance it with `auth_request`:

```nginx
location /antenna/ {
    auth_request /internal/auth-check;
    auth_request_set $hf_user_id    $upstream_http_x_hf_user_id;
    auth_request_set $hf_callsign   $upstream_http_x_hf_callsign;
    auth_request_set $hf_ai_enabled $upstream_http_x_hf_ai_enabled;
  auth_request_set $hf_is_admin   $upstream_http_x_hf_is_admin;

    proxy_set_header X-HF-User-Id     $hf_user_id;
    proxy_set_header X-HF-Callsign    $hf_callsign;
    proxy_set_header X-HF-AI-Enabled  $hf_ai_enabled;
  proxy_set_header X-HF-Is-Admin    $hf_is_admin;

    proxy_pass http://ac-dashboard:8501/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Prefix /antenna;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;
}

location = /internal/auth-check {
    internal;
    proxy_pass http://wspr_dashboard:8050/api/auth/verify-antenna;
    proxy_pass_request_body off;
    proxy_set_header Content-Length "";
    proxy_set_header X-Original-URI $request_uri;
    proxy_set_header Cookie $http_cookie;
}
```

### 6.2 hamfeeds Auth Endpoint

New Flask endpoint (no CSRF needed — internal only):

```python
@app.route('/api/auth/verify-antenna')
@csrf.exempt
def verify_antenna_auth():
    """nginx auth_request subrequest — returns 200 with user headers or 401."""
    if not current_user.is_authenticated:
        return '', 401
    
    # Check if AI feature is enabled for this user
    ai_enabled = _has_feature_flag(current_user.id, 'my_antenna_ai')
    
    resp = make_response('', 200)
    resp.headers['X-HF-User-Id'] = str(current_user.id)
    resp.headers['X-HF-Callsign'] = current_user.callsign or ''
    resp.headers['X-HF-AI-Enabled'] = '1' if ai_enabled else '0'
    return resp
```

### 6.3 Anonymous Browsing (Optional)

If we want unauthenticated users to browse the NEC catalog (read-only) but not use "My Antenna", the auth-check returns 200 with empty `X-HF-User-Id`. The antenna-classifier frontend shows/hides the My Antenna panel based on whether a user ID is present.

**Decision point:** Full login-wall vs. catalog-public. _Recommend catalog-public_ — it's a great showcase. The My Antenna panel and AI buttons only appear when authenticated.

---

## 7. antenna-classifier Changes

### 7.1 Read Trusted Headers

In `dashboard.py`, read the forwarded user identity:

```python
def _get_hf_user(request: Request) -> dict | None:
    """Extract hamfeeds user from trusted proxy headers."""
    uid = request.headers.get("X-HF-User-Id")
    if not uid:
        return None
    return {
        "user_id": int(uid),
        "callsign": request.headers.get("X-HF-Callsign", ""),
        "ai_enabled": request.headers.get("X-HF-AI-Enabled") == "1",
    }
```

### 7.2 Scope My Antenna to User

All `/api/my-antennas` endpoints filter by `owner_user_id`:

- `list_antennas()` → `WHERE owner_user_id = %s`
- `create_antenna()` → sets `owner_user_id`
- `get/update/delete` → verify `owner_user_id` matches (authz)

### 7.3 Gate AI Endpoints

`POST /api/my-antennas/generate` and `POST /api/my-antennas/upload-pdf` check `hf_user["ai_enabled"]`. If `False`, return `403 {"error": "AI features not enabled for your account. Contact admin."}`.

### 7.4 Frontend Awareness

The index.html SPA receives user info via a new `GET /api/me` endpoint that returns the header data. The frontend:

- Hides "My Antennas" panel entirely if no user
- Hides "AI Generate" and "From PDF" buttons if `ai_enabled` is false
- Shows callsign in a header badge

---

## 8. hamfeeds Changes

### 8.1 Sidebar Navigation

In `templates/base.html`, add within the authenticated user section:

```html
{% if current_user.is_authenticated %}
<div class="nav-section">
    <div class="nav-section-title">Tools</div>
    <a href="/antenna/" class="nav-item {% if request.path.startswith('/antenna') %}active{% endif %}">
        <span class="icon">📐</span>My Antenna
    </a>
</div>
{% endif %}
```

### 8.2 Admin Feature Flag UI

Add a new section in `templates/admin.html` for managing per-user AI feature flags:

- Table of users with a toggle column for `my_antenna_ai`
- Bulk enable/disable actions
- Shows who granted the flag and when

### 8.3 Admin API Endpoints

```
GET  /api/admin/feature-flags?feature=my_antenna_ai  → list users + status
POST /api/admin/feature-flags                         → {user_id, feature, enabled}
```

### 8.4 Helper Function

```python
def _has_feature_flag(user_id: int, feature: str) -> bool:
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT enabled FROM user_feature_flags WHERE user_id = %s AND feature = %s",
            (user_id, feature)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return bool(row and row[0])
    except Exception:
        return False
```

---

## 9. Real-World ↔ Simulation Correlation

The unique hamfeeds value-add: comparing simulated antenna performance with real propagation data.

### 9.1 Overlay Feature (Future Phase)

Once a user has a simulated radiation pattern for their antenna and their callsign is in the WSPR/PSK spot database:

1. **Azimuthal comparison:** Plot simulated gain pattern (dBi by azimuth) overlaid with actual received-spot density by azimuth from WSPR data.
2. **Band-by-band SWR vs. actual activity:** Show SWR curve alongside the user's actual spotted bands — highlighting where their antenna works well vs. where the model says it should.
3. **Performance score:** A simple metric comparing simulated F/B ratio with observed F/B from propagation data (already computed in antenna performance module).

This is the "full picture" — simulation meets reality.

### 9.2 Data Flow

```
User's NEC file → ac-dashboard simulates → gain pattern, SWR curve
                                                  ↕
User's callsign → hamfeeds WSPR/PSK data → azimuth/band/SNR aggregates
                                                  ↕
                              Overlay visualization (future)
```

---

## 10. Implementation Phases

### Phase 1: Submodule + Auth Plumbing (1-2 days)

| Step | Owner | Description |
|------|-------|-------------|
| 1.1 | hamfeeds | Add `antenna-classifier` as git submodule |
| 1.2 | hamfeeds | Create `user_feature_flags` migration + helper |
| 1.3 | hamfeeds | Create `ac_user_antenna_links` migration |
| 1.4 | hamfeeds | Add `/api/auth/verify-antenna` endpoint |
| 1.5 | hamfeeds | Update nginx `location /antenna/` with `auth_request` |
| 1.6 | hamfeeds | Add "My Antenna" sidebar nav item |
| 1.7 | antenna-classifier | Add `owner_user_id` column to `ac_user_antennas` |
| 1.8 | antenna-classifier | Add `_get_hf_user()` header reader |
| 1.9 | antenna-classifier | Scope My Antenna CRUD to `owner_user_id` |
| 1.10 | antenna-classifier | Add `GET /api/me` endpoint returning user info |
| 1.11 | antenna-classifier | Gate AI endpoints on `X-HF-AI-Enabled` |

### Phase 2: Admin AI Toggle (1 day)

| Step | Owner | Description |
|------|-------|-------------|
| 2.1 | hamfeeds | Admin feature-flag API endpoints |
| 2.2 | hamfeeds | Admin UI section with user toggles |
| 2.3 | hamfeeds | Bulk enable/disable support |

### Phase 3: Frontend Polish (1 day)

| Step | Owner | Description |
|------|-------|-------------|
| 3.1 | antenna-classifier | Fetch `/api/me` on page load, show/hide panels |
| 3.2 | antenna-classifier | Show callsign badge in header |
| 3.3 | antenna-classifier | Disable AI buttons when `ai_enabled` is false |
| 3.4 | antenna-classifier | Friendly "request AI access" message pointing to admin |

### Phase 4: Real-World Correlation (future)

| Step | Owner | Description |
|------|-------|-------------|
| 4.1 | hamfeeds | API: `/api/operator/azimuth-profile?callsign=XX` |
| 4.2 | antenna-classifier | Overlay simulated pattern with WSPR azimuth data |
| 4.3 | antenna-classifier | Band SWR vs. actual spotted bands comparison |
| 4.4 | antenna-classifier | Combined performance score widget |

---

## 11. Docker Topology (Post-Integration)

```yaml
# hamfeeds/docker-compose.yml additions
services:
  # ... existing hamfeeds services ...

  ac-nec-solver:
    build:
      context: ./submodules/antenna-classifier/docker/nec_solver
    container_name: ac-nec-solver
    networks:
      - wspr_network
    healthcheck: ...

  ac-dashboard:
    build:
      context: ./submodules/antenna-classifier
      dockerfile: docker/dashboard/Dockerfile
    container_name: ac-dashboard
    environment:
      - NEC_DIR=/data/nec_files
      - NEC_SOLVER_URL=http://ac-nec-solver:8787
      - ROOT_PATH=/antenna
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=wspr
      - POSTGRES_USER=wspr_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - ${NEC_FILES_DIR:-./submodules/antenna-classifier/nec_files}:/data/nec_files:ro
      - ac_user_nec_files:/data/user_nec_files
    secrets:
      - openai_api_key
    networks:
      - wspr_network
    depends_on:
      ac-nec-solver:
        condition: service_healthy

volumes:
  ac_user_nec_files:

secrets:
  openai_api_key:
    file: ./secrets/openai_api_key
```

Both services now live on `wspr_network` directly (no external network needed) and share the same Postgres instance.

---

## 12. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Header spoofing | `X-HF-*` headers are set by nginx after `auth_request`. nginx strips any client-supplied `X-HF-*` headers. ac-dashboard only trusts these headers when `X-Forwarded-For` matches expected internal IP. |
| User data isolation | All My Antenna queries enforced with `WHERE owner_user_id = %s`. No user can access another's antennas. |
| AI cost control | AI feature is admin-gated and per-user. OpenAI key in Docker secrets, not env vars. |
| SQL injection | All queries use parameterized `%s` placeholders (psycopg2). |
| CSRF | antenna-classifier API is stateless (no cookies, user identified by trusted headers). |
| NEC file content | NEC content stored as text, never executed on the server — nec-solver runs in a separate container with no network egress. |

---

## 13. Migration Checklist

- [ ] Run `add_user_feature_flags.sql` against production Postgres
- [ ] Run `add_ac_user_antenna_links.sql` against production Postgres
- [ ] Run `ALTER TABLE ac_user_antennas ADD COLUMN ...` migration
- [ ] Verify nginx auth_request works (test: unauthenticated → 401, authenticated → 200 + headers)
- [ ] Verify ac-dashboard reads `X-HF-User-Id` correctly
- [ ] Verify AI endpoints return 403 when `X-HF-AI-Enabled` = 0
- [ ] Admin enables AI for test user → verify AI endpoints work
- [ ] Test catalog browsing as anonymous user
- [ ] Test My Antenna CRUD as authenticated user
- [ ] Verify user A cannot see user B's antennas
- [ ] Load test: antenna simulation under concurrent users

---

## 14. Cross-Repo Regression Testing

Whenever changes are made in either repository that touch integration surfaces (auth headers, API endpoints, database schema, Docker networking, nginx config), **run regression tests in both repos** before merging:

```bash
# antenna-classifier (from repo root)
PYTHONPATH=src python3 -m pytest tests/ -x -q

# hamfeeds (from repo root)
python3 -m pytest tests/ -x -q
```

Key areas that require cross-repo test runs:
- Changes to `X-HF-*` trusted header names or values
- Database schema changes to shared tables (`ac_user_antennas`, `user_feature_flags`, `ac_user_antenna_links`)
- Docker network or service name changes
- nginx proxy or `auth_request` configuration changes
- API contract changes on `/api/auth/verify-antenna` or `/api/me`

---

## 15. Open Questions

1. **Anonymous catalog access?** Recommend yes (catalog-public, My Antenna requires login). Settable via config if needed.
2. **Rate limiting AI calls?** Consider per-user rate limit (e.g., 10 AI generations/hour) to control OpenAI cost.
3. **NEC file size limit?** Current limit is implicit (OpenAI output). Should we cap stored NEC at 100KB?
4. **Submodule update cadence?** Pin to tagged releases. `git submodule update` in CI/CD.
5. **Shared NEC catalog vs. user-only?** Current plan: catalog is read-only shared (1,000+ files shipped with antenna-classifier). User antennas are private by default. Future: optional "publish to community" feature.
