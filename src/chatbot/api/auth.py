"""Request authentication -> :class:`Principal` (docs/13 §3: group memberships are
resolved per request and are the input to the ACL pre-filter).

Two modes, chosen with ``CHATBOT_AUTH_MODE``:

* ``gateway`` (default): an upstream API gateway / IdP proxy has authenticated the
  user and forwards ``X-User-Id``, ``X-User-Groups`` (comma separated) and
  ``X-Tenant-Id``. The service must only be reachable through that gateway; set
  ``CHATBOT_GATEWAY_SECRET`` so requests must also carry a matching
  ``X-Gateway-Secret`` header.
* ``api_key``: ``Authorization: Bearer <key>`` looked up in ``CHATBOT_API_KEYS``, a JSON
  object ``{"<key>": {"user_id": ..., "groups": [...], "tenant_id": ..., "roles": [...]}}``.
"""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from chatbot.core.types import Principal


@dataclass(eq=False)
class Authenticator:
    mode: str = "gateway"
    gateway_secret: str | None = None
    api_keys: dict[str, Principal] | None = None

    @classmethod
    def from_env(cls) -> Authenticator:
        mode = os.environ.get("CHATBOT_AUTH_MODE", "gateway")
        keys: dict[str, Principal] = {}
        raw = os.environ.get("CHATBOT_API_KEYS")
        if raw:
            for key, spec in json.loads(raw).items():
                keys[key] = Principal(
                    user_id=spec["user_id"],
                    groups=frozenset(spec.get("groups", [])),
                    tenant_id=spec.get("tenant_id", "default"),
                    roles=frozenset(spec.get("roles", [])),
                )
        return cls(
            mode=mode, gateway_secret=os.environ.get("CHATBOT_GATEWAY_SECRET"), api_keys=keys
        )

    def __call__(self, request: Request) -> Principal:
        if self.mode == "api_key":
            header = request.headers.get("authorization", "")
            token = header.removeprefix("Bearer ").strip()
            for key, principal in (self.api_keys or {}).items():
                if token and hmac.compare_digest(token, key):
                    return principal
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
        if self.gateway_secret is not None:
            supplied = request.headers.get("x-gateway-secret", "")
            if not hmac.compare_digest(supplied, self.gateway_secret):
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "untrusted caller")
        user = request.headers.get("x-user-id", "").strip()
        if not user:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing X-User-Id")
        groups = frozenset(
            g.strip() for g in request.headers.get("x-user-groups", "").split(",") if g.strip()
        )
        roles = frozenset(
            r.strip() for r in request.headers.get("x-user-roles", "").split(",") if r.strip()
        )
        tenant = request.headers.get("x-tenant-id", "default").strip() or "default"
        return Principal(user_id=user, groups=groups, tenant_id=tenant, roles=roles)
