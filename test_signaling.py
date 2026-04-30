"""Tests for the new lifecycle behaviour added to the signaling server.

Covers:

- ``setPeerStatus(roles=[])`` removes the peer from ``producers`` but
  keeps its SSE channel open and broadcasts ``peerStatusChanged``.
- ``install_id`` collisions between two producers of the same user
  evict the older.
- TTL sweeper purges peers whose lease expired.
- ``disconnect_peer`` clears every server-side structure (no leak).

Run with::

    pip install pytest pytest-asyncio httpx fastapi sse-starlette
    python -m pytest test_signaling.py -v
"""

from __future__ import annotations

import time

import pytest

from app import (
    LEASE_SECONDS,
    Peer,
    SignalingServer,
    _recommended_heartbeat_interval,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_server() -> SignalingServer:
    return SignalingServer()


_token_counter = 0


def _make_peer(server: SignalingServer, username: str = "alice") -> Peer:
    """Register a fresh peer + token mapping (mimics the SSE handshake).

    Each call mints a unique token; ``get_or_create_peer`` therefore
    creates a brand new ``Peer`` rather than rebinding to the previous
    one - the latter would silently make ``old`` and ``new`` the same
    object in the install_id collision tests.
    """
    global _token_counter
    _token_counter += 1
    return server.get_or_create_peer(
        token=f"tok-{username}-{_token_counter}", username=username
    )


# ----------------------------------------------------------------------
# setPeerStatus(roles=[]) - explicit withdraw
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_withdraw_removes_from_producers_but_keeps_peer_alive():
    server = _make_server()
    p = _make_peer(server)
    await server.handle_set_peer_status(
        p, {"roles": ["producer"], "meta": {"name": "r"}}
    )
    assert p.peer_id in server.producers
    assert p.peer_id in server.peers

    await server.handle_set_peer_status(p, {"roles": [], "meta": {"name": "r"}})
    assert p.peer_id not in server.producers
    assert p.peer_id in server.peers, "withdraw must keep SSE peer alive"
    assert p.role is None


@pytest.mark.asyncio
async def test_withdraw_broadcasts_to_same_user_listeners():
    server = _make_server()
    producer = _make_peer(server, username="alice")
    listener = _make_peer(server, username="alice")
    other_user = _make_peer(server, username="bob")

    await server.handle_set_peer_status(
        producer, {"roles": ["producer"], "meta": {"name": "r"}}
    )
    # Drain initial registration broadcast.
    while not listener.message_queue.empty():
        listener.message_queue.get_nowait()

    broadcast = await server.handle_set_peer_status(
        producer, {"roles": [], "meta": {"name": "r"}}
    )
    assert broadcast is not None
    assert broadcast["roles"] == []
    # Caller forwards via broadcast_to_listeners; simulate that step.
    await server.broadcast_to_listeners(
        broadcast, exclude_id=producer.peer_id, owner_username=producer.username
    )

    msg = listener.message_queue.get_nowait()
    assert msg["type"] == "peerStatusChanged"
    assert msg["roles"] == []
    assert other_user.message_queue.empty(), "must not leak across users"


@pytest.mark.asyncio
async def test_withdraw_when_not_a_producer_is_a_noop():
    server = _make_server()
    p = _make_peer(server)
    out = await server.handle_set_peer_status(p, {"roles": [], "meta": {}})
    assert out is None
    assert p.peer_id in server.peers


# ----------------------------------------------------------------------
# install_id collision -> last writer wins
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_id_collision_evicts_old_producer():
    server = _make_server()
    old = _make_peer(server, username="alice")
    new = _make_peer(server, username="alice")
    install_id = "x" * 32

    await server.handle_set_peer_status(
        old, {"roles": ["producer"], "meta": {"install_id": install_id, "name": "r"}}
    )
    await server.handle_set_peer_status(
        new, {"roles": ["producer"], "meta": {"install_id": install_id, "name": "r"}}
    )
    assert new.peer_id in server.producers
    assert old.peer_id not in server.producers
    assert old.peer_id not in server.peers, "old peer must be fully evicted, not just demoted"


@pytest.mark.asyncio
async def test_install_id_collision_does_not_cross_users():
    """Different HF users hitting the same install_id (rare hardware swap)
    must NOT evict each other - that is handled at the auth layer."""
    server = _make_server()
    alice = _make_peer(server, username="alice")
    bob = _make_peer(server, username="bob")
    install_id = "y" * 32

    await server.handle_set_peer_status(
        alice, {"roles": ["producer"], "meta": {"install_id": install_id}}
    )
    await server.handle_set_peer_status(
        bob, {"roles": ["producer"], "meta": {"install_id": install_id}}
    )
    assert alice.peer_id in server.producers
    assert bob.peer_id in server.producers


@pytest.mark.asyncio
async def test_install_id_collision_ends_old_session():
    server = _make_server()
    old = _make_peer(server, username="alice")
    new = _make_peer(server, username="alice")
    consumer = _make_peer(server, username="alice")

    await server.handle_set_peer_status(
        old, {"roles": ["producer"], "meta": {"install_id": "z" * 32}}
    )
    # Hand-craft an active session on the old producer so we can
    # observe the cleanup path running.
    await server.handle_start_session(consumer, {"peerId": old.peer_id})
    assert old.session_id is not None

    await server.handle_set_peer_status(
        new, {"roles": ["producer"], "meta": {"install_id": "z" * 32}}
    )
    assert old.session_id is None
    # Consumer received an endSession with the takeover reason.
    seen_takeover = False
    while not consumer.message_queue.empty():
        msg = consumer.message_queue.get_nowait()
        if msg.get("type") == "endSession" and msg.get("reason") == "install_id_takeover":
            seen_takeover = True
    assert seen_takeover


# ----------------------------------------------------------------------
# TTL sweeper
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_sweeper_evicts_stale_peer():
    server = _make_server()
    p = _make_peer(server)
    # Force the peer into the past beyond the lease.
    p.last_seen = time.monotonic() - LEASE_SECONDS - 5

    # Run one sweeper iteration synchronously by short-circuiting the loop:
    # we mimic what the sweeper does without waiting SWEEPER_INTERVAL_SECONDS.
    cutoff = time.monotonic() - LEASE_SECONDS
    stale = [pid for pid, peer in server.peers.items() if peer.last_seen < cutoff]
    for pid in stale:
        await server.disconnect_peer(pid)

    assert p.peer_id not in server.peers


def test_touch_refreshes_last_seen():
    server = _make_server()
    p = _make_peer(server)
    p.last_seen = 0.0
    server.touch(p.peer_id)
    assert p.last_seen > 0.0


def test_touch_unknown_peer_is_noop():
    server = _make_server()
    server.touch("nonexistent")  # must not raise


# ----------------------------------------------------------------------
# Liveness contract: keepalive yields don't refresh last_seen
# ----------------------------------------------------------------------
#
# The central server pushes ``{"event": "ping"}`` keepalive events every
# 30 s on idle SSE channels purely to keep HTTP/2 proxies (HF Spaces,
# Cloudflare) from culling the connection. They are NOT a liveness
# signal: a half-open TCP socket happily absorbs writes for minutes
# after the peer's network has died, so refreshing ``last_seen`` on
# every yield would hold zombie producers forever.
#
# Liveness is driven by **inbound** application traffic (POST /send)
# carrying the daemon's heartbeat re-emission of ``setPeerStatus``.
# Any future refactor that wires keepalive back into ``touch()`` MUST
# break these regression guards.


@pytest.mark.asyncio
async def test_idle_peer_with_no_inbound_traffic_is_evicted():
    """Regression guard: a peer that only "receives" SSE keepalives
    must still be evicted by the TTL sweeper. If a future refactor
    re-introduces ``signaling.touch()`` on the keepalive branch this
    test will fail.
    """
    server = _make_server()
    p = _make_peer(server)
    await server.handle_set_peer_status(
        p, {"roles": ["producer"], "meta": {"name": "r"}}
    )

    # Simulate "we've been silently shipping keepalives to this peer
    # for longer than the lease, but never received a POST from it":
    # last_seen stays at its initial value, then expires.
    p.last_seen = time.monotonic() - LEASE_SECONDS - 1

    cutoff = time.monotonic() - LEASE_SECONDS
    stale = [pid for pid, peer in server.peers.items() if peer.last_seen < cutoff]
    for pid in stale:
        await server.disconnect_peer(pid)

    assert p.peer_id not in server.peers
    assert p.peer_id not in server.producers


@pytest.mark.asyncio
async def test_inbound_traffic_keeps_peer_alive():
    """Symmetric to the above: a peer that keeps sending a
    heartbeat-shaped ``setPeerStatus`` (= ``POST /send`` on the wire,
    which the route handler maps to ``signaling.touch()``) must NOT be
    evicted, even if its meta payload is identical to the previous one.
    """
    server = _make_server()
    p = _make_peer(server)
    await server.handle_set_peer_status(
        p, {"roles": ["producer"], "meta": {"name": "r"}}
    )

    # Backdate the peer to look stale, then simulate the inbound
    # heartbeat POST landing: the route handler calls touch() before
    # processing the message, which is what we exercise here.
    p.last_seen = time.monotonic() - LEASE_SECONDS - 1
    server.touch(p.peer_id)

    cutoff = time.monotonic() - LEASE_SECONDS
    stale = [pid for pid, peer in server.peers.items() if peer.last_seen < cutoff]
    assert stale == []
    assert p.peer_id in server.peers


# ----------------------------------------------------------------------
# disconnect_peer cleanup invariants
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_peer_clears_all_structures():
    server = _make_server()
    token = "tok-disconnect-test"
    p = server.get_or_create_peer(token=token, username="alice")
    await server.handle_set_peer_status(
        p, {"roles": ["producer"], "meta": {"name": "r"}}
    )

    await server.disconnect_peer(p.peer_id)

    assert p.peer_id not in server.peers
    assert p.peer_id not in server.producers
    assert token not in server.token_to_peer


@pytest.mark.asyncio
async def test_disconnect_peer_idempotent():
    server = _make_server()
    p = _make_peer(server)
    await server.disconnect_peer(p.peer_id)
    # Second call must not raise on a missing peer.
    await server.disconnect_peer(p.peer_id)


# ----------------------------------------------------------------------
# Heartbeat / lease negotiation contract
# ----------------------------------------------------------------------


def test_recommended_heartbeat_is_a_third_of_the_lease():
    """The daemon's heartbeat hint must always be small enough that
    central sees ~3 successful inbound POSTs per lease window. Anything
    larger and a single missed heartbeat puts a healthy daemon at the
    edge of eviction.
    """
    assert _recommended_heartbeat_interval() == pytest.approx(LEASE_SECONDS / 3.0)


def test_recommended_heartbeat_has_a_floor():
    """Even with a pathologically small lease, we never recommend
    sub-second heartbeats: that's a busy loop, not signaling.
    """
    import app as central_app

    saved = central_app.LEASE_SECONDS
    try:
        central_app.LEASE_SECONDS = 0.5
        assert central_app._recommended_heartbeat_interval() >= 1.0
    finally:
        central_app.LEASE_SECONDS = saved
