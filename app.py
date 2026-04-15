"""
Reachy Mini Central - WebRTC Signaling Server

This server implements the GStreamer WebRTC signaling protocol using:
- SSE (Server-Sent Events) for server-to-client messages
- HTTP POST for client-to-server messages

This works reliably through HTTP/2 proxies like HuggingFace Spaces.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Reachy Mini Central")

# Add CORS middleware for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for validated tokens (token -> username)
token_cache: dict[str, str] = {}

# Rate limiting: username -> (request_count, window_start)
rate_limit_cache: dict[str, tuple[int, float]] = {}
RATE_LIMIT_REQUESTS = 100  # Max requests per window
RATE_LIMIT_WINDOW = 60  # Window in seconds


def check_rate_limit(username: str) -> bool:
    """Check if user is rate limited. Returns True if allowed, False if limited."""
    now = time.time()

    if username not in rate_limit_cache:
        rate_limit_cache[username] = (1, now)
        return True

    count, window_start = rate_limit_cache[username]

    # Reset window if expired
    if now - window_start > RATE_LIMIT_WINDOW:
        rate_limit_cache[username] = (1, now)
        return True

    # Check limit
    if count >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for user: {username}")
        return False

    # Increment counter
    rate_limit_cache[username] = (count + 1, window_start)
    return True


async def validate_hf_token(token: str) -> Optional[str]:
    """Validate HuggingFace token and return username if valid."""
    if not token:
        return None

    # Check cache first
    if token in token_cache:
        return token_cache[token]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                username = data.get("name", "unknown")
                token_cache[token] = username
                logger.info(f"Token validated for user: {username}")
                return username
            else:
                logger.warning(f"Token validation failed: {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        return None


@dataclass
class Peer:
    """Represents a connected peer (robot or client)."""
    peer_id: str
    username: str
    role: Optional[str] = None
    meta: dict = field(default_factory=dict)
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    connected: bool = True
    session_id: Optional[str] = None
    partner_id: Optional[str] = None


class SignalingServer:
    """HTTP-based WebRTC signaling server."""

    def __init__(self):
        self.peers: dict[str, Peer] = {}
        self.sessions: dict[str, tuple[str, str]] = {}  # session_id -> (producer_id, consumer_id)
        self.producers: dict[str, Peer] = {}  # producer_id -> Peer
        # Token to peer_id mapping for reconnection
        self.token_to_peer: dict[str, str] = {}

    def get_or_create_peer(self, token: str, username: str) -> Peer:
        """Get existing peer or create new one."""
        # Check if this token already has a peer
        if token in self.token_to_peer:
            peer_id = self.token_to_peer[token]
            if peer_id in self.peers:
                peer = self.peers[peer_id]
                peer.connected = True
                logger.info(f"Peer reconnected: {peer_id}")
                return peer

        # Create new peer
        peer_id = str(uuid.uuid4())
        peer = Peer(peer_id=peer_id, username=username)
        self.peers[peer_id] = peer
        self.token_to_peer[token] = peer_id
        logger.info(f"New peer created: {peer_id} for user {username}")
        return peer

    async def send_to_peer(self, peer_id: str, message: dict):
        """Queue a message for a peer."""
        if peer_id in self.peers:
            await self.peers[peer_id].message_queue.put(message)

    async def broadcast_to_listeners(self, message: dict, exclude_id: str = None, owner_username: str = None):
        """Send message to connected peers with same owner, except the sender."""
        for peer_id, peer in self.peers.items():
            if peer_id != exclude_id and peer.connected:
                # If owner specified, only send to peers with same username
                if owner_username and peer.username != owner_username:
                    continue
                await peer.message_queue.put(message)

    def handle_set_peer_status(self, peer: Peer, message: dict):
        """Handle peer status update (producer/listener registration)."""
        roles = message.get("roles", [])
        meta = message.get("meta", {})
        peer.meta = meta

        if "producer" in roles:
            peer.role = "producer"
            self.producers[peer.peer_id] = peer
            logger.info(f"Producer registered: {peer.peer_id} with meta: {meta}")
            return {"type": "peerStatusChanged", "peerId": peer.peer_id, "roles": ["producer"], "meta": meta}
        elif "listener" in roles:
            peer.role = "listener"
            logger.info(f"Listener registered: {peer.peer_id}")
        return None

    def get_producers_list(self, username: str) -> list:
        """Get list of producers owned by the given user."""
        return [
            {"id": p.peer_id, "meta": p.meta}
            for p in self.producers.values()
            if p.connected and p.username == username
        ]

    async def handle_start_session(self, peer: Peer, message: dict) -> dict:
        """Handle session start request."""
        producer_id = message.get("peerId")

        if producer_id not in self.producers:
            return {"type": "error", "details": f"Producer {producer_id} not found"}

        producer = self.producers[producer_id]

        # Security: verify the user owns this producer
        if producer.username != peer.username:
            logger.warning(f"User {peer.username} tried to access producer owned by {producer.username}")
            return {"type": "error", "details": "Access denied: you don't own this robot"}

        # Concurrency gate: reject if producer already has an active session.
        # The existing consumer's app name is read from its meta (set via setPeerStatus).
        if producer.session_id is not None:
            active_app = "another app"
            if producer.partner_id and producer.partner_id in self.peers:
                active_app = self.peers[producer.partner_id].meta.get("name") or active_app
            logger.info(
                f"Rejected session: producer {producer_id} busy with '{active_app}' "
                f"(requested by {peer.peer_id}, app={peer.meta.get('name')!r})"
            )
            return {
                "type": "sessionRejected",
                "reason": "robot_busy",
                "peerId": producer_id,
                "activeApp": active_app,
            }

        session_id = str(uuid.uuid4())

        # Store session
        self.sessions[session_id] = (producer_id, peer.peer_id)
        peer.session_id = session_id
        peer.partner_id = producer_id
        peer.role = "consumer"
        producer.session_id = session_id
        producer.partner_id = peer.peer_id

        # Notify producer
        await self.send_to_peer(producer_id, {
            "type": "startSession",
            "peerId": peer.peer_id,
            "sessionId": session_id
        })

        logger.info(f"Session started: {session_id}")
        return {"type": "sessionStarted", "peerId": producer_id, "sessionId": session_id}

    async def handle_peer_message(self, peer: Peer, message: dict):
        """Relay SDP/ICE messages between peers."""
        session_id = message.get("sessionId")

        if session_id not in self.sessions:
            logger.warning(f"Unknown session: {session_id}")
            return

        producer_id, consumer_id = self.sessions[session_id]
        target_id = consumer_id if peer.peer_id == producer_id else producer_id

        # Relay the message
        relay_message = {
            "type": "peer",
            "sessionId": session_id,
            **{k: v for k, v in message.items() if k not in ["type", "sessionId"]}
        }
        await self.send_to_peer(target_id, relay_message)
        logger.debug(f"Relayed peer message from {peer.peer_id} to {target_id}")

    async def handle_end_session(self, session_id: str):
        """End a session."""
        if session_id not in self.sessions:
            return

        producer_id, consumer_id = self.sessions[session_id]

        for peer_id in [producer_id, consumer_id]:
            if peer_id in self.peers:
                peer = self.peers[peer_id]
                peer.session_id = None
                peer.partner_id = None
                await self.send_to_peer(peer_id, {"type": "endSession", "sessionId": session_id})

        del self.sessions[session_id]
        logger.info(f"Session ended: {session_id}")

    async def handle_message(self, peer: Peer, message: dict) -> Optional[dict]:
        """Process incoming message and return response if any."""
        msg_type = message.get("type", "")
        logger.debug(f"Received from {peer.peer_id}: {msg_type}")

        if msg_type == "setPeerStatus":
            broadcast = self.handle_set_peer_status(peer, message)
            if broadcast:
                # Only notify users with same username (owner)
                await self.broadcast_to_listeners(broadcast, exclude_id=peer.peer_id, owner_username=peer.username)
            return None

        elif msg_type == "list":
            return {"type": "list", "producers": self.get_producers_list(peer.username)}

        elif msg_type == "startSession":
            return await self.handle_start_session(peer, message)

        elif msg_type == "peer":
            await self.handle_peer_message(peer, message)
            return None

        elif msg_type == "endSession":
            await self.handle_end_session(message.get("sessionId"))
            return None

        else:
            logger.warning(f"Unknown message type: {msg_type}")
            return None

    def disconnect_peer(self, peer_id: str):
        """Mark peer as disconnected."""
        if peer_id in self.peers:
            peer = self.peers[peer_id]
            peer.connected = False

            # Remove from producers if applicable
            if peer_id in self.producers:
                del self.producers[peer_id]
                logger.info(f"Producer disconnected: {peer_id}")

            logger.info(f"Peer disconnected: {peer_id}")


# Global signaling server instance
signaling = SignalingServer()


@app.get("/events")
async def events(request: Request, token: str = Query(...)):
    """SSE endpoint for receiving messages from server."""
    logger.info(f"SSE connection request with token: {token[:20]}...")

    username = await validate_hf_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not check_rate_limit(username):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    peer = signaling.get_or_create_peer(token, username)

    async def event_generator() -> AsyncGenerator[dict, None]:
        # Send welcome message with username for client info
        yield {"event": "message", "data": json.dumps({"type": "welcome", "peerId": peer.peer_id, "username": username})}

        # Send current producers list for listeners (filtered by owner)
        yield {"event": "message", "data": json.dumps({"type": "list", "producers": signaling.get_producers_list(username)})}

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(peer.message_queue.get(), timeout=30.0)
                    yield {"event": "message", "data": json.dumps(message)}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}

        finally:
            signaling.disconnect_peer(peer.peer_id)

    return EventSourceResponse(event_generator())


@app.post("/send")
async def send_message(request: Request, token: str = Query(...)):
    """HTTP POST endpoint for sending messages to server."""
    username = await validate_hf_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not check_rate_limit(username):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    # Get or reconnect peer
    if token not in signaling.token_to_peer:
        raise HTTPException(status_code=400, detail="Connect to /events first")

    peer_id = signaling.token_to_peer[token]
    if peer_id not in signaling.peers:
        raise HTTPException(status_code=400, detail="Peer not found")

    peer = signaling.peers[peer_id]

    body = await request.json()
    response = await signaling.handle_message(peer, body)

    return response or {"status": "ok"}


@app.get("/")
async def root():
    """Simple status page."""
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reachy Mini Central</title>
        <style>
            body {{ font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            .status {{ padding: 10px; background: #e8f5e9; border-radius: 5px; }}
            .security {{ padding: 10px; background: #e3f2fd; border-radius: 5px; margin-top: 10px; }}
            code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>Reachy Mini Central</h1>
        <div class="status">
            <p><strong>Status:</strong> Running</p>
            <p><strong>Connected Peers:</strong> {len([p for p in signaling.peers.values() if p.connected])}</p>
            <p><strong>Active Producers:</strong> {len(signaling.producers)}</p>
            <p><strong>Active Sessions:</strong> {len(signaling.sessions)}</p>
        </div>
        <div class="security">
            <p><strong>Security:</strong></p>
            <ul>
                <li>HuggingFace token authentication required</li>
                <li>Owner-based filtering: users only see their own robots</li>
                <li>Rate limiting: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s per user</li>
            </ul>
        </div>
        <h2>Endpoints</h2>
        <ul>
            <li><code>GET /events?token=...</code> - SSE stream for receiving messages</li>
            <li><code>POST /send?token=...</code> - Send messages to server</li>
        </ul>
        <h2>Protocol</h2>
        <p>This server implements the GStreamer WebRTC signaling protocol over HTTP/SSE.</p>
    </body>
    </html>
    """)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "peers": len([p for p in signaling.peers.values() if p.connected]),
        "producers": len(signaling.producers),
        "sessions": len(signaling.sessions)
    }
