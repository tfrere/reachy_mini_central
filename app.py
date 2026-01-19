"""
Reachy Mini Central - WebRTC Signaling Server

This server implements the GStreamer WebRTC signaling protocol for:
- Robot (producer): Streams video/audio
- Client (consumer): Views the stream
- Command relay: Bidirectional commands between robot and client
- Media relay: Fallback when P2P fails
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Reachy Mini Central")


class PeerRole(Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    LISTENER = "listener"


@dataclass
class Peer:
    """Represents a connected peer (robot or client)."""
    peer_id: str
    websocket: WebSocket
    role: Optional[PeerRole] = None
    meta: dict = field(default_factory=dict)
    session_id: Optional[str] = None
    partner_id: Optional[str] = None  # Connected peer for session


class SignalingServer:
    """GStreamer-compatible WebRTC signaling server."""

    def __init__(self):
        self.peers: dict[str, Peer] = {}
        self.sessions: dict[str, tuple[str, str]] = {}  # session_id -> (producer_id, consumer_id)
        self.producers: dict[str, Peer] = {}  # producer_id -> Peer

    async def register_peer(self, websocket: WebSocket) -> Peer:
        """Register a new peer and send welcome message."""
        peer_id = str(uuid.uuid4())
        peer = Peer(peer_id=peer_id, websocket=websocket)
        self.peers[peer_id] = peer

        # Send welcome message (GStreamer protocol)
        await self.send_message(peer, {
            "type": "welcome",
            "peerId": peer_id
        })

        logger.info(f"Peer registered: {peer_id}")
        return peer

    async def unregister_peer(self, peer: Peer):
        """Clean up when peer disconnects."""
        peer_id = peer.peer_id

        # Clean up sessions
        if peer.session_id:
            await self.end_session(peer.session_id)

        # Remove from producers
        if peer_id in self.producers:
            del self.producers[peer_id]
            # Notify listeners that producer left
            await self.broadcast_peer_status(peer_id, None, removed=True)

        # Remove peer
        if peer_id in self.peers:
            del self.peers[peer_id]

        logger.info(f"Peer unregistered: {peer_id}")

    async def send_message(self, peer: Peer, message: dict):
        """Send JSON message to peer."""
        try:
            await peer.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send to {peer.peer_id}: {e}")

    async def broadcast_peer_status(self, peer_id: str, meta: Optional[dict], removed: bool = False):
        """Broadcast producer status to all listeners/consumers."""
        message = {
            "type": "peerStatusChanged",
            "peerId": peer_id,
            "roles": [] if removed else ["producer"],
            "meta": meta or {}
        }

        for p in self.peers.values():
            if p.peer_id != peer_id:
                await self.send_message(p, message)

    async def handle_set_peer_status(self, peer: Peer, message: dict):
        """Handle producer registration."""
        roles = message.get("roles", [])
        meta = message.get("meta", {})

        peer.meta = meta

        if "producer" in roles:
            peer.role = PeerRole.PRODUCER
            self.producers[peer.peer_id] = peer
            logger.info(f"Producer registered: {peer.peer_id} with meta: {meta}")

            # Broadcast to all connected peers
            await self.broadcast_peer_status(peer.peer_id, meta)

        elif "listener" in roles:
            peer.role = PeerRole.LISTENER
            # Send current producers list
            for prod_id, prod in self.producers.items():
                await self.send_message(peer, {
                    "type": "peerStatusChanged",
                    "peerId": prod_id,
                    "roles": ["producer"],
                    "meta": prod.meta
                })

    async def handle_start_session(self, peer: Peer, message: dict):
        """Handle consumer requesting session with producer."""
        producer_id = message.get("peerId")

        if producer_id not in self.producers:
            await self.send_message(peer, {
                "type": "error",
                "details": f"Producer {producer_id} not found"
            })
            return

        producer = self.producers[producer_id]
        session_id = str(uuid.uuid4())

        # Store session
        self.sessions[session_id] = (producer_id, peer.peer_id)
        peer.session_id = session_id
        peer.partner_id = producer_id
        peer.role = PeerRole.CONSUMER
        producer.session_id = session_id
        producer.partner_id = peer.peer_id

        # Notify both peers
        await self.send_message(peer, {
            "type": "sessionStarted",
            "peerId": producer_id,
            "sessionId": session_id
        })

        await self.send_message(producer, {
            "type": "startSession",
            "peerId": peer.peer_id,
            "sessionId": session_id
        })

        logger.info(f"Session started: {session_id} between producer {producer_id} and consumer {peer.peer_id}")

    async def handle_peer_message(self, peer: Peer, message: dict):
        """Relay SDP/ICE messages between peers in a session."""
        session_id = message.get("sessionId")

        if session_id not in self.sessions:
            logger.warning(f"Unknown session: {session_id}")
            return

        # Find the other peer in the session
        producer_id, consumer_id = self.sessions[session_id]
        target_id = consumer_id if peer.peer_id == producer_id else producer_id

        if target_id not in self.peers:
            logger.warning(f"Target peer not found: {target_id}")
            return

        target = self.peers[target_id]

        # Relay the message
        relay_message = {
            "type": "peer",
            "sessionId": session_id,
            **{k: v for k, v in message.items() if k not in ["type", "sessionId"]}
        }

        await self.send_message(target, relay_message)
        logger.debug(f"Relayed peer message from {peer.peer_id} to {target_id}")

    async def handle_end_session(self, peer: Peer, message: dict):
        """Handle session end request."""
        session_id = message.get("sessionId")
        if session_id:
            await self.end_session(session_id)

    async def end_session(self, session_id: str):
        """End a session and notify peers."""
        if session_id not in self.sessions:
            return

        producer_id, consumer_id = self.sessions[session_id]

        # Notify peers
        for peer_id in [producer_id, consumer_id]:
            if peer_id in self.peers:
                peer = self.peers[peer_id]
                peer.session_id = None
                peer.partner_id = None
                await self.send_message(peer, {
                    "type": "endSession",
                    "sessionId": session_id
                })

        del self.sessions[session_id]
        logger.info(f"Session ended: {session_id}")

    async def handle_command(self, peer: Peer, message: dict):
        """Handle custom commands (robot control, etc.)."""
        if not peer.partner_id or peer.partner_id not in self.peers:
            logger.warning(f"No partner for command relay from {peer.peer_id}")
            return

        target = self.peers[peer.partner_id]

        # Relay command to partner
        await self.send_message(target, {
            "type": "command",
            "from": peer.peer_id,
            "data": message.get("data", {})
        })

        logger.debug(f"Command relayed from {peer.peer_id} to {target.peer_id}")

    async def handle_message(self, peer: Peer, raw_message: str):
        """Route incoming messages to appropriate handlers."""
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type", "")

            logger.debug(f"Received from {peer.peer_id}: {msg_type}")

            if msg_type == "setPeerStatus":
                await self.handle_set_peer_status(peer, message)
            elif msg_type == "startSession":
                await self.handle_start_session(peer, message)
            elif msg_type == "peer":
                await self.handle_peer_message(peer, message)
            elif msg_type == "endSession":
                await self.handle_end_session(peer, message)
            elif msg_type == "command":
                await self.handle_command(peer, message)
            elif msg_type == "list":
                # Return list of producers
                producers_list = [
                    {"peerId": p.peer_id, "meta": p.meta}
                    for p in self.producers.values()
                ]
                await self.send_message(peer, {
                    "type": "list",
                    "producers": producers_list
                })
            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {peer.peer_id}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")


# Global signaling server instance
signaling = SignalingServer()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for signaling."""
    await websocket.accept()
    peer = await signaling.register_peer(websocket)

    try:
        while True:
            message = await websocket.receive_text()
            await signaling.handle_message(peer, message)
    except WebSocketDisconnect:
        logger.info(f"Peer disconnected: {peer.peer_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {peer.peer_id}: {e}")
    finally:
        await signaling.unregister_peer(peer)


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
            code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>Reachy Mini Central</h1>
        <div class="status">
            <p><strong>Status:</strong> Running</p>
            <p><strong>Connected Peers:</strong> {len(signaling.peers)}</p>
            <p><strong>Active Producers:</strong> {len(signaling.producers)}</p>
            <p><strong>Active Sessions:</strong> {len(signaling.sessions)}</p>
        </div>
        <h2>WebSocket Endpoint</h2>
        <p>Connect to: <code>wss://{{host}}/ws</code></p>
        <h2>Protocol</h2>
        <p>This server implements the GStreamer WebRTC signaling protocol.</p>
    </body>
    </html>
    """)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "peers": len(signaling.peers),
        "producers": len(signaling.producers),
        "sessions": len(signaling.sessions)
    }


@app.get("/api/producers")
async def list_producers():
    """List all registered producers."""
    return {
        "producers": [
            {"peerId": p.peer_id, "meta": p.meta}
            for p in signaling.producers.values()
        ]
    }
