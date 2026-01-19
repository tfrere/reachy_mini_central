---
title: Reachy Mini Central
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Reachy Mini Central

WebRTC signaling server for Reachy Mini baby camera.

## Features

- GStreamer-compatible WebRTC signaling protocol
- Producer/Consumer session management
- Command relay between robot and client
- Real-time status monitoring

## WebSocket Endpoint

Connect to: `wss://<space-url>/ws`

## Protocol

Implements the GStreamer webrtcsink/webrtcsrc signaling protocol.
