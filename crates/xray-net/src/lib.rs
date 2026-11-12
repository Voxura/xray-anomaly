//! Native Pure-Rust P2P Co-Op Networking for X-Ray Engine (`xray-net`).
//!
//! Provides a 3-tiered connection architecture:
//! - **Tier 1: Native IPv6 Direct P2P** — Zero NAT, lowest latency direct UDP socket connections.
//! - **Tier 2: STUN Hole Punching** — RFC 5389 reflexive address discovery and simultaneous UDP hole punching across IPv4 CGNATs.
//! - **Tier 3: In-Engine Fallback Relay** — Transparent packet relaying for strict Symmetric NATs and mobile hotspots without third-party VPNs.

pub mod protocol;
pub mod relay;
pub mod reliability;
pub mod session;
pub mod stun;
pub mod transport;

pub use protocol::{
    flags, ConnectChallenge, ConnectRequest, ConnectResponse, EntityDelta, Heartbeat, PacketHeader,
    PacketType, PROTOCOL_MAGIC, PROTOCOL_VERSION,
};
pub use relay::{RelayClient, RelayHeader, RelayServer, RELAY_HEADER_SIZE, RELAY_MAGIC};
pub use reliability::ReliabilityTracker;
pub use session::{ClientSession, ClientState, HostSession, PeerInfo, SessionRole};
pub use stun::{
    build_binding_request, parse_binding_response, HolePunchCoordinator, HolePunchState, NatType,
    ATTR_MAPPED_ADDRESS, ATTR_XOR_MAPPED_ADDRESS, BINDING_REQUEST, BINDING_RESPONSE,
    DEFAULT_STUN_SERVERS, STUN_MAGIC_COOKIE,
};
pub use transport::{TransportConfig, UdpTransport};
