//! Co-op session management and client/host state machines.

#![allow(clippy::collapsible_if)]

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use crate::protocol::{
    flags, ConnectChallenge, ConnectRequest, ConnectResponse, EntityDelta, Heartbeat,
    PacketHeader, PacketType,
};
use crate::reliability::ReliabilityTracker;
use crate::transport::{TransportConfig, UdpTransport};

pub const HEARTBEAT_INTERVAL: Duration = Duration::from_millis(500);
pub const TIMEOUT_THRESHOLD: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRole {
    Host,
    Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientState {
    Disconnected,
    SendingRequest,
    Challenged,
    Connected,
    DisconnectedTimeout,
}

pub struct PeerInfo {
    pub slot: u8,
    pub lan_id: u64,
    pub nickname: String,
    pub endpoint: SocketAddr,
    pub last_seen: Instant,
    pub reliability: ReliabilityTracker,
}

/// Manages a co-op host session accepting connections from peer players.
pub struct HostSession {
    transport: UdpTransport,
    session_id: u64,
    tick_rate: u8,
    peer: Option<PeerInfo>,
    expected_cookie: u64,
}

impl HostSession {
    pub fn new(port: u16, use_ipv6: bool) -> std::io::Result<Self> {
        let transport = UdpTransport::bind(TransportConfig {
            port,
            use_ipv6,
            non_blocking: true,
        })?;

        Ok(Self {
            transport,
            session_id: 0x54414c4b45520001, // 'TALKER\0\1'
            tick_rate: 60,
            peer: None,
            expected_cookie: 0xfeed_cafe_1234_5678,
        })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.transport.local_addr()
    }

    pub fn has_peer(&self) -> bool {
        self.peer.is_some()
    }

    pub fn peer_nickname(&self) -> Option<&str> {
        self.peer.as_ref().map(|p| p.nickname.as_str())
    }

    pub fn peer_lan_id(&self) -> Option<u64> {
        self.peer.as_ref().map(|p| p.lan_id)
    }

    /// Ticks the host session, processing incoming packets and connection handshakes.
    pub fn update(&mut self) -> std::io::Result<Vec<EntityDelta>> {
        let mut deltas = Vec::new();

        while let Some((header, payload, addr)) = self.transport.poll_packet()? {
            match header.packet_type {
                PacketType::ConnectRequest => {
                    if let Ok(req) = ConnectRequest::read_from(payload) {
                        // Generate challenge response
                        let challenge = ConnectChallenge {
                            cookie: self.expected_cookie,
                        };
                        let challenge_hdr = PacketHeader::new(
                            PacketType::ConnectChallenge,
                            0,
                            0,
                            header.sequence,
                            0,
                        );
                        let mut msg = Vec::new();
                        challenge_hdr.write_to(&mut msg);
                        challenge.write_to(&mut msg);
                        self.transport.send_to(&msg, addr)?;

                        // Prepare or update peer record
                        let mut rel = ReliabilityTracker::new();
                        rel.on_packet_received(header.sequence);
                        self.peer = Some(PeerInfo {
                            slot: 1,
                            lan_id: req.lan_id,
                            nickname: req.nickname,
                            endpoint: addr,
                            last_seen: Instant::now(),
                            reliability: rel,
                        });
                    }
                }
                PacketType::ConnectChallenge => {
                    // Client answering challenge
                    if let Ok(challenge) = ConnectChallenge::read_from(payload) {
                        if challenge.cookie == self.expected_cookie {
                            if let Some(peer) = &mut self.peer {
                                peer.endpoint = addr;
                                peer.last_seen = Instant::now();

                                let (ack, ack_bits) = peer.reliability.generate_ack();
                                let resp = ConnectResponse {
                                    accepted: true,
                                    assigned_slot: 1,
                                    session_id: self.session_id,
                                    tick_rate: self.tick_rate,
                                };
                                let resp_hdr = PacketHeader::new(
                                    PacketType::ConnectResponse,
                                    flags::RELIABLE | flags::ACK_REQUIRED,
                                    peer.reliability.next_sequence(),
                                    ack,
                                    ack_bits,
                                );
                                let mut msg = Vec::new();
                                resp_hdr.write_to(&mut msg);
                                resp.write_to(&mut msg);
                                self.transport.send_to(&msg, addr)?;
                            }
                        }
                    }
                }
                PacketType::Heartbeat => {
                    if let Some(peer) = &mut self.peer {
                        if peer.endpoint == addr {
                            peer.last_seen = Instant::now();
                            peer.reliability.on_packet_received(header.sequence);
                            peer.reliability.process_incoming_ack(header.ack, header.ack_bits);

                            // Reply with pong heartbeat
                            let (ack, ack_bits) = peer.reliability.generate_ack();
                            let hb = Heartbeat {
                                client_time_ms: u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]),
                                server_time_ms: 0,
                            };
                            let hb_hdr = PacketHeader::new(
                                PacketType::Heartbeat,
                                0,
                                peer.reliability.next_sequence(),
                                ack,
                                ack_bits,
                            );
                            let mut msg = Vec::new();
                            hb_hdr.write_to(&mut msg);
                            hb.write_to(&mut msg);
                            self.transport.send_to(&msg, addr)?;
                        }
                    }
                }
                PacketType::EntityDelta => {
                    if let Some(peer) = &mut self.peer {
                        if peer.endpoint == addr {
                            peer.last_seen = Instant::now();
                            peer.reliability.on_packet_received(header.sequence);
                            peer.reliability.process_incoming_ack(header.ack, header.ack_bits);

                            if let Ok(delta) = EntityDelta::read_from(payload) {
                                deltas.push(delta);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(deltas)
    }

    /// Sends an entity delta update to the connected peer.
    pub fn send_entity_delta(&mut self, delta: &EntityDelta) -> std::io::Result<()> {
        if let Some(peer) = &mut self.peer {
            let (ack, ack_bits) = peer.reliability.generate_ack();
            let seq = peer.reliability.next_sequence();
            let header = PacketHeader::new(
                PacketType::EntityDelta,
                0,
                seq,
                ack,
                ack_bits,
            );
            let mut msg = Vec::new();
            header.write_to(&mut msg);
            delta.write_to(&mut msg);
            self.transport.send_to(&msg, peer.endpoint)?;
        }
        Ok(())
    }
}

/// Manages a co-op client session connected to a host.
pub struct ClientSession {
    transport: UdpTransport,
    host_addr: SocketAddr,
    lan_id: u64,
    nickname: String,
    state: ClientState,
    assigned_slot: Option<u8>,
    session_id: Option<u64>,
    reliability: ReliabilityTracker,
    last_sent_request: Instant,
    last_heard: Instant,
}

impl ClientSession {
    pub fn connect(host_addr: SocketAddr, lan_id: u64, nickname: &str, use_ipv6: bool) -> std::io::Result<Self> {
        let transport = UdpTransport::bind(TransportConfig {
            port: 0,
            use_ipv6,
            non_blocking: true,
        })?;

        let mut session = Self {
            transport,
            host_addr,
            lan_id,
            nickname: nickname.to_string(),
            state: ClientState::SendingRequest,
            assigned_slot: None,
            session_id: None,
            reliability: ReliabilityTracker::new(),
            last_sent_request: Instant::now() - Duration::from_secs(1),
            last_heard: Instant::now(),
        };

        session.send_connect_request()?;
        Ok(session)
    }

    pub fn state(&self) -> ClientState {
        self.state
    }

    pub fn assigned_slot(&self) -> Option<u8> {
        self.assigned_slot
    }

    fn send_connect_request(&mut self) -> std::io::Result<()> {
        let req = ConnectRequest {
            protocol_version: crate::protocol::PROTOCOL_VERSION,
            lan_id: self.lan_id,
            nickname: self.nickname.clone(),
        };
        let (ack, ack_bits) = self.reliability.generate_ack();
        let hdr = PacketHeader::new(
            PacketType::ConnectRequest,
            flags::RELIABLE,
            self.reliability.next_sequence(),
            ack,
            ack_bits,
        );
        let mut msg = Vec::new();
        hdr.write_to(&mut msg);
        req.write_to(&mut msg);

        self.last_sent_request = Instant::now();
        self.transport.send_to(&msg, self.host_addr)?;
        Ok(())
    }

    /// Ticks the client session, advancing connection state and processing packets.
    pub fn update(&mut self) -> std::io::Result<Vec<EntityDelta>> {
        let mut deltas = Vec::new();

        // Resend connection request if still pending
        if self.state == ClientState::SendingRequest && self.last_sent_request.elapsed() > Duration::from_millis(250) {
            self.send_connect_request()?;
        }

        while let Some((header, payload, addr)) = self.transport.poll_packet()? {
            if addr != self.host_addr {
                continue;
            }

            self.last_heard = Instant::now();
            self.reliability.on_packet_received(header.sequence);
            self.reliability.process_incoming_ack(header.ack, header.ack_bits);

            match header.packet_type {
                PacketType::ConnectChallenge => {
                    if let Ok(challenge) = ConnectChallenge::read_from(payload) {
                        self.state = ClientState::Challenged;

                        // Answer challenge back to host
                        let (ack, ack_bits) = self.reliability.generate_ack();
                        let hdr = PacketHeader::new(
                            PacketType::ConnectChallenge,
                            0,
                            self.reliability.next_sequence(),
                            ack,
                            ack_bits,
                        );
                        let mut msg = Vec::new();
                        hdr.write_to(&mut msg);
                        challenge.write_to(&mut msg);
                        self.transport.send_to(&msg, self.host_addr)?;
                    }
                }
                PacketType::ConnectResponse => {
                    if let Ok(resp) = ConnectResponse::read_from(payload) {
                        if resp.accepted {
                            self.state = ClientState::Connected;
                            self.assigned_slot = Some(resp.assigned_slot);
                            self.session_id = Some(resp.session_id);
                        }
                    }
                }
                PacketType::EntityDelta => {
                    if let Ok(delta) = EntityDelta::read_from(payload) {
                        deltas.push(delta);
                    }
                }
                _ => {}
            }
        }

        Ok(deltas)
    }

    /// Sends client player entity delta to host.
    pub fn send_entity_delta(&mut self, delta: &EntityDelta) -> std::io::Result<()> {
        if self.state != ClientState::Connected {
            return Ok(());
        }

        let (ack, ack_bits) = self.reliability.generate_ack();
        let seq = self.reliability.next_sequence();
        let header = PacketHeader::new(
            PacketType::EntityDelta,
            0,
            seq,
            ack,
            ack_bits,
        );
        let mut msg = Vec::new();
        header.write_to(&mut msg);
        delta.write_to(&mut msg);
        self.transport.send_to(&msg, self.host_addr)?;
        Ok(())
    }
}
