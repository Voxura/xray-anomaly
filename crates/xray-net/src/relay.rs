//! Tier 3: In-Engine Fallback Relay (No Radmin Required).
//!
//! Provides transparent packet relaying for worst-case strict Symmetric NATs
//! or restrictive mobile hotspots without IPv6. Operates without installing
//! any virtual network adapters, drivers, or requiring administrator privileges.

use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::net::{SocketAddr, UdpSocket};

pub const RELAY_MAGIC: [u8; 4] = *b"XRRL"; // X-Ray Relay
pub const RELAY_HEADER_SIZE: usize = 16;
pub const MAX_RELAY_PAYLOAD: usize = 1400;

/// 16-byte fixed relay header encapsulating game datagrams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelayHeader {
    pub magic: [u8; 4],
    pub session_id: u64,
    pub source_slot: u8,
    pub target_slot: u8,
    pub payload_len: u16,
}

impl RelayHeader {
    pub fn new(session_id: u64, source_slot: u8, target_slot: u8, payload_len: u16) -> Self {
        Self {
            magic: RELAY_MAGIC,
            session_id,
            source_slot,
            target_slot,
            payload_len,
        }
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.session_id.to_le_bytes());
        out.push(self.source_slot);
        out.push(self.target_slot);
        out.extend_from_slice(&self.payload_len.to_le_bytes());
    }

    pub fn read_from(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < RELAY_HEADER_SIZE {
            return Err("relay packet smaller than header");
        }
        let magic = [data[0], data[1], data[2], data[3]];
        if magic != RELAY_MAGIC {
            return Err("invalid relay magic");
        }
        let session_id = u64::from_le_bytes([
            data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11],
        ]);
        let source_slot = data[12];
        let target_slot = data[13];
        let payload_len = u16::from_le_bytes([data[14], data[15]]);

        Ok(Self {
            magic,
            session_id,
            source_slot,
            target_slot,
            payload_len,
        })
    }
}

/// Client component that routes packets through the fallback relay.
pub struct RelayClient {
    socket: UdpSocket,
    relay_addr: SocketAddr,
    session_id: u64,
    local_slot: u8,
    recv_buf: [u8; RELAY_HEADER_SIZE + MAX_RELAY_PAYLOAD],
}

impl RelayClient {
    pub fn connect(relay_addr: SocketAddr, session_id: u64, local_slot: u8) -> io::Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.set_nonblocking(true)?;

        Ok(Self {
            socket,
            relay_addr,
            session_id,
            local_slot,
            recv_buf: [0u8; RELAY_HEADER_SIZE + MAX_RELAY_PAYLOAD],
        })
    }

    /// Envelopes and transmits a game payload to the target slot via relay.
    pub fn send_relayed(&self, target_slot: u8, payload: &[u8]) -> io::Result<usize> {
        let header = RelayHeader::new(
            self.session_id,
            self.local_slot,
            target_slot,
            payload.len() as u16,
        );
        let mut msg = Vec::with_capacity(RELAY_HEADER_SIZE + payload.len());
        header.write_to(&mut msg);
        msg.extend_from_slice(payload);
        self.socket.send_to(&msg, self.relay_addr)
    }

    /// Polls for incoming game payloads forwarded by the relay.
    ///
    /// Returns `Ok(Some((source_slot, payload)))` or `Ok(None)` if no packet available.
    pub fn poll_relayed(&mut self) -> io::Result<Option<(u8, &[u8])>> {
        match self.socket.recv_from(&mut self.recv_buf) {
            Ok((len, addr)) => {
                if addr != self.relay_addr || len < RELAY_HEADER_SIZE {
                    return Ok(None);
                }
                if let Ok(header) = RelayHeader::read_from(&self.recv_buf[..len]) {
                    if header.session_id != self.session_id || header.target_slot != self.local_slot {
                        return Ok(None);
                    }
                    let end = (RELAY_HEADER_SIZE + header.payload_len as usize).min(len);
                    return Ok(Some((header.source_slot, &self.recv_buf[RELAY_HEADER_SIZE..end])));
                }
                Ok(None)
            }
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }
}

/// Lightweight, in-engine packet relay server capable of routing co-op traffic.
pub struct RelayServer {
    socket: UdpSocket,
    // session_id -> (slot -> SocketAddr)
    sessions: HashMap<u64, HashMap<u8, SocketAddr>>,
    buf: [u8; RELAY_HEADER_SIZE + MAX_RELAY_PAYLOAD],
}

impl RelayServer {
    pub fn bind(bind_addr: &str) -> io::Result<Self> {
        let socket = UdpSocket::bind(bind_addr)?;
        socket.set_nonblocking(true)?;

        Ok(Self {
            socket,
            sessions: HashMap::new(),
            buf: [0u8; RELAY_HEADER_SIZE + MAX_RELAY_PAYLOAD],
        })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Processes incoming relay packets and forwards them to target peers.
    /// Returns the number of packets forwarded.
    pub fn forward_pending(&mut self) -> io::Result<usize> {
        let mut forwarded = 0;

        loop {
            match self.socket.recv_from(&mut self.buf) {
                Ok((len, sender_addr)) => {
                    if len < RELAY_HEADER_SIZE {
                        continue;
                    }
                    if let Ok(header) = RelayHeader::read_from(&self.buf[..len]) {
                        // Register or update sender's address for this session slot
                        let session_slots = self.sessions.entry(header.session_id).or_default();
                        session_slots.insert(header.source_slot, sender_addr);

                        // If target slot is known, forward packet
                        if let Some(&target_addr) = session_slots.get(&header.target_slot) {
                            let _ = self.socket.send_to(&self.buf[..len], target_addr);
                            forwarded += 1;
                        }
                    }
                }
                Err(ref e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) => return Err(e),
            }
        }

        Ok(forwarded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relay_header_roundtrip() {
        let header = RelayHeader::new(0x1122_3344_5566_7788, 1, 2, 42);
        let mut buf = Vec::new();
        header.write_to(&mut buf);
        assert_eq!(buf.len(), RELAY_HEADER_SIZE);

        let decoded = RelayHeader::read_from(&buf).expect("failed to decode relay header");
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_in_engine_relay_loopback() {
        // Start relay server
        let mut relay = RelayServer::bind("127.0.0.1:0").expect("failed to bind relay");
        let relay_addr = relay.local_addr().unwrap();

        let session_id = 0x9999_8888_7777_6666;

        // Player 1 (Host, slot 0)
        let mut client1 = RelayClient::connect(relay_addr, session_id, 0).expect("failed to create client1");
        // Player 2 (Falcon, slot 1)
        let client2 = RelayClient::connect(relay_addr, session_id, 1).expect("failed to create client2");

        // Client 1 sends ping to register slot 0 with relay
        client1.send_relayed(1, b"reg0").expect("failed send client1");
        relay.forward_pending().expect("forward failed");

        // Client 2 sends payload to slot 0
        client2.send_relayed(0, b"stalker_coop_payload").expect("failed send client2");
        let forwarded = relay.forward_pending().expect("forward failed");
        assert!(forwarded >= 1, "relay should forward payload");

        // Client 1 polls for forwarded payload
        std::thread::sleep(std::time::Duration::from_millis(10));
        let polled = client1.poll_relayed().expect("poll failed");
        assert!(polled.is_some(), "client1 should receive forwarded payload");
        let (src_slot, data) = polled.unwrap();
        assert_eq!(src_slot, 1);
        assert_eq!(data, b"stalker_coop_payload");
    }
}
