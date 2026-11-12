//! Tier 2: RFC 5389 STUN Client and UDP Hole Punching.
//!
//! Discovers reflexive public IPv4/IPv6 addresses through NAT/CGNAT gateways
//! and performs simultaneous UDP hole punching for direct P2P connections.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

pub const STUN_MAGIC_COOKIE: u32 = 0x2112A442;
pub const BINDING_REQUEST: u16 = 0x0001;
pub const BINDING_RESPONSE: u16 = 0x0101;

pub const ATTR_MAPPED_ADDRESS: u16 = 0x0001;
pub const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;

/// Default public STUN servers for NAT reflexive address discovery.
pub const DEFAULT_STUN_SERVERS: &[&str] = &[
    "stun.l.google.com:19302",
    "stun1.l.google.com:19302",
    "stun.cloudflare.com:3478",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatType {
    /// Direct public IP or Open Internet
    OpenInternet,
    /// Full cone NAT: Any external host can send packets to the mapped port
    FullCone,
    /// Restricted cone / Port restricted: External host must be target of outbound packet
    RestrictedCone,
    /// Symmetric NAT: Port mapping changes for each destination host
    Symmetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HolePunchState {
    Idle,
    DiscoveringNat,
    Punching {
        target: SocketAddr,
        attempts: u32,
    },
    Success(SocketAddr),
    Failed,
}

/// Builds a 20-byte RFC 5389 STUN Binding Request packet.
pub fn build_binding_request(transaction_id: &[u8; 12]) -> [u8; 20] {
    let mut packet = [0u8; 20];
    packet[0..2].copy_from_slice(&BINDING_REQUEST.to_be_bytes());
    packet[2..4].copy_from_slice(&0u16.to_be_bytes()); // Message Length: 0
    packet[4..8].copy_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
    packet[8..20].copy_from_slice(transaction_id);
    packet
}

/// Parses an RFC 5389 STUN Binding Success Response and extracts the reflexive `SocketAddr`.
pub fn parse_binding_response(data: &[u8], expected_tx_id: &[u8; 12]) -> Result<SocketAddr, &'static str> {
    if data.len() < 20 {
        return Err("STUN response too short");
    }

    let msg_type = u16::from_be_bytes([data[0], data[1]]);
    if msg_type != BINDING_RESPONSE {
        return Err("not a STUN binding response");
    }

    let msg_len = u16::from_be_bytes([data[2], data[3]]) as usize;
    let cookie = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    if cookie != STUN_MAGIC_COOKIE {
        return Err("invalid STUN magic cookie");
    }

    if &data[8..20] != expected_tx_id {
        return Err("STUN transaction ID mismatch");
    }

    if data.len() < 20 + msg_len {
        return Err("STUN response truncated");
    }

    let mut offset = 20;
    let end = 20 + msg_len;

    while offset + 4 <= end {
        let attr_type = u16::from_be_bytes([data[offset], data[offset + 1]]);
        let attr_len = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
        offset += 4;

        if offset + attr_len > end {
            break;
        }

        let attr_data = &data[offset..offset + attr_len];

        if attr_type == ATTR_XOR_MAPPED_ADDRESS && attr_len >= 8 {
            let family = attr_data[1];
            let xor_port = u16::from_be_bytes([attr_data[2], attr_data[3]]);
            let port = xor_port ^ ((STUN_MAGIC_COOKIE >> 16) as u16);

            if family == 1 && attr_len >= 8 {
                // IPv4
                let xor_ip = u32::from_be_bytes([attr_data[4], attr_data[5], attr_data[6], attr_data[7]]);
                let ip = Ipv4Addr::from(xor_ip ^ STUN_MAGIC_COOKIE);
                return Ok(SocketAddr::new(IpAddr::V4(ip), port));
            } else if family == 2 && attr_len >= 20 {
                // IPv6
                let mut ip_bytes = [0u8; 16];
                let cookie_bytes = STUN_MAGIC_COOKIE.to_be_bytes();
                for i in 0..4 {
                    ip_bytes[i] = attr_data[4 + i] ^ cookie_bytes[i];
                }
                for i in 0..12 {
                    ip_bytes[4 + i] = attr_data[8 + i] ^ expected_tx_id[i];
                }
                let ip = Ipv6Addr::from(ip_bytes);
                return Ok(SocketAddr::new(IpAddr::V6(ip), port));
            }
        } else if attr_type == ATTR_MAPPED_ADDRESS && attr_len >= 8 {
            let family = attr_data[1];
            let port = u16::from_be_bytes([attr_data[2], attr_data[3]]);
            if family == 1 && attr_len >= 8 {
                let ip = Ipv4Addr::new(attr_data[4], attr_data[5], attr_data[6], attr_data[7]);
                return Ok(SocketAddr::new(IpAddr::V4(ip), port));
            }
        }

        // STUN attributes are padded to 4-byte boundaries
        offset += (attr_len + 3) & !3;
    }

    Err("no mapped address attribute found in STUN response")
}

/// UDP Hole Punching coordinator managing synchronous probe bursts.
pub struct HolePunchCoordinator {
    state: HolePunchState,
    burst_interval_ms: u64,
    max_burst_attempts: u32,
    tx_id: [u8; 12],
}

impl Default for HolePunchCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl HolePunchCoordinator {
    pub fn new() -> Self {
        Self {
            state: HolePunchState::Idle,
            burst_interval_ms: 100,
            max_burst_attempts: 30, // 3 seconds total punch window
            tx_id: [0x58, 0x52, 0x41, 0x59, 0x48, 0x4f, 0x4c, 0x45, 0x50, 0x55, 0x4e, 0x43], // 'XRAYHOLEPUNC'
        }
    }

    pub fn state(&self) -> HolePunchState {
        self.state
    }

    pub fn transaction_id(&self) -> &[u8; 12] {
        &self.tx_id
    }

    pub fn burst_interval_ms(&self) -> u64 {
        self.burst_interval_ms
    }

    /// Prepares a STUN query packet to discover local reflexive endpoint.
    pub fn prepare_stun_request(&mut self) -> [u8; 20] {
        self.state = HolePunchState::DiscoveringNat;
        build_binding_request(&self.tx_id)
    }

    /// Handles a STUN response packet.
    pub fn on_stun_response(&mut self, data: &[u8]) -> Result<SocketAddr, &'static str> {
        let addr = parse_binding_response(data, &self.tx_id)?;
        Ok(addr)
    }

    /// Initiates simultaneous UDP hole punching probes to the peer's discovered public endpoint.
    pub fn start_punching(&mut self, target: SocketAddr) {
        self.state = HolePunchState::Punching { target, attempts: 0 };
    }

    /// Evaluates whether a punch probe packet should be transmitted.
    pub fn next_punch_probe(&mut self) -> Option<SocketAddr> {
        match &mut self.state {
            HolePunchState::Punching { target, attempts } => {
                if *attempts >= self.max_burst_attempts {
                    let _ = *target;
                    self.state = HolePunchState::Failed;
                    None
                } else {
                    *attempts += 1;
                    Some(*target)
                }
            }
            _ => None,
        }
    }

    /// Called when a probe or heartbeat is successfully received from the peer.
    pub fn mark_punched(&mut self, endpoint: SocketAddr) {
        self.state = HolePunchState::Success(endpoint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stun_binding_request_format() {
        let tx_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let req = build_binding_request(&tx_id);

        assert_eq!(&req[0..2], &BINDING_REQUEST.to_be_bytes());
        assert_eq!(&req[2..4], &[0, 0]);
        assert_eq!(&req[4..8], &STUN_MAGIC_COOKIE.to_be_bytes());
        assert_eq!(&req[8..20], &tx_id);
    }

    #[test]
    fn test_stun_xor_mapped_ipv4_response() {
        let tx_id = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
        let mut resp = Vec::new();

        // STUN header
        resp.extend_from_slice(&BINDING_RESPONSE.to_be_bytes());
        resp.extend_from_slice(&12u16.to_be_bytes()); // Attribute length: 12 bytes
        resp.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
        resp.extend_from_slice(&tx_id);

        // XOR-MAPPED-ADDRESS attribute: type=0x0020, len=8
        resp.extend_from_slice(&ATTR_XOR_MAPPED_ADDRESS.to_be_bytes());
        resp.extend_from_slice(&8u16.to_be_bytes());

        // Reserved=0, Family=1 (IPv4)
        resp.push(0);
        resp.push(1);

        // Port 54321 XORed with 0x2112
        let port = 54321u16;
        let xor_port = port ^ ((STUN_MAGIC_COOKIE >> 16) as u16);
        resp.extend_from_slice(&xor_port.to_be_bytes());

        // IP 203.0.113.195 XORed with 0x2112A442
        let ip_u32 = u32::from_be_bytes([203, 0, 113, 195]);
        let xor_ip = ip_u32 ^ STUN_MAGIC_COOKIE;
        resp.extend_from_slice(&xor_ip.to_be_bytes());

        let addr = parse_binding_response(&resp, &tx_id).expect("failed to parse STUN response");
        assert_eq!(addr, "203.0.113.195:54321".parse::<SocketAddr>().unwrap());
    }

    #[test]
    fn test_hole_punch_coordinator_lifecycle() {
        let mut coord = HolePunchCoordinator::new();
        assert_eq!(coord.state(), HolePunchState::Idle);

        let target: SocketAddr = "198.51.100.42:6112".parse().unwrap();
        coord.start_punching(target);

        // Probing
        for i in 1..=5 {
            let probe_target = coord.next_punch_probe();
            assert_eq!(probe_target, Some(target));
            if let HolePunchState::Punching { attempts, .. } = coord.state() {
                assert_eq!(attempts, i);
            }
        }

        // Punch success
        coord.mark_punched(target);
        assert_eq!(coord.state(), HolePunchState::Success(target));
    }
}
