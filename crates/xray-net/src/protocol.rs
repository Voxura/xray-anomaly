//! Pure-Rust network packet protocol for X-Ray native co-op multiplayer.
//!
//! Provides binary packet framing, sequence numbers, sliding-window ACK generation,
//! and low-overhead zero-copy serialization without external dependencies.

pub const PROTOCOL_MAGIC: [u8; 4] = *b"XRAN"; // X-Ray Anomaly Net
pub const PROTOCOL_VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 20;
pub const MAX_PACKET_SIZE: usize = 1400; // MTU-safe payload size for IPv6/IPv4 UDP

/// Packet control flags.
pub mod flags {
    pub const RELIABLE: u8 = 1 << 0;
    pub const ACK_REQUIRED: u8 = 1 << 1;
    pub const COMPRESSED: u8 = 1 << 2;
}

/// Enumeration of all network packet types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    ConnectRequest = 1,
    ConnectChallenge = 2,
    ConnectResponse = 3,
    Disconnect = 4,
    Heartbeat = 5,
    EntityDelta = 6,
    Event = 7,
    Rpc = 8,
    SaveSync = 9,
    Unknown = 255,
}

impl From<u8> for PacketType {
    fn from(val: u8) -> Self {
        match val {
            1 => PacketType::ConnectRequest,
            2 => PacketType::ConnectChallenge,
            3 => PacketType::ConnectResponse,
            4 => PacketType::Disconnect,
            5 => PacketType::Heartbeat,
            6 => PacketType::EntityDelta,
            7 => PacketType::Event,
            8 => PacketType::Rpc,
            9 => PacketType::SaveSync,
            _ => PacketType::Unknown,
        }
    }
}

/// Fixed 20-byte packet header accompanying every UDP datagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    pub magic: [u8; 4],
    pub protocol_version: u16,
    pub packet_type: PacketType,
    pub flags: u8,
    pub sequence: u32,
    pub ack: u32,
    pub ack_bits: u32,
}

impl PacketHeader {
    pub fn new(packet_type: PacketType, flags: u8, sequence: u32, ack: u32, ack_bits: u32) -> Self {
        Self {
            magic: PROTOCOL_MAGIC,
            protocol_version: PROTOCOL_VERSION,
            packet_type,
            flags,
            sequence,
            ack,
            ack_bits,
        }
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.protocol_version.to_le_bytes());
        out.push(self.packet_type as u8);
        out.push(self.flags);
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out.extend_from_slice(&self.ack.to_le_bytes());
        out.extend_from_slice(&self.ack_bits.to_le_bytes());
    }

    pub fn read_from(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < HEADER_SIZE {
            return Err("packet smaller than header size");
        }
        let magic = [data[0], data[1], data[2], data[3]];
        if magic != PROTOCOL_MAGIC {
            return Err("invalid protocol magic");
        }
        let protocol_version = u16::from_le_bytes([data[4], data[5]]);
        if protocol_version != PROTOCOL_VERSION {
            return Err("unsupported protocol version");
        }
        let packet_type = PacketType::from(data[6]);
        if packet_type == PacketType::Unknown {
            return Err("unknown packet type");
        }
        let flags = data[7];
        let sequence = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let ack = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let ack_bits = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);

        Ok(Self {
            magic,
            protocol_version,
            packet_type,
            flags,
            sequence,
            ack,
            ack_bits,
        })
    }
}

/// Initial connection request payload from a peer client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectRequest {
    pub protocol_version: u16,
    pub lan_id: u64,
    pub nickname: String,
}

impl ConnectRequest {
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.protocol_version.to_le_bytes());
        out.extend_from_slice(&self.lan_id.to_le_bytes());
        let nick_bytes = self.nickname.as_bytes();
        let len = (nick_bytes.len().min(64)) as u8;
        out.push(len);
        out.extend_from_slice(&nick_bytes[..len as usize]);
    }

    pub fn read_from(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 11 {
            return Err("connect request truncated");
        }
        let protocol_version = u16::from_le_bytes([data[0], data[1]]);
        let lan_id = u64::from_le_bytes([
            data[2], data[3], data[4], data[5],
            data[6], data[7], data[8], data[9],
        ]);
        let nick_len = data[10] as usize;
        if data.len() < 11 + nick_len {
            return Err("connect request nickname truncated");
        }
        let nickname = String::from_utf8_lossy(&data[11..11 + nick_len]).to_string();

        Ok(Self {
            protocol_version,
            lan_id,
            nickname,
        })
    }
}

/// Server challenge cookie payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectChallenge {
    pub cookie: u64,
}

impl ConnectChallenge {
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.cookie.to_le_bytes());
    }

    pub fn read_from(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 8 {
            return Err("connect challenge truncated");
        }
        let cookie = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        Ok(Self { cookie })
    }
}

/// Server connection response payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectResponse {
    pub accepted: bool,
    pub assigned_slot: u8,
    pub session_id: u64,
    pub tick_rate: u8,
}

impl ConnectResponse {
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.push(if self.accepted { 1 } else { 0 });
        out.push(self.assigned_slot);
        out.extend_from_slice(&self.session_id.to_le_bytes());
        out.push(self.tick_rate);
    }

    pub fn read_from(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 11 {
            return Err("connect response truncated");
        }
        let accepted = data[0] != 0;
        let assigned_slot = data[1];
        let session_id = u64::from_le_bytes([
            data[2], data[3], data[4], data[5],
            data[6], data[7], data[8], data[9],
        ]);
        let tick_rate = data[10];

        Ok(Self {
            accepted,
            assigned_slot,
            session_id,
            tick_rate,
        })
    }
}

/// High-rate entity transform delta payload for position, rotation, and animation state.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityDelta {
    pub tick: u32,
    pub entity_id: u16,
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub vel: [f32; 3],
    pub anim_state: u16,
    pub weapon_state: u16,
    pub health: f32,
}

impl EntityDelta {
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&self.entity_id.to_le_bytes());
        for &p in &self.pos {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out.extend_from_slice(&self.yaw.to_le_bytes());
        out.extend_from_slice(&self.pitch.to_le_bytes());
        for &v in &self.vel {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.anim_state.to_le_bytes());
        out.extend_from_slice(&self.weapon_state.to_le_bytes());
        out.extend_from_slice(&self.health.to_le_bytes());
    }

    pub fn read_from(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 46 {
            return Err("entity delta truncated");
        }
        let tick = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let entity_id = u16::from_le_bytes([data[4], data[5]]);
        let mut offset = 6;

        let read_f32 = |offset: &mut usize| -> f32 {
            let val = f32::from_le_bytes([data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]]);
            *offset += 4;
            val
        };

        let pos = [read_f32(&mut offset), read_f32(&mut offset), read_f32(&mut offset)];
        let yaw = read_f32(&mut offset);
        let pitch = read_f32(&mut offset);
        let vel = [read_f32(&mut offset), read_f32(&mut offset), read_f32(&mut offset)];

        let anim_state = u16::from_le_bytes([data[offset], data[offset + 1]]);
        offset += 2;
        let weapon_state = u16::from_le_bytes([data[offset], data[offset + 1]]);
        offset += 2;
        let health = read_f32(&mut offset);

        Ok(Self {
            tick,
            entity_id,
            pos,
            yaw,
            pitch,
            vel,
            anim_state,
            weapon_state,
            health,
        })
    }
}

/// Heartbeat ping/pong payload for latency and clock synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heartbeat {
    pub client_time_ms: u32,
    pub server_time_ms: u32,
}

impl Heartbeat {
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.client_time_ms.to_le_bytes());
        out.extend_from_slice(&self.server_time_ms.to_le_bytes());
    }

    pub fn read_from(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 8 {
            return Err("heartbeat truncated");
        }
        let client_time_ms = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let server_time_ms = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        Ok(Self {
            client_time_ms,
            server_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = PacketHeader::new(PacketType::EntityDelta, flags::RELIABLE | flags::ACK_REQUIRED, 1024, 1020, 0b1111);
        let mut buf = Vec::new();
        header.write_to(&mut buf);
        assert_eq!(buf.len(), HEADER_SIZE);

        let decoded = PacketHeader::read_from(&buf).expect("failed to decode header");
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_connect_request_roundtrip() {
        let req = ConnectRequest {
            protocol_version: PROTOCOL_VERSION,
            lan_id: 212375856848651,
            nickname: "falcon".to_string(),
        };
        let mut buf = Vec::new();
        req.write_to(&mut buf);

        let decoded = ConnectRequest::read_from(&buf).expect("failed to decode connect request");
        assert_eq!(decoded, req);
    }

    #[test]
    fn test_connect_response_roundtrip() {
        let resp = ConnectResponse {
            accepted: true,
            assigned_slot: 1,
            session_id: 0xfeed_dead_beef_cafe,
            tick_rate: 60,
        };
        let mut buf = Vec::new();
        resp.write_to(&mut buf);

        let decoded = ConnectResponse::read_from(&buf).expect("failed to decode connect response");
        assert_eq!(decoded, resp);
    }

    #[test]
    fn test_entity_delta_roundtrip() {
        let delta = EntityDelta {
            tick: 5042,
            entity_id: 1234,
            pos: [100.5, 2.0, -45.25],
            yaw: 1.57,
            pitch: -0.25,
            vel: [0.0, 0.0, 5.2],
            anim_state: 42,
            weapon_state: 7,
            health: 0.95,
        };
        let mut buf = Vec::new();
        delta.write_to(&mut buf);
        assert_eq!(buf.len(), 46);

        let decoded = EntityDelta::read_from(&buf).expect("failed to decode entity delta");
        assert_eq!(decoded, delta);
    }

    #[test]
    fn test_heartbeat_roundtrip() {
        let hb = Heartbeat {
            client_time_ms: 123456,
            server_time_ms: 123500,
        };
        let mut buf = Vec::new();
        hb.write_to(&mut buf);

        let decoded = Heartbeat::read_from(&buf).expect("failed to decode heartbeat");
        assert_eq!(decoded, hb);
    }
}
