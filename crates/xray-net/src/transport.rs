//! Non-blocking UDP transport layer supporting dual-stack IPv6 and IPv4.

use std::io::{self, ErrorKind};
use std::net::{SocketAddr, UdpSocket};
use crate::protocol::{PacketHeader, HEADER_SIZE, MAX_PACKET_SIZE};

/// Configuration for the UDP transport socket.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub port: u16,
    pub use_ipv6: bool,
    pub non_blocking: bool,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            port: 0,
            use_ipv6: true,
            non_blocking: true,
        }
    }
}

/// A non-blocking UDP socket wrapper tailored for peer-to-peer co-op communication.
pub struct UdpTransport {
    socket: UdpSocket,
    recv_buf: [u8; MAX_PACKET_SIZE],
}

impl UdpTransport {
    /// Binds a UDP socket with the specified configuration.
    pub fn bind(config: TransportConfig) -> io::Result<Self> {
        let bind_addr = if config.use_ipv6 {
            format!("[::]:{}", config.port)
        } else {
            format!("0.0.0.0:{}", config.port)
        };

        let socket = UdpSocket::bind(&bind_addr)?;
        if config.non_blocking {
            socket.set_nonblocking(true)?;
        }

        Ok(Self {
            socket,
            recv_buf: [0u8; MAX_PACKET_SIZE],
        })
    }

    /// Sends a raw byte payload to the target peer address.
    pub fn send_to(&self, data: &[u8], target: SocketAddr) -> io::Result<usize> {
        self.socket.send_to(data, target)
    }

    /// Receives an incoming datagram without blocking.
    ///
    /// Returns `Ok(Some((bytes_read, sender_addr)))` on packet receipt,
    /// or `Ok(None)` if no packet was available in non-blocking mode.
    pub fn poll_recv(&mut self) -> io::Result<Option<(&[u8], SocketAddr)>> {
        match self.socket.recv_from(&mut self.recv_buf) {
            Ok((len, addr)) => Ok(Some((&self.recv_buf[..len], addr))),
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Helper to poll and decode the header and payload directly.
    pub fn poll_packet(&mut self) -> io::Result<Option<(PacketHeader, &[u8], SocketAddr)>> {
        if let Some((raw, addr)) = self.poll_recv()? {
            if raw.len() < HEADER_SIZE {
                return Ok(None); // Drop runt packet
            }
            if let Ok(header) = PacketHeader::read_from(raw) {
                let payload = &raw[HEADER_SIZE..];
                return Ok(Some((header, payload, addr)));
            }
        }
        Ok(None)
    }

    /// Gets the local socket address.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_udp_transport_loopback_ipv6() {
        // Bind host on ephemeral port
        let mut host = UdpTransport::bind(TransportConfig {
            port: 0,
            use_ipv6: true,
            non_blocking: true,
        }).expect("failed to bind host IPv6 socket");

        let host_addr = host.local_addr().unwrap();

        // Bind client on ephemeral port
        let client = UdpTransport::bind(TransportConfig {
            port: 0,
            use_ipv6: true,
            non_blocking: true,
        }).expect("failed to bind client IPv6 socket");

        // Construct packet
        let header = PacketHeader::new(
            crate::protocol::PacketType::Heartbeat,
            0,
            1,
            0,
            0,
        );
        let mut msg = Vec::new();
        header.write_to(&mut msg);
        msg.extend_from_slice(&[42, 42, 42, 42]);

        // Send to host
        let target: SocketAddr = format!("[::1]:{}", host_addr.port()).parse().unwrap();
        client.send_to(&msg, target).expect("failed to send packet");

        // Wait a small amount for OS loopback delivery
        std::thread::sleep(std::time::Duration::from_millis(10));

        let res = host.poll_packet().expect("poll failed");
        assert!(res.is_some(), "expected packet at host");
        let (recv_header, recv_payload, _) = res.unwrap();
        assert_eq!(recv_header, header);
        assert_eq!(recv_payload, &[42, 42, 42, 42]);
    }
}
