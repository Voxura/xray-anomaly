//! Hardware integration tests for IPv6 P2P co-op networking.

use std::net::SocketAddr;
use std::thread;
use std::time::Duration;
use xray_net::{ClientSession, ClientState, EntityDelta, HostSession};

#[test]
fn test_ipv6_direct_p2p_handshake() {
    // Bind host on IPv6 ephemeral port
    let mut host = HostSession::new(0, true).expect("failed to bind host IPv6");
    let host_addr = host.local_addr().expect("failed to get local addr");

    // Client connects using loopback IPv6 [::1]:port
    let connect_target: SocketAddr = format!("[::1]:{}", host_addr.port()).parse().unwrap();
    let mut client = ClientSession::connect(
        connect_target,
        212375856848651, // Falcon's persistent LAN ID
        "falcon",
        true,
    ).expect("failed to create client session");

    // Advance network loop for handshake convergence
    let mut connected = false;
    for _ in 0..30 {
        let _ = host.update();
        let _ = client.update();

        if client.state() == ClientState::Connected && host.has_peer() {
            connected = true;
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert!(connected, "P2P handshake failed to complete over IPv6");
    assert_eq!(client.state(), ClientState::Connected);
    assert_eq!(client.assigned_slot(), Some(1));
    assert_eq!(host.peer_nickname(), Some("falcon"));
    assert_eq!(host.peer_lan_id(), Some(212375856848651));
}

#[test]
fn test_entity_delta_bidirectional_sync() {
    let mut host = HostSession::new(0, true).expect("failed to bind host IPv6");
    let host_addr = host.local_addr().expect("failed to get local addr");

    let connect_target: SocketAddr = format!("[::1]:{}", host_addr.port()).parse().unwrap();
    let mut client = ClientSession::connect(
        connect_target,
        212375856848651,
        "falcon",
        true,
    ).expect("failed to create client session");

    // Perform handshake
    for _ in 0..30 {
        let _ = host.update();
        let _ = client.update();
        if client.state() == ClientState::Connected {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(client.state(), ClientState::Connected);

    // 1. Client sends entity delta to host
    let client_delta = EntityDelta {
        tick: 100,
        entity_id: 1, // Player 2 (Falcon)
        pos: [12.5, 0.0, -85.0],
        yaw: std::f32::consts::PI,
        pitch: 0.1,
        vel: [0.0, 0.0, 4.5],
        anim_state: 12,
        weapon_state: 3,
        health: 1.0,
    };
    client.send_entity_delta(&client_delta).expect("failed to send client delta");

    thread::sleep(Duration::from_millis(15));
    let host_received = host.update().expect("host update failed");
    assert_eq!(host_received.len(), 1, "host should receive exactly 1 entity delta");
    assert_eq!(host_received[0], client_delta);

    // 2. Host sends entity delta to client
    let host_delta = EntityDelta {
        tick: 101,
        entity_id: 0, // Player 1 (Zilla)
        pos: [15.0, 0.0, -80.0],
        yaw: 0.0,
        pitch: -0.2,
        vel: [1.2, 0.0, 0.0],
        anim_state: 8,
        weapon_state: 1,
        health: 0.85,
    };
    host.send_entity_delta(&host_delta).expect("failed to send host delta");

    thread::sleep(Duration::from_millis(15));
    let client_received = client.update().expect("client update failed");
    assert_eq!(client_received.len(), 1, "client should receive exactly 1 entity delta");
    assert_eq!(client_received[0], host_delta);
}
