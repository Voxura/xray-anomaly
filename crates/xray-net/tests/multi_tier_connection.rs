//! Multi-Tier connection architecture integration tests.
//!
//! Verifies:
//! - Tier 1: Direct IPv6 P2P communication.
//! - Tier 2: RFC 5389 STUN NAT traversal and simultaneous hole punching.
//! - Tier 3: In-Engine Fallback Relay transparent packet routing.

use std::net::SocketAddr;
use std::thread;
use std::time::Duration;
use xray_net::{
    ClientSession, ClientState, EntityDelta, HolePunchCoordinator, HolePunchState,
    HostSession, RelayClient, RelayServer, STUN_MAGIC_COOKIE,
};

#[test]
fn test_tier1_direct_ipv6() {
    let mut host = HostSession::new(0, true).expect("failed to bind host IPv6");
    let host_addr = host.local_addr().expect("failed local addr");

    let target: SocketAddr = format!("[::1]:{}", host_addr.port()).parse().unwrap();
    let mut client = ClientSession::connect(target, 212375856848651, "falcon", true)
        .expect("failed client connect");

    let mut connected = false;
    for _ in 0..25 {
        let _ = host.update();
        let _ = client.update();
        if client.state() == ClientState::Connected && host.has_peer() {
            connected = true;
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert!(connected, "Tier 1 IPv6 direct handshake must succeed");
    assert_eq!(client.assigned_slot(), Some(1));
}

#[test]
fn test_tier2_stun_and_hole_punch_coordination() {
    // 1. Verify STUN discovery mechanics
    let mut coordinator = HolePunchCoordinator::new();
    let _stun_req = coordinator.prepare_stun_request();
    assert_eq!(coordinator.state(), HolePunchState::DiscoveringNat);

    // Mock STUN server response simulating CGNAT reflexive endpoint mapping
    let mut mock_stun_resp = Vec::new();
    mock_stun_resp.extend_from_slice(&0x0101u16.to_be_bytes()); // Binding Success
    mock_stun_resp.extend_from_slice(&12u16.to_be_bytes());     // Length
    mock_stun_resp.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
    mock_stun_resp.extend_from_slice(coordinator.transaction_id());

    // XOR-MAPPED-ADDRESS: 198.51.100.77:49152
    mock_stun_resp.extend_from_slice(&0x0020u16.to_be_bytes());
    mock_stun_resp.extend_from_slice(&8u16.to_be_bytes());
    mock_stun_resp.push(0);
    mock_stun_resp.push(1); // IPv4
    let xor_port = 49152u16 ^ ((STUN_MAGIC_COOKIE >> 16) as u16);
    mock_stun_resp.extend_from_slice(&xor_port.to_be_bytes());
    let xor_ip = u32::from_be_bytes([198, 51, 100, 77]) ^ STUN_MAGIC_COOKIE;
    mock_stun_resp.extend_from_slice(&xor_ip.to_be_bytes());

    let discovered_addr = coordinator.on_stun_response(&mock_stun_resp)
        .expect("STUN response should decode reflexive endpoint");
    assert_eq!(discovered_addr, "198.51.100.77:49152".parse::<SocketAddr>().unwrap());

    // 2. Begin simultaneous hole punching to peer reflexive address
    let peer_reflexive: SocketAddr = "203.0.113.88:51234".parse().unwrap();
    coordinator.start_punching(peer_reflexive);

    let probe1 = coordinator.next_punch_probe();
    assert_eq!(probe1, Some(peer_reflexive));

    // Simulated arrival of peer return packet punching the hole
    coordinator.mark_punched(peer_reflexive);
    assert_eq!(coordinator.state(), HolePunchState::Success(peer_reflexive));
}

#[test]
fn test_tier3_in_engine_fallback_relay() {
    // Start in-engine relay server on loopback
    let mut relay_server = RelayServer::bind("127.0.0.1:0")
        .expect("failed to start fallback relay server");
    let relay_addr = relay_server.local_addr().unwrap();

    let session_id = 0x5441_4c4b_4552_0001; // S.T.A.L.K.E.R. co-op session

    // Host binds relay client (slot 0)
    let mut host_relay = RelayClient::connect(relay_addr, session_id, 0)
        .expect("failed to connect host relay client");
    // Falcon binds relay client (slot 1)
    let client_relay = RelayClient::connect(relay_addr, session_id, 1)
        .expect("failed to connect client relay client");

    // Host registers with relay
    host_relay.send_relayed(1, b"host_hello").unwrap();
    relay_server.forward_pending().unwrap();

    // Client sends player transform delta through fallback relay
    let delta = EntityDelta {
        tick: 400,
        entity_id: 1,
        pos: [50.0, 1.5, -20.0],
        yaw: 0.785,
        pitch: 0.0,
        vel: [3.0, 0.0, 0.0],
        anim_state: 4,
        weapon_state: 1,
        health: 0.99,
    };
    let mut delta_bytes = Vec::new();
    delta.write_to(&mut delta_bytes);

    client_relay.send_relayed(0, &delta_bytes).expect("client should send relayed packet");
    let forwarded = relay_server.forward_pending().expect("relay should forward packet");
    assert!(forwarded >= 1, "relay must forward client packet to host");

    // Host reads relayed payload
    thread::sleep(Duration::from_millis(10));
    let relayed = host_relay.poll_relayed().expect("host poll failed");
    assert!(relayed.is_some(), "host must receive relayed packet");

    let (src_slot, received_payload) = relayed.unwrap();
    assert_eq!(src_slot, 1, "payload must originate from slot 1");

    let received_delta = EntityDelta::read_from(received_payload)
        .expect("relayed payload should decode into EntityDelta");
    assert_eq!(received_delta, delta);
}
