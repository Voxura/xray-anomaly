//! Reliable UDP sliding-window sequence tracking, ACK generation, and RTT estimation.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Window size for sliding ACK bitmask (32 packets past highest ACK).
pub const ACK_WINDOW_SIZE: u32 = 32;

/// Default retransmission timeout if no RTT measurement is available.
pub const DEFAULT_RTO_MS: u64 = 200;

/// Maximum number of retransmission attempts before declaring a disconnect.
pub const MAX_RESENDS: u32 = 10;

/// Pending unacknowledged reliable packet.
#[derive(Debug, Clone)]
pub struct PendingPacket {
    pub sequence: u32,
    pub payload: Vec<u8>,
    pub sent_at: Instant,
    pub resend_count: u32,
}

/// Tracks sequence numbers, ACK generation, and reliable packet retransmissions.
pub struct ReliabilityTracker {
    next_sequence: u32,
    remote_ack: u32,
    remote_ack_bits: u32,
    received_sequences: VecDeque<u32>,
    pending_reliable: VecDeque<PendingPacket>,
    srtt_ms: f32,
    rtt_variance_ms: f32,
}

impl Default for ReliabilityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ReliabilityTracker {
    pub fn new() -> Self {
        Self {
            next_sequence: 0,
            remote_ack: 0,
            remote_ack_bits: 0,
            received_sequences: VecDeque::with_capacity(64),
            pending_reliable: VecDeque::with_capacity(32),
            srtt_ms: 50.0,
            rtt_variance_ms: 10.0,
        }
    }

    /// Allocates the next outgoing sequence number.
    pub fn next_sequence(&mut self) -> u32 {
        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        seq
    }

    /// Registers a packet queued as reliable so it can be tracked for ACKs and resends.
    pub fn queue_reliable(&mut self, sequence: u32, payload: Vec<u8>) {
        self.pending_reliable.push_back(PendingPacket {
            sequence,
            payload,
            sent_at: Instant::now(),
            resend_count: 0,
        });
    }

    /// Records an incoming packet's sequence number and updates local ACK information.
    pub fn on_packet_received(&mut self, sequence: u32) {
        if self.received_sequences.is_empty() {
            self.received_sequences.push_back(sequence);
            return;
        }

        let highest = *self.received_sequences.front().unwrap();

        // Sequence numbers use modular wrapping arithmetic.
        let diff = sequence.wrapping_sub(highest) as i32;
        if diff > 0 {
            // New highest sequence number received.
            self.received_sequences.push_front(sequence);
            if self.received_sequences.len() > 64 {
                self.received_sequences.pop_back();
            }
        } else if diff < 0 && diff > -64 {
            // Older sequence within window; insert if not already present.
            if !self.received_sequences.contains(&sequence) {
                self.received_sequences.push_back(sequence);
            }
        }
    }

    /// Calculates the current (ack, ack_bits) to write into an outgoing packet header.
    pub fn generate_ack(&self) -> (u32, u32) {
        if self.received_sequences.is_empty() {
            return (0, 0);
        }

        let highest = *self.received_sequences.front().unwrap();
        let mut ack_bits = 0u32;

        for i in 1..=ACK_WINDOW_SIZE {
            let target_seq = highest.wrapping_sub(i);
            if self.received_sequences.contains(&target_seq) {
                ack_bits |= 1 << (i - 1);
            }
        }

        (highest, ack_bits)
    }

    /// Processes an incoming packet's ACK and ack_bits, retiring acknowledged packets
    /// and updating round-trip time measurements.
    pub fn process_incoming_ack(&mut self, ack: u32, ack_bits: u32) {
        self.remote_ack = ack;
        self.remote_ack_bits = ack_bits;

        let now = Instant::now();
        let mut i = 0;
        while i < self.pending_reliable.len() {
            let packet = &self.pending_reliable[i];
            let seq = packet.sequence;

            let is_acked = if seq == ack {
                true
            } else {
                let diff = ack.wrapping_sub(seq) as i32;
                if diff > 0 && diff <= (ACK_WINDOW_SIZE as i32) {
                    (ack_bits & (1 << (diff - 1))) != 0
                } else {
                    false
                }
            };

            if is_acked {
                // Update smoothed RTT
                let sample_rtt = now.duration_since(packet.sent_at).as_secs_f32() * 1000.0;
                let delta = (sample_rtt - self.srtt_ms).abs();
                self.srtt_ms = 0.875 * self.srtt_ms + 0.125 * sample_rtt;
                self.rtt_variance_ms = 0.75 * self.rtt_variance_ms + 0.25 * delta;

                self.pending_reliable.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Collects reliable packets that have exceeded the retransmission timeout (RTO).
    pub fn collect_resends(&mut self) -> Vec<PendingPacket> {
        let rto = Duration::from_millis(
            ((self.srtt_ms + 4.0 * self.rtt_variance_ms).max(DEFAULT_RTO_MS as f32) as u64).min(1000),
        );

        let now = Instant::now();
        let mut resends = Vec::new();

        for packet in self.pending_reliable.iter_mut() {
            if now.duration_since(packet.sent_at) >= rto {
                packet.resend_count += 1;
                packet.sent_at = now;
                resends.push(packet.clone());
            }
        }

        resends
    }

    pub fn smoothed_rtt_ms(&self) -> f32 {
        self.srtt_ms
    }

    pub fn pending_reliable_count(&self) -> usize {
        self.pending_reliable.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_increment() {
        let mut tracker = ReliabilityTracker::new();
        assert_eq!(tracker.next_sequence(), 0);
        assert_eq!(tracker.next_sequence(), 1);
        assert_eq!(tracker.next_sequence(), 2);
    }

    #[test]
    fn test_ack_bits_generation() {
        let mut tracker = ReliabilityTracker::new();
        // Receive packet 10, 9, 8, 7
        tracker.on_packet_received(7);
        tracker.on_packet_received(8);
        tracker.on_packet_received(9);
        tracker.on_packet_received(10);

        let (ack, ack_bits) = tracker.generate_ack();
        assert_eq!(ack, 10);
        // Bit 0 = 9 (present: 1)
        // Bit 1 = 8 (present: 1)
        // Bit 2 = 7 (present: 1)
        assert_eq!(ack_bits & 0b111, 0b111);
    }

    #[test]
    fn test_ack_processing_retires_packets() {
        let mut tracker = ReliabilityTracker::new();
        tracker.queue_reliable(5, vec![1, 2, 3]);
        tracker.queue_reliable(6, vec![4, 5, 6]);
        assert_eq!(tracker.pending_reliable_count(), 2);

        // Remote sends ack=6 with ack_bits bit 0 set (seq 5 acked)
        tracker.process_incoming_ack(6, 0b1);
        assert_eq!(tracker.pending_reliable_count(), 0);
    }
}
