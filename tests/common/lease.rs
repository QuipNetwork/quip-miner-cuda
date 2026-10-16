// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Ported from the Metal lease coordinator at 98e04a0.
//! A protocol-2 test coordinator for salt leases. One miner process connects
//! over a Unix socket. The coordinator folds every miner frame into a `Log`.

use quip_solver_core::quip_proto::v1::{
    coord_msg, miner_msg,
    miner_service_server::{MinerService, MinerServiceServer},
    Configure, CoordMsg, EdgeList, Fatal, GeneratorAlgorithm, Hello, IsingProblemGenerator, Job,
    JobKind, LeaseDone, MinerMsg, Ping, Reject, SetTarget, Shutdown, Topology, Welcome,
};
use quip_solver_core::quip_protocol::lease::TopologyView;
use quip_solver_core::quip_protocol::target::Target;
use std::sync::Mutex;
use std::time::Duration;
use std::{io::Write, process::Stdio};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::UnixListener;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::{Request, Response, Status, Streaming};

type Outbound = mpsc::Sender<Result<CoordMsg, Status>>;
type Connection = (Streaming<MinerMsg>, Outbound);

struct Coordinator(Mutex<Option<oneshot::Sender<Connection>>>);

#[tonic::async_trait]
impl MinerService for Coordinator {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let (send, receive) = mpsc::channel(256);
        let connection = self
            .0
            .lock()
            .expect("coordinator lock")
            .take()
            .ok_or_else(|| Status::already_exists("one miner connection expected"))?;
        connection
            .send((request.into_inner(), send))
            .map_err(|_| Status::cancelled("test stopped"))?;
        Ok(Response::new(ReceiverStream::new(receive)))
    }
}

/// Every frame the miner sent, folded by kind.
#[derive(Debug, Default)]
pub(crate) struct Log {
    pub(crate) hello: Option<Hello>,
    pub(crate) ready: bool,
    /// Credits per `JobRequest`. The first entry is the initial grant.
    pub(crate) credits: Vec<u32>,
    pub(crate) results: Vec<quip_solver_core::quip_proto::v1::Result>,
    pub(crate) lease_done: Vec<LeaseDone>,
    pub(crate) rejects: Vec<Reject>,
    /// `jobs_done` from the latest `Status`.
    pub(crate) jobs_done: Option<u64>,
    pub(crate) fatal: Option<Fatal>,
    pub(crate) closed: bool,
    /// Arrival order: `result:<job>`, `done:<job>`, `credit`.
    pub(crate) order: Vec<String>,
}

impl Log {
    /// Credits returned after the initial grant.
    pub(crate) fn refunds(&self) -> u32 {
        self.credits.iter().skip(1).sum()
    }

    pub(crate) fn done(&self, job: &[u8]) -> Option<&LeaseDone> {
        self.lease_done.iter().find(|d| d.job_id == job)
    }

    pub(crate) fn done_count(&self, job: &[u8]) -> usize {
        self.lease_done.iter().filter(|d| d.job_id == job).count()
    }

    pub(crate) fn results_for(&self, job: &[u8]) -> Vec<&quip_solver_core::quip_proto::v1::Result> {
        self.results.iter().filter(|r| r.job_id == job).collect()
    }
}

pub(crate) struct Session {
    pub(crate) log: Log,
    inbound: Streaming<MinerMsg>,
    outbound: Option<Outbound>,
    child: Child,
    stderr: std::path::PathBuf,
    log_lines: mpsc::UnboundedReceiver<String>,
    logger: tokio::task::JoinHandle<()>,
    _server: tokio::task::JoinHandle<()>,
    _directory: tempfile::TempDir,
}

impl Session {
    /// Spawn `binary`, finish the handshake, and wait for the first credits.
    pub(crate) async fn start(binary: &str, queue_depth: u32, backend_toml: &str) -> Self {
        let directory = tempfile::tempdir().expect("socket directory");
        let path = directory.path().join("coord.sock");
        let listener = UnixListener::bind(&path).expect("bind test coordinator");
        let (send, receive) = oneshot::channel();
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(MinerServiceServer::new(Coordinator(Mutex::new(Some(send)))))
                .serve_with_incoming(UnixListenerStream::new(listener))
                .await
                .expect("serve test coordinator");
        });
        let stderr = directory.path().join("miner.log");
        let mut log_file = std::fs::File::create(&stderr).expect("miner log");
        let mut child = Command::new(binary)
            .args(["--quip-coordinator", &format!("unix://{}", path.display())])
            .args(["--miner-id", "cuda-0", "--device", "0"])
            .stderr(Stdio::piped())
            .env(
                "RUST_LOG",
                "info,quip_solver_cuda::lease=debug,quip_solver_cuda::lease_turn=debug",
            )
            .env("QUIP_SESSION_TOKEN", "test-token")
            .kill_on_drop(true)
            .spawn()
            .expect("start miner");
        let mut lines = BufReader::new(child.stderr.take().expect("miner stderr")).lines();
        let (log_send, log_lines) = mpsc::unbounded_channel();
        let logger = tokio::spawn(async move {
            while let Some(line) = lines.next_line().await.expect("read miner stderr") {
                writeln!(log_file, "{line}").expect("write miner log");
                drop(log_send.send(line));
            }
        });
        let (inbound, outbound) = tokio::time::timeout(Duration::from_secs(30), receive)
            .await
            .expect("miner connects within 30 s")
            .expect("coordinator connection");
        let mut session = Self {
            log: Log::default(),
            inbound,
            outbound: Some(outbound),
            child,
            stderr,
            log_lines,
            logger,
            _server: server,
            _directory: directory,
        };
        session
            .until("hello", Duration::from_secs(30), |log| log.hello.is_some())
            .await;
        session
            .send(coord_msg::Msg::Welcome(Welcome {
                protocol_version: 2,
            }))
            .await;
        session
            .send(coord_msg::Msg::Configure(Configure {
                queue_depth,
                idle_timeout_s: 3_600,
                heartbeat_s: 15,
                reconnect_window_s: 60,
                backend_toml: backend_toml.to_owned(),
            }))
            .await;
        session
            .until("ready", Duration::from_mins(1), |log| {
                log.ready && !log.credits.is_empty()
            })
            .await;
        session
    }

    pub(crate) async fn send(&mut self, msg: coord_msg::Msg) {
        self.outbound
            .as_ref()
            .expect("open coordinator stream")
            .send(Ok(CoordMsg { msg: Some(msg) }))
            .await
            .expect("coordinator send");
    }

    /// Fold frames until `done` holds. Panics on a timeout or an early close.
    pub(crate) async fn until(
        &mut self,
        phase: &str,
        limit: Duration,
        done: impl Fn(&Log) -> bool,
    ) {
        let deadline = Instant::now() + limit;
        while !done(&self.log) {
            assert!(
                !self.log.closed,
                "{phase}: stream closed early, fatal {:?}",
                self.log.fatal
            );
            match tokio::time::timeout_at(deadline, self.inbound.message()).await {
                Ok(Ok(Some(frame))) => self.observe(frame),
                Ok(Ok(None)) => self.log.closed = true,
                Ok(Err(error)) => panic!("{phase}: transport error: {error}"),
                Err(error) => panic!(
                    "{phase}: timed out after {limit:?}: {error}\n{}",
                    self.stderr()
                ),
            }
        }
    }

    /// Ping and return `Status.jobs_done`, which counts finished salts.
    pub(crate) async fn jobs_done(&mut self) -> u64 {
        self.log.jobs_done = None;
        self.send(coord_msg::Msg::Ping(Ping {})).await;
        self.until("status", Duration::from_secs(10), |log| {
            log.jobs_done.is_some()
        })
        .await;
        self.log.jobs_done.unwrap_or(0)
    }

    /// Receive real session traffic for a fixed benchmark window.
    pub(crate) async fn collect_until(&mut self, deadline: Instant) {
        loop {
            match tokio::time::timeout_at(deadline, self.inbound.message()).await {
                Ok(Ok(Some(frame))) => self.observe(frame),
                Err(_) => return,
                other => panic!("benchmark stream ended: {other:?}\n{}", self.stderr()),
            }
            assert!(self.log.fatal.is_none(), "{:?}", self.log.fatal);
            assert!(self.log.rejects.is_empty(), "{:?}", self.log.rejects);
        }
    }

    pub(crate) fn stderr(&self) -> String {
        std::fs::read_to_string(&self.stderr).expect("miner log")
    }

    /// Wait for a device lifecycle event without consuming protocol frames.
    pub(crate) async fn until_stderr(&mut self, needle: &str, limit: Duration) {
        let deadline = Instant::now() + limit;
        loop {
            let line = tokio::time::timeout_at(deadline, self.log_lines.recv())
                .await
                .unwrap_or_else(|_| panic!("missing log event {needle}:\n{}", self.stderr()))
                .expect("miner log remains open");
            if line.contains(needle) {
                return;
            }
        }
    }

    /// Send `Shutdown`, read to the end of the stream, and return the exit code.
    pub(crate) async fn shutdown(&mut self, grace_ms: u32) -> i32 {
        let started = Instant::now();
        self.send(coord_msg::Msg::Shutdown(Shutdown { grace_ms }))
            .await;
        // End the command stream so core's final inbound drain can finish.
        // Leaving it open consumes the grace window before NVML teardown.
        drop(self.outbound.take());
        let limit = Duration::from_millis(u64::from(grace_ms)) + Duration::from_secs(10);
        self.until("shutdown", limit, |log| log.closed).await;
        let status = tokio::time::timeout(Duration::from_secs(10), self.child.wait())
            .await
            .expect("miner exits after shutdown")
            .expect("wait for miner");
        (&mut self.logger).await.expect("miner log task");
        // Allow two seconds for transport and scheduler delays beyond grace.
        let tolerance = Duration::from_secs(2);
        let elapsed = started.elapsed();
        assert!(
            elapsed <= Duration::from_millis(u64::from(grace_ms)) + tolerance,
            "shutdown took {elapsed:?}, exceeding grace {grace_ms} ms plus {tolerance:?}"
        );
        status.code().unwrap_or(-1)
    }

    fn observe(&mut self, frame: MinerMsg) {
        let Some(message) = frame.msg else {
            panic!("empty miner message");
        };
        match message {
            miner_msg::Msg::Hello(hello) => self.log.hello = Some(hello),
            miner_msg::Msg::Ready(_) => self.log.ready = true,
            miner_msg::Msg::JobRequest(request) => {
                self.log.credits.push(request.credits);
                self.log.order.push("credit".to_owned());
            }
            miner_msg::Msg::Result(result) => {
                self.log.order.push(format!(
                    "result:{}",
                    String::from_utf8_lossy(&result.job_id)
                ));
                self.log.results.push(result);
            }
            miner_msg::Msg::LeaseDone(done) => {
                self.log
                    .order
                    .push(format!("done:{}", String::from_utf8_lossy(&done.job_id)));
                self.log.lease_done.push(done);
            }
            miner_msg::Msg::Reject(reject) => self.log.rejects.push(reject),
            miner_msg::Msg::Status(status) => self.log.jobs_done = Some(status.jobs_done),
            miner_msg::Msg::Fatal(fatal) => self.log.fatal = Some(fatal),
            miner_msg::Msg::Capabilities(_) => {}
        }
    }
}

/// A screenable ring with zero fields and both unit couplings.
pub(crate) fn ring(n: u32, hash: [u8; 32]) -> (Topology, TopologyView) {
    let topology = Topology {
        hash: hash.to_vec(),
        nodes: (0..n).collect(),
        edges: Some(EdgeList {
            u: (0..n).collect(),
            v: (0..n).map(|i| (i + 1) % n).collect(),
        }),
        allowed_h_milli: vec![0],
        allowed_j_milli: vec![-1000, 1000],
    };
    let view = TopologyView::from_proto(&topology).expect("valid ring topology");
    (topology, view)
}

/// An `ISING_GENERATE` job over `salt_start..salt_start + salt_count`.
pub(crate) fn lease(
    job_id: &[u8],
    generation: u64,
    hash: [u8; 32],
    salt_start: u64,
    salt_count: u64,
) -> Job {
    Job {
        job_id: job_id.to_vec(),
        kind: JobKind::IsingGenerate as i32,
        generation,
        deadline_ms: 0,
        ising: None,
        provenance: None,
        generator: Some(IsingProblemGenerator {
            algorithm: GeneratorAlgorithm::Blake3Chacha8V1 as i32,
            topology_hash: hash.to_vec(),
            last_proof_block_hash: vec![0x11; 32],
            miner_account: vec![0x22; 32],
            base_salt: vec![0x44; 32],
            salt_start,
            salt_count,
        }),
    }
}

/// 64 reads, one proof of up to 32 solutions below `max_energy_milli`.
pub(crate) fn target(max_energy_milli: i64, num_sweeps: u32) -> SetTarget {
    SetTarget {
        max_energy_milli,
        min_solutions: 1,
        min_diversity_milli: 0,
        num_reads: 64,
        num_sweeps,
        anneal_time_us: 0,
        max_proof_solutions: 32,
    }
}

/// A target that only checks a result's energies, nonce, and salt against
/// its redrawn problem — never its solution count or its energy against the
/// live session target. `LeaseSink::push` forwards every non-empty read set
/// unfiltered (no target check, no truncation to a proof-set size), so a
/// lease `Result` carries the salt's full read set: it can exceed the size
/// or miss the energy bound a session `Target` derived from `SetTarget`
/// would enforce for an on-chain proof. The coordinator, not this test,
/// decides which verified rows beat the live target.
pub(crate) fn permissive_target() -> Target {
    Target {
        max_energy_milli: i64::MAX,
        min_solutions: 1,
        min_diversity_milli: 0,
        max_proof_solutions: u32::MAX,
    }
}
