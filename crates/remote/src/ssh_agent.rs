//! SSH agent forwarding as a protocol-agnostic byte tunnel, shared by both
//! ends of the remote protocol.
//!
//! The remote server (guest) listens on a Unix socket exported as
//! `SSH_AUTH_SOCK` and, for each accepted connection, asks the client (host)
//! to open a tunnel; the host connects to its own local agent and both ends
//! pipe raw bytes. One guest socket connection maps 1:1 to one host agent
//! connection. The tunnel itself (proto, guest listener, `pipe()`) is
//! byte-transparent — it never parses the ssh-agent framing, so any stream
//! protocol would forward the same way (mirroring how VS Code / OpenSSH agent
//! forwarding works). The one exception is the Windows host connector, which
//! must interpret the framing to work around an OS constraint (see
//! `named_pipe_agent_connector`); the transparency holds everywhere else.
//!
//! Each `ForwardSshAgentData` chunk is a request acknowledged with `Ack`, so a
//! slow reader on the far end applies backpressure to the near end's socket
//! reads. Chunks are read and sent sequentially per connection, preserving
//! order; distinct connections are independent and demultiplexed by
//! `connection_id`.
//!
//! The tunnel operates on boxed `AsyncRead`/`AsyncWrite` halves rather than a
//! concrete socket type, and the host obtains its agent connection through an
//! injected connector, so the whole path is exercised in-memory by the tests
//! without touching real sockets.

use anyhow::Result;
use collections::HashMap;
use futures::{
    AsyncRead, AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _, StreamExt as _,
    channel::mpsc, future::BoxFuture,
};
use gpui::{App, AppContext as _, AsyncApp, Context, Entity, Task, WeakEntity};
use parking_lot::Mutex;
use rpc::{AnyProtoClient, TypedEnvelope, proto};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering::SeqCst},
};

/// Read chunk size for pumping socket bytes onto the tunnel.
const CHUNK_SIZE: usize = 16 * 1024;

pub type BoxedReader = Box<dyn AsyncRead + Unpin + Send>;
pub type BoxedWriter = Box<dyn AsyncWrite + Unpin + Send>;

/// Opens a fresh connection to the host's local agent, yielding its read/write
/// halves. Injected so tests can supply an in-memory agent.
pub type AgentConnector =
    Arc<dyn Fn() -> BoxFuture<'static, Result<(BoxedReader, BoxedWriter)>> + Send + Sync>;

/// The write end of one tunneled connection: bytes pushed here are written to
/// the local socket by that connection's writer task. Dropping the sender ends
/// the writer task and closes the socket.
type ConnectionSink = mpsc::UnboundedSender<Vec<u8>>;

pub struct SshAgentProxy {
    client: AnyProtoClient,
    connections: Arc<Mutex<HashMap<u64, ConnectionSink>>>,
    next_connection_id: AtomicU64,
    /// `Some` on the host (connects to the local agent on `Open`); `None` on
    /// the guest (which originates connections instead).
    agent_connector: Option<AgentConnector>,
}

impl SshAgentProxy {
    /// Creates the proxy and registers the protocol handlers for this end.
    /// Passing an `agent_connector` marks this as the host and registers the
    /// `Open` handler; the guest passes `None`. Both ends handle `Data`/`Close`.
    /// The returned entity must be retained for the connection's lifetime —
    /// dropping it deregisters the handlers.
    pub fn new(
        client: AnyProtoClient,
        agent_connector: Option<AgentConnector>,
        cx: &mut App,
    ) -> Entity<Self> {
        let is_host = agent_connector.is_some();
        let entity = cx.new(|_| Self {
            client: client.clone(),
            connections: Arc::new(Mutex::new(HashMap::default())),
            next_connection_id: AtomicU64::new(0),
            agent_connector,
        });

        if is_host {
            client.add_request_handler(entity.downgrade(), Self::handle_open);
        }
        client.add_request_handler(entity.downgrade(), Self::handle_data);
        client.add_request_handler(entity.downgrade(), Self::handle_close);
        entity
    }

    /// Guest side: adopt a freshly accepted connection on the forwarding
    /// socket. Asks the host to open its end, then pipes bytes both ways.
    pub fn open_local_connection(
        &self,
        reader: BoxedReader,
        writer: BoxedWriter,
        cx: &mut Context<Self>,
    ) {
        let connection_id = self.next_connection_id.fetch_add(1, SeqCst);
        let client = self.client.clone();
        let connections = self.connections.clone();
        cx.spawn(async move |this, cx| {
            if client
                .request(proto::ForwardSshAgentOpen { connection_id })
                .await
                .is_err()
            {
                // The host has no agent (or it failed); drop the connection.
                return;
            }
            this.update(cx, |_this, cx| {
                pipe(connection_id, reader, writer, client, connections, cx);
            })
            .ok();
        })
        .detach();
    }

    /// Host side: a new tunnel opened; connect to the local agent and pipe.
    async fn handle_open(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::ForwardSshAgentOpen>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let connection_id = envelope.payload.connection_id;
        let connector = this.update(&mut cx, |this, _| this.agent_connector.clone());
        let connector = connector.ok_or_else(|| anyhow::anyhow!("no agent connector"))?;
        let (reader, writer) = connector().await?;
        this.update(&mut cx, |this, cx| {
            let client = this.client.clone();
            let connections = this.connections.clone();
            pipe(connection_id, reader, writer, client, connections, cx);
        });
        Ok(proto::Ack {})
    }

    async fn handle_data(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::ForwardSshAgentData>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let payload = envelope.payload;
        this.update(&mut cx, |this, _cx| {
            if let Some(sink) = this.connections.lock().get(&payload.connection_id) {
                // Unbounded, but Ack-per-chunk means only one chunk per
                // direction is ever in flight, so this cannot grow unbounded.
                sink.unbounded_send(payload.payload).ok();
            }
        });
        Ok(proto::Ack {})
    }

    async fn handle_close(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::ForwardSshAgentClose>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let connection_id = envelope.payload.connection_id;
        this.update(&mut cx, |this, _cx| {
            this.connections.lock().remove(&connection_id);
        });
        Ok(proto::Ack {})
    }
}

/// The production agent connector on Unix: connect a Unix socket to the host's
/// `SSH_AUTH_SOCK` and hand back its halves.
#[cfg(unix)]
pub fn unix_socket_agent_connector() -> AgentConnector {
    Arc::new(|| {
        Box::pin(async move {
            let socket_path = std::env::var("SSH_AUTH_SOCK")
                .map_err(|_| anyhow::anyhow!("no ssh-agent on the client (SSH_AUTH_SOCK unset)"))?;
            let stream = smol::net::unix::UnixStream::connect(&socket_path)
                .await
                .map_err(|e| anyhow::anyhow!("connecting to ssh-agent at {socket_path}: {e}"))?;
            let (reader, writer) = stream.split();
            Ok((
                Box::new(reader) as BoxedReader,
                Box::new(writer) as BoxedWriter,
            ))
        })
    })
}

/// The production agent connector on Windows: connect to the OpenSSH
/// Authentication Agent's named pipe. Windows exposes the agent as a named
/// pipe rather than a Unix socket, so there is no `SSH_AUTH_SOCK`; the path is
/// well-known.
///
/// The pipe is opened as a synchronous file handle, and Windows serializes I/O
/// on a synchronous handle per file object (the `FO_SYNCHRONOUS_IO` lock). Since
/// `pipe()` runs the reader and writer concurrently, a reader blocked in
/// `ReadFile` awaiting a reply would hold that lock and prevent the writer's
/// `WriteFile` from ever sending the request — a deadlock. Duplicating the
/// handle does not help: `DuplicateHandle` still points at the same synchronous
/// file object, so the same lock serializes both.
///
/// Instead, a single dedicated thread drives the real pipe strictly
/// request-then-reply (the ssh-agent protocol is half-duplex: each message is a
/// big-endian `u32` length followed by that many bytes), bridging to the tunnel
/// through in-memory async pipes. Only this Windows connector interprets the
/// ssh-agent framing; the rest of the forwarding path stays byte-transparent.
#[cfg(windows)]
pub fn named_pipe_agent_connector() -> AgentConnector {
    use std::io::{Read, Write as _};

    const AGENT_PIPE: &str = r"\\.\pipe\openssh-ssh-agent";
    // OpenSSH's AGENT_MAX_MSGLEN. Guards against an over-large allocation from a
    // corrupt length field, a risk introduced by parsing the framing here.
    const MAX_MSG_LEN: usize = 256 * 1024;

    fn too_large() -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "ssh-agent message too large")
    }
    async fn read_framed_async(src: &mut (impl AsyncRead + Unpin)) -> std::io::Result<Vec<u8>> {
        let mut len = [0u8; 4];
        src.read_exact(&mut len).await?;
        let n = u32::from_be_bytes(len) as usize;
        if n > MAX_MSG_LEN {
            return Err(too_large());
        }
        let mut msg = vec![0u8; 4 + n];
        msg[..4].copy_from_slice(&len);
        src.read_exact(&mut msg[4..]).await?;
        Ok(msg)
    }
    fn read_framed_blocking(src: &mut impl Read) -> std::io::Result<Vec<u8>> {
        let mut len = [0u8; 4];
        src.read_exact(&mut len)?;
        let n = u32::from_be_bytes(len) as usize;
        if n > MAX_MSG_LEN {
            return Err(too_large());
        }
        let mut msg = vec![0u8; 4 + n];
        msg[..4].copy_from_slice(&len);
        src.read_exact(&mut msg[4..])?;
        Ok(msg)
    }

    Arc::new(|| {
        Box::pin(async move {
            let mut pipe = smol::unblock(|| {
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(AGENT_PIPE)
            })
            .await
            .map_err(|e| anyhow::anyhow!("connecting to ssh-agent pipe {AGENT_PIPE}: {e}"))?;

            // tunnel -> req_writer -> (thread) req_reader -> real pipe
            // real pipe -> (thread) resp_writer -> resp_reader -> tunnel
            let (req_writer, mut req_reader) = async_pipe::pipe();
            let (mut resp_writer, resp_reader) = async_pipe::pipe();

            std::thread::spawn(move || {
                smol::block_on(async move {
                    loop {
                        let Ok(request) = read_framed_async(&mut req_reader).await else {
                            break;
                        };
                        if pipe.write_all(&request).is_err() {
                            break;
                        }
                        let Ok(reply) = read_framed_blocking(&mut pipe) else {
                            break;
                        };
                        if resp_writer.write_all(&reply).await.is_err() {
                            break;
                        }
                    }
                });
            });

            Ok((
                Box::new(resp_reader) as BoxedReader,
                Box::new(req_writer) as BoxedWriter,
            ))
        })
    })
}

/// Splits the work of one tunneled connection into a reader task (local bytes
/// -> `ForwardSshAgentData`) and a writer task (incoming bytes -> local
/// socket), registering the writer sink under `connection_id`.
fn pipe(
    connection_id: u64,
    mut reader: BoxedReader,
    mut writer: BoxedWriter,
    client: AnyProtoClient,
    connections: Arc<Mutex<HashMap<u64, ConnectionSink>>>,
    cx: &mut Context<SshAgentProxy>,
) {
    let (tx, mut rx) = mpsc::unbounded::<Vec<u8>>();
    connections.lock().insert(connection_id, tx);

    // Writer: drain inbound chunks to the local socket until the sink is
    // dropped (peer closed) or a write fails.
    cx.background_spawn(async move {
        while let Some(chunk) = rx.next().await {
            if writer.write_all(&chunk).await.is_err() {
                break;
            }
        }
        writer.close().await.ok();
    })
    .detach();

    // Reader: pump local bytes onto the tunnel, acknowledged per chunk, then
    // signal close.
    cx.spawn(async move |_this, _cx| {
        let mut buffer = vec![0u8; CHUNK_SIZE];
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let request = proto::ForwardSshAgentData {
                        connection_id,
                        payload: buffer[..n].to_vec(),
                    };
                    if client.request(request).await.is_err() {
                        break;
                    }
                }
            }
        }
        connections.lock().remove(&connection_id);
        client
            .request(proto::ForwardSshAgentClose { connection_id })
            .await
            .ok();
    })
    .detach();
}

/// Guest-side helper: the message-handler entity plus the socket path exported
/// as `SSH_AUTH_SOCK`. Held by the server for the session's lifetime.
#[cfg(unix)]
pub struct SshAgentListener {
    pub socket_path: std::path::PathBuf,
    pub _proxy: Entity<SshAgentProxy>,
    pub _task: Task<()>,
}

#[cfg(unix)]
impl SshAgentListener {
    /// Binds the forwarding socket in `server_dir` and starts accepting.
    pub fn start(
        server_dir: &std::path::Path,
        client: AnyProtoClient,
        cx: &mut App,
    ) -> Result<Self> {
        use smol::net::unix::UnixListener;

        let socket_path = server_dir.join("ssh-agent.sock");
        std::fs::remove_file(&socket_path).ok();
        let listener = UnixListener::bind(&socket_path)?;

        let proxy = SshAgentProxy::new(client, None, cx);
        let task = cx.spawn({
            let proxy = proxy.downgrade();
            async move |cx: &mut AsyncApp| accept_loop(listener, proxy, cx).await
        });

        Ok(Self {
            socket_path,
            _proxy: proxy,
            _task: task,
        })
    }
}

#[cfg(unix)]
async fn accept_loop(
    listener: smol::net::unix::UnixListener,
    proxy: WeakEntity<SshAgentProxy>,
    cx: &mut AsyncApp,
) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            break;
        };
        let (reader, writer) = stream.split();
        if proxy
            .update(cx, |proxy, cx| {
                proxy.open_local_connection(Box::new(reader), Box::new(writer), cx)
            })
            .is_err()
        {
            break; // proxy dropped -> session ending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote_client::ChannelClient;
    use gpui::TestAppContext;
    use rpc::proto::Envelope;

    /// Wires two `ChannelClient`s so each one's outgoing envelopes become the
    /// other's incoming — the in-memory equivalent of the remote transport.
    fn connect_clients(cx: &mut TestAppContext) -> (AnyProtoClient, AnyProtoClient) {
        let (a_out_tx, mut a_out_rx) = mpsc::unbounded::<Envelope>();
        let (a_in_tx, a_in_rx) = mpsc::unbounded::<Envelope>();
        let (b_out_tx, mut b_out_rx) = mpsc::unbounded::<Envelope>();
        let (b_in_tx, b_in_rx) = mpsc::unbounded::<Envelope>();

        let a = cx.update(|cx| ChannelClient::new(a_in_rx, a_out_tx, cx, "a", false));
        let b = cx.update(|cx| ChannelClient::new(b_in_rx, b_out_tx, cx, "b", false));

        cx.executor()
            .spawn(async move {
                while let Some(env) = a_out_rx.next().await {
                    b_in_tx.unbounded_send(env).ok();
                }
            })
            .detach();
        cx.executor()
            .spawn(async move {
                while let Some(env) = b_out_rx.next().await {
                    a_in_tx.unbounded_send(env).ok();
                }
            })
            .detach();

        (a.into(), b.into())
    }

    /// An in-memory bidirectional stream as a (reader, writer) pair for one end,
    /// plus the mirrored pair for the other end.
    fn duplex() -> ((BoxedReader, BoxedWriter), (BoxedReader, BoxedWriter)) {
        let (a_to_b_w, a_to_b_r) = async_pipe::pipe();
        let (b_to_a_w, b_to_a_r) = async_pipe::pipe();
        (
            (Box::new(b_to_a_r), Box::new(a_to_b_w)),
            (Box::new(a_to_b_r), Box::new(b_to_a_w)),
        )
    }

    #[gpui::test]
    async fn test_agent_bytes_round_trip(cx: &mut TestAppContext) {
        let (host_client, guest_client) = connect_clients(cx);

        // Host: an in-memory "agent" that appends a marker byte to whatever it
        // receives, so the test can distinguish request from response bytes. The
        // agent task runs on the test executor; the connector just hands the
        // host its pre-wired halves (called once for this single connection).
        let (host_side, agent_side) = duplex();
        let (mut agent_reader, mut agent_writer) = agent_side;
        cx.executor()
            .spawn(async move {
                let mut buf = vec![0u8; 1024];
                while let Ok(n) = agent_reader.read(&mut buf).await {
                    if n == 0 {
                        break;
                    }
                    let mut response = buf[..n].to_vec();
                    response.push(0xAB); // agent marker
                    if agent_writer.write_all(&response).await.is_err() {
                        break;
                    }
                }
            })
            .detach();
        let host_side = Arc::new(Mutex::new(Some(host_side)));
        let agent_connector: AgentConnector = Arc::new(move || {
            let host_side = host_side.lock().take();
            Box::pin(async move {
                host_side.ok_or_else(|| anyhow::anyhow!("agent already connected"))
            })
        });

        let _host = cx.update(|cx| SshAgentProxy::new(host_client, Some(agent_connector), cx));
        let guest = cx.update(|cx| SshAgentProxy::new(guest_client, None, cx));

        // Guest: feed a freshly "accepted" connection (an in-memory stream
        // standing in for the ssh client's socket).
        let (client_side, guest_side) = duplex();
        let (mut client_reader, mut client_writer) = client_side;
        guest.update(cx, |guest, cx| {
            guest.open_local_connection(guest_side.0, guest_side.1, cx);
        });

        // The ssh client writes a request; expect it back with the agent marker.
        let request = b"\x00\x00\x00\x01\x0b".to_vec(); // framed REQUEST_IDENTITIES
        let read_task = cx.executor().spawn(async move {
            client_writer.write_all(&request).await.unwrap();
            let mut response = vec![0u8; request.len() + 1];
            client_reader.read_exact(&mut response).await.unwrap();
            response
        });

        cx.run_until_parked();
        let response = read_task.await;

        let mut expected = b"\x00\x00\x00\x01\x0b".to_vec();
        expected.push(0xAB);
        assert_eq!(response, expected);
    }

    #[gpui::test]
    async fn test_open_fails_when_host_has_no_agent(cx: &mut TestAppContext) {
        let (host_client, guest_client) = connect_clients(cx);

        // Host connector always fails (no agent).
        let agent_connector: AgentConnector =
            Arc::new(|| Box::pin(async { anyhow::bail!("no agent") }));

        let _host = cx.update(|cx| SshAgentProxy::new(host_client, Some(agent_connector), cx));
        let guest = cx.update(|cx| SshAgentProxy::new(guest_client, None, cx));

        let (client_side, guest_side) = duplex();
        let (mut client_reader, _client_writer) = client_side;
        guest.update(cx, |guest, cx| {
            guest.open_local_connection(guest_side.0, guest_side.1, cx);
        });

        // The guest drops the connection when Open fails, so the client sees EOF.
        let read_task = cx.executor().spawn(async move {
            let mut buf = [0u8; 1];
            client_reader.read(&mut buf).await
        });
        cx.run_until_parked();
        assert_eq!(read_task.await.unwrap(), 0, "expected EOF on failed open");
    }
}
