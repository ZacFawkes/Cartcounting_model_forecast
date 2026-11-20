using System.Text.Json;
using CartEvents;
using Dapper;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CartPublisher;

/// <summary>
/// Transactional outbox on SQL Server.
/// Port of cart_platform/integrations/outbox.py.
///
/// Events are enqueued next to the local result they describe and delivered
/// later by <see cref="OutboxRetryWorker"/>, so receiver downtime can never
/// fail a training or scoring request, and a restart loses nothing.
/// </summary>
public sealed class Outbox
{
    private const int DuplicateKeyError = 2627;
    private const int UniqueIndexViolation = 2601;

    private static readonly string[] Columns =
    {
        "id", "event_id", "event_type", "destination", "entity_id", "job_id",
        "model_id", "payload_json", "status", "attempts", "next_attempt_at",
        "claimed_at", "last_error", "response_code", "created_at", "sent_at"
    };

    private readonly OutboxConnectionFactory _factory;
    private readonly PublisherOptions _options;
    private readonly ILogger<Outbox> _logger;
    private readonly string _table;

    public Outbox(OutboxConnectionFactory factory, IOptions<PublisherOptions> options,
                  ILogger<Outbox> logger)
    {
        _factory = factory;
        _options = options.Value;
        _logger = logger;
        _table = $"[{factory.Schema}].[result_outbox]";
    }

    // ------------------------------------------------------------------
    // Enqueue
    // ------------------------------------------------------------------

    /// <summary>
    /// Queue an event. Returns false when the event_id is already queued,
    /// which is how a retried producer is stopped from sending twice.
    /// </summary>
    public async Task<bool> EnqueueAsync(ResultEvent e, string? destination = null,
                                         CancellationToken ct = default)
    {
        var invalid = e.Validate();
        if (invalid is not null)
            throw new ArgumentException(invalid, nameof(e));

        // The queued body is the exact envelope the receiver will be sent.
        var envelope = CanonicalJson.Serialize(JsonSerializer.SerializeToElement(e));

        await using var connection = await _factory.OpenAsync(ct);
        try
        {
            await connection.ExecuteAsync(new CommandDefinition($@"
INSERT INTO {_table}
    (event_id, event_type, destination, entity_id, job_id, model_id,
     payload_json, status, attempts, next_attempt_at, created_at)
VALUES
    (@event_id, @event_type, @destination, @entity_id, @job_id, @model_id,
     @payload_json, @status, 0, @next_attempt_at, @created_at);", new
            {
                event_id = e.EventId,
                event_type = e.EventType,
                destination = (destination ?? _options.ReceiverUrl).TrimEnd('/'),
                entity_id = e.EntityId,
                job_id = e.JobId,
                model_id = e.ModelId,
                payload_json = envelope,
                status = OutboxStatus.Pending,
                next_attempt_at = DateTime.UtcNow,
                created_at = DateTime.UtcNow
            }, cancellationToken: ct));
            return true;
        }
        catch (SqlException ex) when (ex.Number is DuplicateKeyError or UniqueIndexViolation)
        {
            _logger.LogDebug("Outbox replay suppressed for {EventId}", e.EventId);
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Claim and settle
    // ------------------------------------------------------------------

    /// <summary>
    /// Atomically move due pending rows to "sending" and return them.
    ///
    /// A single UPDATE ... OUTPUT over a TOP/ORDER BY CTE, with READPAST so
    /// concurrent workers skip each other's locked rows instead of blocking or
    /// double-sending. This replaces the Python select-then-update loop, which
    /// relied on SQLite's single writer.
    /// </summary>
    public async Task<IReadOnlyList<OutboxRow>> ClaimBatchAsync(
        int? limit = null, CancellationToken ct = default)
    {
        var output = string.Join(", ", Columns.Select(c => $"inserted.[{c}]"));
        await using var connection = await _factory.OpenAsync(ct);
        var rows = await connection.QueryAsync<OutboxRow>(new CommandDefinition($@"
WITH due AS (
    SELECT TOP (@limit) *
    FROM   {_table} WITH (READPAST, UPDLOCK, ROWLOCK)
    WHERE  status = @pending AND next_attempt_at <= @now
    ORDER  BY id
)
UPDATE due
SET    status = @sending, claimed_at = @now
OUTPUT {output};", new
        {
            limit = limit ?? _options.BatchSize,
            pending = OutboxStatus.Pending,
            sending = OutboxStatus.Sending,
            now = DateTime.UtcNow
        }, cancellationToken: ct));

        return rows.ToList();
    }

    public async Task MarkSentAsync(long id, int? responseCode,
                                    CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        await connection.ExecuteAsync(new CommandDefinition($@"
UPDATE {_table}
SET    status = @status, sent_at = @sent_at, response_code = @response_code,
       last_error = NULL
WHERE  id = @id;", new
        {
            status = OutboxStatus.Sent,
            sent_at = DateTime.UtcNow,
            response_code = responseCode,
            id
        }, cancellationToken: ct));
    }

    /// <summary>
    /// Schedule another attempt, or dead-letter the row once MaxAttempts is
    /// reached. Backoff is exponential with jitter, unless the receiver sent a
    /// Retry-After, which always wins.
    /// </summary>
    public async Task MarkRetryAsync(long id, int attempts, string error,
                                     int? responseCode = null,
                                     double? retryAfterSeconds = null,
                                     CancellationToken ct = default)
    {
        attempts += 1;
        await using var connection = await _factory.OpenAsync(ct);

        if (attempts >= _options.MaxAttempts)
        {
            await connection.ExecuteAsync(new CommandDefinition($@"
UPDATE {_table}
SET    status = @status, attempts = @attempts, last_error = @last_error,
       response_code = @response_code
WHERE  id = @id;", new
            {
                status = OutboxStatus.DeadLetter,
                attempts,
                last_error = $"max attempts reached: {error}",
                response_code = responseCode,
                id
            }, cancellationToken: ct));

            _logger.LogError("Outbox row {Id} dead-lettered after {Attempts} attempts",
                id, attempts);
            return;
        }

        double delay;
        if (retryAfterSeconds is not null)
        {
            delay = retryAfterSeconds.Value;
        }
        else
        {
            delay = Math.Min(_options.BaseBackoffSeconds * Math.Pow(2, attempts - 1),
                             _options.MaxBackoffSeconds);
            delay *= 0.5 + Random.Shared.NextDouble();   // jitter
        }

        await connection.ExecuteAsync(new CommandDefinition($@"
UPDATE {_table}
SET    status = @status, attempts = @attempts, next_attempt_at = @next_attempt_at,
       claimed_at = NULL, last_error = @last_error, response_code = @response_code
WHERE  id = @id;", new
        {
            status = OutboxStatus.Pending,
            attempts,
            next_attempt_at = DateTime.UtcNow.AddSeconds(delay),
            last_error = error,
            response_code = responseCode,
            id
        }, cancellationToken: ct));
    }

    /// <summary>Permanent failure: no retry can fix it (409, 4xx, unknown type).</summary>
    public async Task MarkDeadAsync(long id, int attempts, string error,
                                    int responseCode, CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        await connection.ExecuteAsync(new CommandDefinition($@"
UPDATE {_table}
SET    status = @status, attempts = @attempts, last_error = @last_error,
       response_code = @response_code
WHERE  id = @id;", new
        {
            status = OutboxStatus.DeadLetter,
            attempts = attempts + 1,
            last_error = error,
            response_code = responseCode,
            id
        }, cancellationToken: ct));
    }

    // ------------------------------------------------------------------
    // Recovery
    // ------------------------------------------------------------------

    /// <summary>
    /// Return rows abandoned mid-flight by a crashed worker to "pending",
    /// judged by how long ago they were claimed.
    /// </summary>
    public async Task<int> RecoverStuckAsync(CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        var recovered = await connection.ExecuteAsync(new CommandDefinition($@"
UPDATE {_table}
SET    status = @pending, claimed_at = NULL
WHERE  status = @sending
  AND  sent_at IS NULL
  AND  (claimed_at IS NULL OR claimed_at <= @cutoff);", new
        {
            pending = OutboxStatus.Pending,
            sending = OutboxStatus.Sending,
            cutoff = DateTime.UtcNow.AddSeconds(-_options.StuckAfterSeconds)
        }, cancellationToken: ct));

        if (recovered > 0)
            _logger.LogWarning("Recovered {Count} stuck outbox row(s)", recovered);
        return recovered;
    }

    /// <summary>
    /// Called at startup: with a single worker nothing can legitimately be in
    /// flight yet. Run only when this process owns the queue.
    /// </summary>
    public async Task<int> ResetSendingAsync(CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        var reset = await connection.ExecuteAsync(new CommandDefinition($@"
UPDATE {_table} SET status = @pending, claimed_at = NULL WHERE status = @sending;",
            new { pending = OutboxStatus.Pending, sending = OutboxStatus.Sending },
            cancellationToken: ct));

        if (reset > 0)
            _logger.LogInformation("Reset {Count} in-flight outbox row(s) after restart",
                reset);
        return reset;
    }

    /// <summary>Requeue dead letters, all of them or one event.</summary>
    public async Task<int> RetryDeadLettersAsync(string? eventId = null,
                                                 CancellationToken ct = default)
    {
        var filter = eventId is null ? string.Empty : " AND event_id = @event_id";
        await using var connection = await _factory.OpenAsync(ct);
        return await connection.ExecuteAsync(new CommandDefinition($@"
UPDATE {_table}
SET    status = @pending, attempts = 0, next_attempt_at = @now,
       claimed_at = NULL, last_error = NULL
WHERE  status = @dead{filter};", new
        {
            pending = OutboxStatus.Pending,
            dead = OutboxStatus.DeadLetter,
            now = DateTime.UtcNow,
            event_id = eventId
        }, cancellationToken: ct));
    }

    // ------------------------------------------------------------------
    // Observability
    // ------------------------------------------------------------------

    public async Task<OutboxStats> StatsAsync(CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        var rows = await connection.QueryAsync<(string Status, int N)>(
            new CommandDefinition(
                $"SELECT status, COUNT(*) AS n FROM {_table} GROUP BY status;",
                cancellationToken: ct));

        var stats = new OutboxStats();
        foreach (var (status, n) in rows)
        {
            switch (status)
            {
                case OutboxStatus.Pending: stats.Pending = n; break;
                case OutboxStatus.Sending: stats.Sending = n; break;
                case OutboxStatus.Sent: stats.Sent = n; break;
                case OutboxStatus.DeadLetter: stats.DeadLetter = n; break;
            }
        }
        return stats;
    }

    public async Task<IReadOnlyList<OutboxRow>> ListAsync(
        string? status = null, string? entityId = null, string? eventType = null,
        int limit = 100, CancellationToken ct = default)
    {
        var clauses = new List<string> { "1 = 1" };
        if (status is not null) clauses.Add("status = @status");
        if (entityId is not null) clauses.Add("entity_id = @entity_id");
        if (eventType is not null) clauses.Add("event_type = @event_type");

        await using var connection = await _factory.OpenAsync(ct);
        var rows = await connection.QueryAsync<OutboxRow>(new CommandDefinition($@"
SELECT TOP (@limit) * FROM {_table}
WHERE {string.Join(" AND ", clauses)}
ORDER BY id DESC;",
            new
            {
                status,
                entity_id = entityId,
                event_type = eventType,
                limit = Math.Clamp(limit, 1, 1000)
            }, cancellationToken: ct));

        return rows.ToList();
    }

    public async Task<OutboxRow?> GetAsync(string eventId, CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        return await connection.QueryFirstOrDefaultAsync<OutboxRow>(new CommandDefinition(
            $"SELECT * FROM {_table} WHERE event_id = @event_id;",
            new { event_id = eventId }, cancellationToken: ct));
    }

    /// <summary>Fails fast at startup when the table is missing.</summary>
    public async Task EnsureReadyAsync(CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        var exists = await connection.ExecuteScalarAsync<int>(new CommandDefinition(
            @"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
              WHERE TABLE_SCHEMA = @schema AND TABLE_NAME = 'result_outbox';",
            new { schema = _factory.Schema }, cancellationToken: ct));

        if (exists == 0)
        {
            throw new InvalidOperationException(
                $"Table [{_factory.Schema}].[result_outbox] is missing. " +
                "Run sql/04_create_outbox.sql.");
        }
    }
}
