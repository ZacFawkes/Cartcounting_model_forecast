using System.Data;
using System.Globalization;
using System.Text.Json;
using CartEvents;
using Dapper;
using Microsoft.Data.SqlClient;
using ReceiverNet.Data;
using ReceiverNet.Models;

namespace ReceiverNet.Services;

/// <summary>
/// Idempotent inserts and filtered queries.
/// Port of cart_platform/receiver/store.py onto SQL Server.
/// </summary>
public sealed class EventStore
{
    private const int DuplicateKeyError = 2627;
    private const int UniqueIndexViolation = 2601;

    /// <summary>Columns a caller is allowed to filter on, per table.</summary>
    private static readonly Dictionary<string, HashSet<string>> FilterableColumns = new()
    {
        ["training_results"] = new()
        {
            "entity_id", "site_key", "upload_id", "job_id", "model_id", "status"
        },
        ["calibration_results"] = new()
        {
            "entity_id", "site_key", "upload_id", "job_id", "model_id", "status", "scope"
        },
        ["backtest_results"] = new()
        {
            "entity_id", "site_key", "upload_id", "job_id", "model_id", "status",
            "backtest_type"
        },
        ["scoring_results"] = new() { "entity_id", "site_key", "model_id" }
    };

    private readonly SqlConnectionFactory _factory;
    private readonly ILogger<EventStore> _logger;

    private readonly string _events;
    private readonly string _training;
    private readonly string _calibration;
    private readonly string _backtest;
    private readonly string _scoring;

    public EventStore(SqlConnectionFactory factory, ILogger<EventStore> logger)
    {
        _factory = factory;
        _logger = logger;

        string Qualify(string table) => $"[{factory.Schema}].[{table}]";
        _events = Qualify("received_events");
        _training = Qualify("training_results");
        _calibration = Qualify("calibration_results");
        _backtest = Qualify("backtest_results");
        _scoring = Qualify("scoring_results");
    }

    /// <summary>Schema-qualified, bracketed table name.</summary>
    private string Table(string table) => $"[{_factory.Schema}].[{table}]";

    // ------------------------------------------------------------------
    // Ingestion
    // ------------------------------------------------------------------

    /// <summary>
    /// Insert an event idempotently.
    /// <see cref="StoreOutcome.Created"/> for a new event,
    /// <see cref="StoreOutcome.Replay"/> for a byte-identical resend,
    /// <see cref="StoreOutcome.Conflict"/> when the same event_id arrives with a
    /// different canonical payload. Historical results are never overwritten.
    /// </summary>
    public async Task<(StoreOutcome Outcome, Dictionary<string, object?> Record)>
        StoreEventAsync(ResultEventIn incoming, CancellationToken ct = default)
    {
        var digest = CanonicalJson.Hash(incoming.Payload);

        await using var connection = await _factory.OpenAsync(ct);

        var existing = await FindEventAsync(connection, null, incoming.EventId, ct);
        if (existing is not null)
            return (ClassifyExisting(existing, digest), existing);

        await using var transaction =
            (SqlTransaction)await connection.BeginTransactionAsync(ct);
        try
        {
            await InsertEnvelopeAsync(connection, transaction, incoming, digest, ct);
            await InsertProjectionAsync(connection, transaction, incoming, ct);
            await transaction.CommitAsync(ct);
        }
        catch (SqlException ex) when (ex.Number is DuplicateKeyError or UniqueIndexViolation)
        {
            // Another worker inserted the same event_id between our read and
            // our write. Re-read and classify instead of failing the request.
            await transaction.RollbackAsync(ct);
            var raced = await FindEventAsync(connection, null, incoming.EventId, ct);
            if (raced is not null)
                return (ClassifyExisting(raced, digest), raced);

            // No matching event_id, so the clash came from a projection index -
            // in practice ux_scoring_decision: a different event_id claiming a
            // decision slot that is already taken. Retrying can never succeed,
            // so report it as a conflict rather than a retryable 5xx.
            throw new DuplicateProjectionException(
                $"event '{incoming.EventId}' violates a uniqueness constraint on the " +
                $"{incoming.EventType} projection (most likely one decision per " +
                "entity, decision_time and model)", ex);
        }
        catch
        {
            await transaction.RollbackAsync(ct);
            throw;
        }

        var stored = await FindEventAsync(connection, null, incoming.EventId, ct)
                     ?? throw new InvalidOperationException(
                         $"event '{incoming.EventId}' vanished after insert");

        _logger.LogInformation("Stored {EventType} event {EventId} (entity={EntityId})",
            incoming.EventType, incoming.EventId, incoming.EntityId);
        return (StoreOutcome.Created, stored);
    }

    private static StoreOutcome ClassifyExisting(Dictionary<string, object?> existing,
                                                 string digest)
    {
        var stored = existing.TryGetValue("payload_hash", out var value)
            ? value as string
            : null;
        return string.Equals(stored, digest, StringComparison.Ordinal)
            ? StoreOutcome.Replay
            : StoreOutcome.Conflict;
    }

    private async Task InsertEnvelopeAsync(SqlConnection connection,
                                           SqlTransaction transaction,
                                           ResultEventIn e, string digest,
                                           CancellationToken ct)
    {
        var sql = $@"
INSERT INTO {_events}
    (event_id, event_type, schema_version, entity_id, site_key, upload_id,
     job_id, model_id, occurred_at, received_at, payload_hash, payload_json)
VALUES
    (@event_id, @event_type, @schema_version, @entity_id, @site_key, @upload_id,
     @job_id, @model_id, @occurred_at, @received_at, @payload_hash, @payload_json);";

        await connection.ExecuteAsync(new CommandDefinition(sql, new
        {
            event_id = e.EventId,
            event_type = e.EventType,
            schema_version = e.SchemaVersion,
            entity_id = e.EntityId,
            site_key = e.SiteKey,
            upload_id = e.UploadId,
            job_id = e.JobId,
            model_id = e.ModelId,
            occurred_at = ParseInstant(e.OccurredAt),
            received_at = DateTime.UtcNow,
            payload_hash = digest,
            payload_json = e.Payload.GetRawText()
        }, transaction, cancellationToken: ct));
    }

    private async Task InsertProjectionAsync(SqlConnection connection,
                                             SqlTransaction transaction,
                                             ResultEventIn e, CancellationToken ct)
    {
        var p = e.Payload;
        var received = DateTime.UtcNow;
        var occurred = ParseInstant(e.OccurredAt);
        var configHash = Str(p, "config_hash");
        var frozen = Raw(p, "frozen_model_config") ?? Raw(p, "frozen_thresholds");
        var effective = Raw(p, "effective_operational_config")
                        ?? Raw(p, "effective_config_at_training")
                        ?? Raw(p, "effective_config");

        switch (e.EventType)
        {
            case "training.completed":
                await connection.ExecuteAsync(new CommandDefinition($@"
INSERT INTO {_training}
    (event_id, job_id, entity_id, site_key, upload_id, model_id, status,
     occurred_at, received_at, metrics_json, metadata_json,
     config_hash, frozen_config_json, effective_config_json)
VALUES
    (@event_id, @job_id, @entity_id, @site_key, @upload_id, @model_id, @status,
     @occurred_at, @received_at, @metrics_json, @metadata_json,
     @config_hash, @frozen_config_json, @effective_config_json);", new
                {
                    event_id = e.EventId,
                    job_id = e.JobId,
                    entity_id = e.EntityId,
                    site_key = e.SiteKey,
                    upload_id = e.UploadId,
                    model_id = e.ModelId,
                    status = Str(p, "status"),
                    occurred_at = occurred,
                    received_at = received,
                    metrics_json = Raw(p, "metrics"),
                    metadata_json = p.GetRawText(),
                    config_hash = configHash,
                    frozen_config_json = frozen,
                    effective_config_json = effective
                }, transaction, cancellationToken: ct));
                break;

            case "calibration.completed":
                await connection.ExecuteAsync(new CommandDefinition($@"
INSERT INTO {_calibration}
    (event_id, job_id, entity_id, site_key, upload_id, model_id, calibrator_id,
     status, scope, method, occurred_at, received_at, metrics_json,
     metadata_json, config_hash)
VALUES
    (@event_id, @job_id, @entity_id, @site_key, @upload_id, @model_id, @calibrator_id,
     @status, @scope, @method, @occurred_at, @received_at, @metrics_json,
     @metadata_json, @config_hash);", new
                {
                    event_id = e.EventId,
                    job_id = e.JobId,
                    entity_id = e.EntityId,
                    site_key = e.SiteKey,
                    upload_id = e.UploadId,
                    model_id = e.ModelId,
                    calibrator_id = Str(p, "calibrator_id"),
                    status = Str(p, "status"),
                    scope = Str(p, "scope"),
                    method = Str(p, "method"),
                    occurred_at = occurred,
                    received_at = received,
                    metrics_json = Raw(p, "metrics"),
                    metadata_json = p.GetRawText(),
                    config_hash = configHash
                }, transaction, cancellationToken: ct));
                break;

            case "backtest.completed":
                await connection.ExecuteAsync(new CommandDefinition($@"
INSERT INTO {_backtest}
    (event_id, job_id, entity_id, site_key, upload_id, model_id, report_id,
     backtest_type, status, reason, occurred_at, received_at, metrics_json,
     assumptions_json, caveat, artifacts_json,
     config_hash, frozen_config_json, effective_config_json)
VALUES
    (@event_id, @job_id, @entity_id, @site_key, @upload_id, @model_id, @report_id,
     @backtest_type, @status, @reason, @occurred_at, @received_at, @metrics_json,
     @assumptions_json, @caveat, @artifacts_json,
     @config_hash, @frozen_config_json, @effective_config_json);", new
                {
                    event_id = e.EventId,
                    job_id = e.JobId,
                    entity_id = e.EntityId,
                    site_key = e.SiteKey,
                    upload_id = e.UploadId,
                    model_id = e.ModelId,
                    report_id = Str(p, "report_id"),
                    backtest_type = Str(p, "backtest_type"),
                    status = Str(p, "status"),
                    reason = Str(p, "reason"),
                    occurred_at = occurred,
                    received_at = received,
                    metrics_json = Raw(p, "metrics"),
                    assumptions_json = Raw(p, "assumptions"),
                    caveat = Str(p, "caveat"),
                    artifacts_json = Raw(p, "artifacts"),
                    config_hash = configHash,
                    frozen_config_json = frozen,
                    effective_config_json = effective
                }, transaction, cancellationToken: ct));
                break;

            case "scoring.recorded":
                var action = Obj(p, "action");
                var forecast = Obj(p, "forecast");
                var (kdvr, chanNo) = SplitEntity(e.EntityId);
                await connection.ExecuteAsync(new CommandDefinition($@"
INSERT INTO {_scoring}
    (event_id, request_id, entity_id, kdvr, chan_no, site_key, decision_time,
     current_cart, simulated_cart, pred_cart_20min, pred_cart_40min,
     pred_min_20min, pred_min_40min, lower_bound_min_40min, shortage_risk,
     alert_state, quantity, model_id, calibration_id, scored_at)
VALUES
    (@event_id, @request_id, @entity_id, @kdvr, @chan_no, @site_key, @decision_time,
     @current_cart, @simulated_cart, @pred_cart_20min, @pred_cart_40min,
     @pred_min_20min, @pred_min_40min, @lower_bound_min_40min, @shortage_risk,
     @alert_state, @quantity, @model_id, @calibration_id, @scored_at);", new
                {
                    event_id = e.EventId,
                    request_id = Str(p, "request_id"),
                    entity_id = e.EntityId,
                    kdvr,
                    chan_no = chanNo,
                    site_key = e.SiteKey,
                    decision_time = ParseInstant(Str(p, "decision_time")),
                    current_cart = Num(p, "current_cart"),
                    simulated_cart = Num(p, "simulated_cart"),
                    pred_cart_20min = Num(forecast, "cart_at_20min"),
                    pred_cart_40min = Num(forecast, "cart_at_40min"),
                    pred_min_20min = Num(forecast, "min_within_20min"),
                    pred_min_40min = Num(forecast, "min_within_40min"),
                    lower_bound_min_40min = Num(forecast, "lower_bound_min_within_40min"),
                    shortage_risk = Num(p, "shortage_risk"),
                    alert_state = Str(p, "alert_state"),
                    quantity = Int(action, "quantity"),
                    model_id = e.ModelId,
                    calibration_id = Str(p, "calibration_id"),
                    scored_at = ParseInstant(Str(p, "scored_at")) ?? received
                }, transaction, cancellationToken: ct));
                break;
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    public Task<PageOut> ListTrainingAsync(ResultQuery q, int limit, int offset,
                                           CancellationToken ct = default) =>
        ListAsync("training_results", q, limit, offset, "occurred_at", ct);

    public Task<PageOut> ListCalibrationAsync(ResultQuery q, int limit, int offset,
                                              CancellationToken ct = default) =>
        ListAsync("calibration_results", q, limit, offset, "occurred_at", ct);

    public Task<PageOut> ListBacktestsAsync(ResultQuery q, int limit, int offset,
                                            CancellationToken ct = default) =>
        ListAsync("backtest_results", q, limit, offset, "occurred_at", ct);

    public Task<PageOut> ListScoringAsync(ResultQuery q, int limit, int offset,
                                          CancellationToken ct = default) =>
        ListAsync("scoring_results", q, limit, offset, "decision_time", ct);

    private async Task<PageOut> ListAsync(string table, ResultQuery q, int limit,
                                          int offset, string orderColumn,
                                          CancellationToken ct)
    {
        var allowed = FilterableColumns[table];
        var clauses = new List<string> { "1 = 1" };
        var parameters = new DynamicParameters();

        void Filter(string column, string? value)
        {
            if (value is null || !allowed.Contains(column))
                return;
            clauses.Add($"[{column}] = @{column}");
            parameters.Add(column, value);
        }

        Filter("entity_id", q.EntityId);
        Filter("site_key", q.SiteKey);
        Filter("upload_id", q.UploadId);
        Filter("job_id", q.JobId);
        Filter("model_id", q.ModelId);
        Filter("status", q.Status);
        Filter("scope", q.Scope);
        Filter("backtest_type", q.BacktestType);

        var since = ParseInstant(q.Since);
        if (since is not null)
        {
            clauses.Add($"[{orderColumn}] >= @since");
            parameters.Add("since", since, DbType.DateTime2);
        }

        var until = ParseInstant(q.Until);
        if (until is not null)
        {
            clauses.Add($"[{orderColumn}] <= @until");
            parameters.Add("until", until, DbType.DateTime2);
        }

        var where = string.Join(" AND ", clauses);
        await using var connection = await _factory.OpenAsync(ct);

        var total = await connection.ExecuteScalarAsync<int>(new CommandDefinition(
            $"SELECT COUNT(*) FROM {Table(table)} WHERE {where};",
            parameters, cancellationToken: ct));

        parameters.Add("limit", limit);
        parameters.Add("offset", offset);
        var rows = await connection.QueryAsync<dynamic>(new CommandDefinition(
            $@"SELECT * FROM {Table(table)} WHERE {where}
               ORDER BY [{orderColumn}] DESC, [id] DESC
               OFFSET @offset ROWS FETCH NEXT @limit ROWS ONLY;",
            parameters, cancellationToken: ct));

        return new PageOut
        {
            Total = total,
            Limit = limit,
            Offset = offset,
            Items = rows.Select(r => MapRow((object)r)).ToList()
        };
    }

    /// <summary>Single event with its payload rehydrated as JSON.</summary>
    public async Task<Dictionary<string, object?>?> GetEventAsync(
        string eventId, CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        var row = await FindEventAsync(connection, null, eventId, ct);
        if (row is null)
            return null;

        row.Remove("payload_json", out var payload);
        row["payload"] = payload is string json
            ? JsonSerializer.Deserialize<JsonElement>(json)
            : null;
        return row;
    }

    /// <summary>Everything the receiver knows about one entity.</summary>
    public async Task<Dictionary<string, object?>> EntitySummaryAsync(
        string entityId, CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        var args = new { entity_id = entityId };

        async Task<int> CountAsync(string table) =>
            await connection.ExecuteScalarAsync<int>(new CommandDefinition(
                $"SELECT COUNT(*) FROM {Table(table)} WHERE entity_id = @entity_id;",
                args, cancellationToken: ct));

        async Task<Dictionary<string, object?>?> LatestAsync(string table, string order)
        {
            var row = await connection.QueryFirstOrDefaultAsync<dynamic>(
                new CommandDefinition(
                    $@"SELECT TOP (1) * FROM {Table(table)} WHERE entity_id = @entity_id
                       ORDER BY [{order}] DESC, [id] DESC;",
                    args, cancellationToken: ct));
            return row is null ? null : MapRow((object)row);
        }

        var backtests = await connection.QueryAsync<dynamic>(new CommandDefinition(
            $@"SELECT backtest_type, status, report_id, job_id, model_id, occurred_at
               FROM {_backtest} WHERE entity_id = @entity_id
               ORDER BY occurred_at DESC, id DESC;",
            args, cancellationToken: ct));

        return new Dictionary<string, object?>
        {
            ["entity_id"] = entityId,
            ["counts"] = new Dictionary<string, int>
            {
                ["training"] = await CountAsync("training_results"),
                ["calibration"] = await CountAsync("calibration_results"),
                ["backtest"] = await CountAsync("backtest_results"),
                ["scoring"] = await CountAsync("scoring_results")
            },
            ["backtests"] = backtests.Select(r => MapRow((object)r)).ToList(),
            ["latest_training"] = await LatestAsync("training_results", "occurred_at"),
            ["latest_calibration"] = await LatestAsync("calibration_results", "occurred_at"),
            ["latest_scoring"] = await LatestAsync("scoring_results", "decision_time")
        };
    }

    public async Task<int> CountEventsAsync(CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);
        return await connection.ExecuteScalarAsync<int>(new CommandDefinition(
            $"SELECT COUNT(*) FROM {_events};", cancellationToken: ct));
    }

    private async Task<Dictionary<string, object?>?> FindEventAsync(
        SqlConnection connection, SqlTransaction? transaction, string eventId,
        CancellationToken ct)
    {
        var row = await connection.QueryFirstOrDefaultAsync<dynamic>(
            new CommandDefinition(
                $"SELECT * FROM {_events} WHERE event_id = @event_id;",
                new { event_id = eventId }, transaction, cancellationToken: ct));
        return row is null ? null : MapRow((object)row);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// <summary>
    /// Dapper row -> plain dictionary, with DATETIME2 rendered as the same
    /// ISO-8601 "...Z" string the Python receiver emits.
    /// </summary>
    private static Dictionary<string, object?> MapRow(object row)
    {
        var source = (IDictionary<string, object>)row;
        var mapped = new Dictionary<string, object?>(source.Count);
        foreach (var (key, value) in source)
        {
            mapped[key] = value switch
            {
                null or DBNull => null,
                DateTime dt => FormatInstant(dt),
                _ => value
            };
        }
        return mapped;
    }

    public static string FormatInstant(DateTime value) =>
        DateTime.SpecifyKind(value, DateTimeKind.Utc)
            .ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture);

    /// <summary>ISO-8601 (with or without offset) -> UTC DateTime.</summary>
    public static DateTime? ParseInstant(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;
        if (DateTimeOffset.TryParse(value, CultureInfo.InvariantCulture,
                                    DateTimeStyles.AssumeUniversal, out var offset))
            return offset.UtcDateTime;
        if (DateTime.TryParse(value, CultureInfo.InvariantCulture,
                              DateTimeStyles.AdjustToUniversal |
                              DateTimeStyles.AssumeUniversal, out var plain))
            return plain;
        return null;
    }

    /// <summary>
    /// "KDVR:ChanNo", "site:KDVR:ChanNo", "KDVR:ChanNo:bay" or
    /// "site:KDVR:ChanNo:bay" -> (kdvr, chan_no). The numeric pair is located
    /// positionally, because a bay id is not numeric.
    /// </summary>
    public static (int? Kdvr, int? ChanNo) SplitEntity(string? entityId)
    {
        if (string.IsNullOrWhiteSpace(entityId))
            return (null, null);
        var parts = entityId.Split(':');
        string[] pair;
        if (parts.Length >= 4)
            pair = new[] { parts[1], parts[2] };
        else if (parts.Length == 3)
            pair = int.TryParse(parts[2], out _)
                ? new[] { parts[1], parts[2] }
                : new[] { parts[0], parts[1] };
        else if (parts.Length == 2)
            pair = parts;
        else
            return (null, null);
        return int.TryParse(pair[0], out var kdvr) && int.TryParse(pair[1], out var chan)
            ? (kdvr, chan)
            : (null, null);
    }

    private static JsonElement Obj(JsonElement parent, string name) =>
        parent.ValueKind == JsonValueKind.Object &&
        parent.TryGetProperty(name, out var child) &&
        child.ValueKind == JsonValueKind.Object
            ? child
            : default;

    /// <summary>Raw JSON text of a property, or null when absent/null.</summary>
    private static string? Raw(JsonElement parent, string name) =>
        parent.ValueKind == JsonValueKind.Object &&
        parent.TryGetProperty(name, out var child) &&
        child.ValueKind != JsonValueKind.Null &&
        child.ValueKind != JsonValueKind.Undefined
            ? child.GetRawText()
            : null;

    private static string? Str(JsonElement parent, string name)
    {
        if (parent.ValueKind != JsonValueKind.Object ||
            !parent.TryGetProperty(name, out var child))
            return null;
        return child.ValueKind switch
        {
            JsonValueKind.String => child.GetString(),
            JsonValueKind.Null or JsonValueKind.Undefined => null,
            _ => child.GetRawText()
        };
    }

    private static double? Num(JsonElement parent, string name) =>
        parent.ValueKind == JsonValueKind.Object &&
        parent.TryGetProperty(name, out var child) &&
        child.ValueKind == JsonValueKind.Number &&
        child.TryGetDouble(out var number)
            ? number
            : null;

    private static int? Int(JsonElement parent, string name) =>
        parent.ValueKind == JsonValueKind.Object &&
        parent.TryGetProperty(name, out var child) &&
        child.ValueKind == JsonValueKind.Number &&
        child.TryGetInt32(out var number)
            ? number
            : null;
}
