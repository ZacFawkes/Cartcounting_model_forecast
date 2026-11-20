using ReceiverNet.Configuration;
using ReceiverNet.Models;
using ReceiverNet.Security;
using ReceiverNet.Services;

namespace ReceiverNet.Endpoints;

/// <summary>
/// Route table. Wire-compatible with cart_platform/receiver/app.py so the ML
/// server can point CART_RESULT_RECEIVER_URL at this service unchanged.
/// </summary>
public static class ResultsEndpoints
{
    public static IEndpointRouteBuilder MapResultEndpoints(
        this IEndpointRouteBuilder app, ReceiverOptions options)
    {
        MapWrites(app);
        MapReads(app, options);
        return app;
    }

    // ------------------------------------------------------------------
    // Ingestion. Append-only: an event is created once and never updated.
    // ------------------------------------------------------------------
    private static void MapWrites(IEndpointRouteBuilder app)
    {
        var writes = app.MapGroup("/v1/results").AddEndpointFilter<BearerTokenFilter>();

        writes.MapPost("/training",
                (ResultEventIn e, EventStore store, CancellationToken ct) =>
                    IngestAsync(e, "training.completed", store, ct))
            .WithName("PostTraining")
            .WithSummary("Record a completed training run.");

        writes.MapPost("/calibration",
                (ResultEventIn e, EventStore store, CancellationToken ct) =>
                    IngestAsync(e, "calibration.completed", store, ct))
            .WithName("PostCalibration")
            .WithSummary("Record a completed calibration run.");

        writes.MapPost("/backtest",
                (ResultEventIn e, EventStore store, CancellationToken ct) =>
                {
                    var backtestType = ReadBacktestType(e);
                    return string.IsNullOrWhiteSpace(backtestType)
                        ? Task.FromResult(Unprocessable("payload.backtest_type is required"))
                        : IngestAsync(e, "backtest.completed", store, ct);
                })
            .WithName("PostBacktest")
            .WithSummary("Record a completed backtest.");

        writes.MapPost("/scoring",
                (ResultEventIn e, EventStore store, CancellationToken ct) =>
                    IngestAsync(e, "scoring.recorded", store, ct))
            .WithName("PostScoring")
            .WithSummary("Record a scoring decision.");
    }

    private static async Task<IResult> IngestAsync(ResultEventIn e, string expectedType,
                                                   EventStore store, CancellationToken ct)
    {
        var invalid = e.Validate();
        if (invalid is not null)
            return Unprocessable(invalid);

        if (!string.Equals(e.EventType, expectedType, StringComparison.Ordinal))
        {
            return Unprocessable(
                $"event_type '{e.EventType}' does not match this route " +
                $"(expected '{expectedType}')");
        }

        var (outcome, record) = (StoreOutcome.Created, new Dictionary<string, object?>());
        try
        {
            (outcome, record) = await store.StoreEventAsync(e, ct);
        }
        catch (DuplicateProjectionException ex)
        {
            return Results.Json(new
            {
                detail = new
                {
                    message = ex.Message,
                    event_id = e.EventId
                }
            }, statusCode: StatusCodes.Status409Conflict);
        }

        if (outcome == StoreOutcome.Conflict)
        {
            return Results.Json(new
            {
                detail = new
                {
                    message = "event_id already stored with a different payload; " +
                              "historical results are never overwritten",
                    event_id = e.EventId,
                    stored_payload_hash = record.GetValueOrDefault("payload_hash")
                }
            }, statusCode: StatusCodes.Status409Conflict);
        }

        var ack = new StoreAck
        {
            EventId = e.EventId,
            Outcome = outcome == StoreOutcome.Created ? "created" : "replay",
            StoredRecord = Summarise(record)
        };

        return Results.Json(ack, statusCode: outcome == StoreOutcome.Created
            ? StatusCodes.Status201Created
            : StatusCodes.Status200OK);
    }

    /// <summary>The subset of the stored row echoed back in the ack.</summary>
    private static Dictionary<string, object?> Summarise(Dictionary<string, object?> row)
    {
        string[] keys =
        {
            "event_id", "event_type", "entity_id", "job_id", "model_id", "received_at"
        };
        return keys.Where(row.ContainsKey).ToDictionary(k => k, k => row[k]);
    }

    private static string? ReadBacktestType(ResultEventIn e)
    {
        if (e.Payload.ValueKind != System.Text.Json.JsonValueKind.Object ||
            !e.Payload.TryGetProperty("backtest_type", out var value))
            return null;
        return value.ValueKind == System.Text.Json.JsonValueKind.String
            ? value.GetString()
            : null;
    }

    private static IResult Unprocessable(string detail) =>
        Results.Problem(detail: detail,
                        statusCode: StatusCodes.Status422UnprocessableEntity);

    // ------------------------------------------------------------------
    // Retrieval
    // ------------------------------------------------------------------
    private static void MapReads(IEndpointRouteBuilder app, ReceiverOptions options)
    {
        var reads = app.MapGroup("/v1/results");

        reads.MapGet("/training",
                ([AsParameters] ResultQuery q, EventStore store, CancellationToken ct) =>
                    PageAsync(q, options, (l, o) => store.ListTrainingAsync(q, l, o, ct)))
            .WithName("GetTraining")
            .WithSummary("List training results.");

        reads.MapGet("/calibration",
                ([AsParameters] ResultQuery q, EventStore store, CancellationToken ct) =>
                    PageAsync(q, options, (l, o) => store.ListCalibrationAsync(q, l, o, ct)))
            .WithName("GetCalibration")
            .WithSummary("List calibration results.");

        reads.MapGet("/backtest",
                ([AsParameters] ResultQuery q, EventStore store, CancellationToken ct) =>
                    PageAsync(q, options, (l, o) => store.ListBacktestsAsync(q, l, o, ct)))
            .WithName("GetBacktest")
            .WithSummary("List backtest results.");

        reads.MapGet("/scoring",
                ([AsParameters] ResultQuery q, EventStore store, CancellationToken ct) =>
                    PageAsync(q, options, (l, o) => store.ListScoringAsync(q, l, o, ct)))
            .WithName("GetScoring")
            .WithSummary("List scoring results, newest decision first.");

        reads.MapGet("/entities/{**entityId}",
                async (string entityId, EventStore store, CancellationToken ct) =>
                    Results.Ok(await store.EntitySummaryAsync(entityId, ct)))
            .WithName("GetEntityResults")
            .WithSummary("Everything the receiver holds for one entity.");

        reads.MapGet("/events/{**eventId}",
                async (string eventId, EventStore store, CancellationToken ct) =>
                {
                    var row = await store.GetEventAsync(eventId, ct);
                    return row is null
                        ? Results.Problem(detail: $"event '{eventId}' not found",
                                          statusCode: StatusCodes.Status404NotFound)
                        : Results.Ok(row);
                })
            .WithName("GetEvent")
            .WithSummary("Fetch a single stored event with its raw payload.");
    }

    private static async Task<IResult> PageAsync(ResultQuery q, ReceiverOptions options,
                                                 Func<int, int, Task<PageOut>> fetch)
    {
        var limit = q.Limit ?? options.DefaultLimit;
        var offset = q.Offset ?? 0;

        if (limit < 1 || limit > options.MaxLimit)
            return Unprocessable($"limit must be between 1 and {options.MaxLimit}");
        if (offset < 0)
            return Unprocessable("offset must be >= 0");

        return Results.Ok(await fetch(limit, offset));
    }
}
