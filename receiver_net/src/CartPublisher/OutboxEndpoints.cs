using System.Text.Json;
using System.Text.Json.Serialization;
using CartEvents;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace CartPublisher;

/// <summary>Request body for POST /v1/integrations/outbox/retry.</summary>
public sealed class OutboxRetryRequest
{
    /// <summary>Requeue one event; omit to requeue every dead letter.</summary>
    [JsonPropertyName("event_id")]
    public string? EventId { get; set; }
}

/// <summary>
/// Outbox observability and control.
/// Port of the /v1/integrations/outbox routes in cart_platform/server/ondemand.py,
/// plus an enqueue route so a producer in any language can hand events over.
/// </summary>
public static class OutboxEndpoints
{
    public static IEndpointRouteBuilder MapOutboxEndpoints(this IEndpointRouteBuilder app)
    {
        var group = app.MapGroup("/v1/integrations/outbox");

        group.MapGet("/", async (string? status, string? entity_id, string? event_type,
                                 int? limit, Outbox outbox, OutboxRetryWorker worker,
                                 CancellationToken ct) =>
            {
                var events = await outbox.ListAsync(status, entity_id, event_type,
                                                    limit ?? 100, ct);
                return Results.Ok(new Dictionary<string, object?>
                {
                    ["worker_running"] = worker.Running,
                    ["counts"] = await outbox.StatsAsync(ct),
                    ["events"] = events
                });
            })
            .WithName("OutboxStatus")
            .WithSummary("Queue depth, worker state and the most recent events.");

        group.MapPost("/", async (JsonElement body, Outbox outbox,
                                  CancellationToken ct) =>
            {
                ResultEvent? e;
                try
                {
                    e = body.Deserialize<ResultEvent>();
                }
                catch (JsonException ex)
                {
                    return Results.Problem(detail: $"malformed event: {ex.Message}",
                        statusCode: StatusCodes.Status422UnprocessableEntity);
                }

                if (e is null)
                    return Results.Problem(detail: "event body is required",
                        statusCode: StatusCodes.Status422UnprocessableEntity);

                var invalid = e.Validate();
                if (invalid is not null)
                    return Results.Problem(detail: invalid,
                        statusCode: StatusCodes.Status422UnprocessableEntity);

                var queued = await outbox.EnqueueAsync(e, ct: ct);

                // 202: accepted for delivery, not yet delivered. 200 means the
                // event_id was already queued, so this call changed nothing.
                return Results.Json(new
                {
                    event_id = e.EventId,
                    outcome = queued ? "queued" : "duplicate_suppressed"
                }, statusCode: queued ? StatusCodes.Status202Accepted
                                      : StatusCodes.Status200OK);
            })
            .WithName("OutboxEnqueue")
            .WithSummary("Queue a result event for durable delivery to the receiver.");

        group.MapPost("/retry", async (OutboxRetryRequest request, Outbox outbox,
                                       ResultPublisher publisher,
                                       CancellationToken ct) =>
            {
                var requeued = await outbox.RetryDeadLettersAsync(request.EventId, ct);
                var flushed = await publisher.FlushOnceAsync(ct: ct);
                return Results.Ok(new Dictionary<string, object?>
                {
                    ["requeued"] = requeued,
                    ["flush"] = flushed,
                    ["counts"] = await outbox.StatsAsync(ct)
                });
            })
            .WithName("OutboxRetry")
            .WithSummary("Requeue dead letters and flush immediately.");

        group.MapPost("/flush", async (int? limit, ResultPublisher publisher,
                                       CancellationToken ct) =>
                Results.Ok(await publisher.FlushOnceAsync(limit, ct)))
            .WithName("OutboxFlush")
            .WithSummary("Deliver one batch now, without waiting for the worker.");

        // Catch-all last: event_id contains colons and slashes.
        group.MapGet("/{**eventId}", async (string eventId, Outbox outbox,
                                            CancellationToken ct) =>
            {
                var row = await outbox.GetAsync(eventId, ct);
                return row is null
                    ? Results.Problem(detail: $"event '{eventId}' not found",
                                      statusCode: StatusCodes.Status404NotFound)
                    : Results.Ok(row);
            })
            .WithName("OutboxEvent")
            .WithSummary("One queued event, with its delivery history.");

        return app;
    }
}
