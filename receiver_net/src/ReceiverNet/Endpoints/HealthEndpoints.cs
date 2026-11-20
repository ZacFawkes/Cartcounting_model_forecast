using System.Diagnostics;
using ReceiverNet.Data;
using ReceiverNet.Services;

namespace ReceiverNet.Endpoints;

/// <summary>Liveness probe. Port of GET /health in the Python receiver.</summary>
public static class HealthEndpoints
{
    private static readonly Stopwatch Uptime = Stopwatch.StartNew();

    public static IEndpointRouteBuilder MapHealthEndpoint(this IEndpointRouteBuilder app)
    {
        app.MapGet("/health", async (EventStore store, SqlConnectionFactory factory,
                                     CancellationToken ct) =>
            {
                int events;
                bool ok;
                string? error;
                try
                {
                    events = await store.CountEventsAsync(ct);
                    ok = true;
                    error = null;
                }
                catch (Exception ex)
                {
                    events = 0;
                    ok = false;
                    error = ex.Message;
                }

                return Results.Ok(new Dictionary<string, object?>
                {
                    ["status"] = ok ? "ok" : "degraded",
                    ["service"] = "cart-result-receiver",
                    ["version"] = "1.0.0",
                    ["database"] = factory.DatabaseDescription,
                    ["sql_server_ok"] = ok,
                    ["error"] = error,
                    ["uptime_seconds"] = Math.Round(Uptime.Elapsed.TotalSeconds, 1),
                    ["events_stored"] = events
                });
            })
            .WithName("Health")
            .WithSummary("Service and database health.");

        return app;
    }
}
