using System.Text.Json.Serialization;
using CartPublisher;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Options;

// ---------------------------------------------------------------------------
// The SENDER side: owns cart.result_outbox, delivers events to the receiver.
//
// Producers enqueue over HTTP (POST /v1/integrations/outbox) or, if they are
// .NET, by referencing CartPublisher and calling Outbox.EnqueueAsync directly
// inside the same transaction as the result they are recording.
// ---------------------------------------------------------------------------
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddCartPublisher(builder.Configuration);

builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.Never;
});

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new()
    {
        Title = "Cart Result Publisher (.NET)",
        Version = "1.0.0",
        Description = "Durable outbox and retry worker. Accepts result events, " +
                      "stores them in SQL Server, and delivers them to the Cart " +
                      "Result Receiver with exponential backoff."
    });
});

var app = builder.Build();

var options = app.Services.GetRequiredService<IOptions<PublisherOptions>>().Value;

// Fail fast: enqueuing into a missing table would silently lose results.
try
{
    using var scope = app.Services.CreateScope();
    await scope.ServiceProvider.GetRequiredService<Outbox>().EnsureReadyAsync();
}
catch (Exception ex)
{
    app.Logger.LogCritical(ex, "Publisher cannot start: {Message}", ex.Message);
    return 1;
}

app.UseSwagger();
app.UseSwaggerUI(o =>
{
    o.SwaggerEndpoint("/swagger/v1/swagger.json", "Cart Result Publisher v1");
    o.RoutePrefix = "docs";
});

app.Use(async (context, next) =>
{
    try
    {
        await next();
    }
    catch (SqlException ex)
    {
        app.Logger.LogError(ex, "SQL Server error handling {Path}", context.Request.Path);
        if (!context.Response.HasStarted)
        {
            context.Response.StatusCode = StatusCodes.Status503ServiceUnavailable;
            await context.Response.WriteAsJsonAsync(new
            {
                detail = "outbox unavailable",
                error = ex.Message
            });
        }
    }
});

app.MapGet("/health", async (Outbox outbox, OutboxRetryWorker worker,
                             OutboxConnectionFactory factory,
                             CancellationToken ct) =>
    {
        try
        {
            var counts = await outbox.StatsAsync(ct);
            return Results.Ok(new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["service"] = "cart-result-publisher",
                ["version"] = "1.0.0",
                ["database"] = factory.DatabaseDescription,
                ["receiver_url"] = options.ReceiverUrl,
                ["publishing_enabled"] = options.Enabled,
                ["worker_running"] = worker.Running,
                ["counts"] = counts
            });
        }
        catch (Exception ex)
        {
            return Results.Ok(new Dictionary<string, object?>
            {
                ["status"] = "degraded",
                ["service"] = "cart-result-publisher",
                ["database"] = factory.DatabaseDescription,
                ["worker_running"] = worker.Running,
                ["error"] = ex.Message
            });
        }
    })
    .WithName("PublisherHealth")
    .WithSummary("Publisher health and queue depth.");

app.MapOutboxEndpoints();

app.Logger.LogInformation(
    "Cart Result Publisher ready; outbox {Database} -> receiver {Receiver}",
    app.Services.GetRequiredService<OutboxConnectionFactory>().DatabaseDescription,
    options.ReceiverUrl);

app.Run();
return 0;
