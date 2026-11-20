using System.Text.Json.Serialization;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Options;
using ReceiverNet.Configuration;
using ReceiverNet.Data;
using ReceiverNet.Endpoints;
using ReceiverNet.Services;

var builder = WebApplication.CreateBuilder(args);

// --------------------------------------------------------------------------
// Configuration. Environment variables win, so no secret needs to be committed:
//   CART_RECEIVER_SQL_CONNECTION  -> Receiver:ConnectionString
//   CART_RESULT_RECEIVER_TOKEN    -> Receiver:Token
// --------------------------------------------------------------------------
builder.Services.AddOptions<ReceiverOptions>()
    .Bind(builder.Configuration.GetSection(ReceiverOptions.SectionName))
    .PostConfigure(options =>
    {
        var connection = builder.Configuration["CART_RECEIVER_SQL_CONNECTION"];
        if (!string.IsNullOrWhiteSpace(connection))
            options.ConnectionString = connection;

        var token = builder.Configuration["CART_RESULT_RECEIVER_TOKEN"];
        if (!string.IsNullOrWhiteSpace(token))
            options.Token = token;
    });

builder.Services.AddSingleton<SqlConnectionFactory>();
builder.Services.AddSingleton<DatabaseInitializer>();
builder.Services.AddSingleton<EventStore>();

builder.Services.ConfigureHttpJsonOptions(options =>
{
    // Property names are already snake_case on the wire via [JsonPropertyName];
    // null members are kept so responses match the Python receiver exactly.
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.Never;
});

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new()
    {
        Title = "Cart Result Receiver (.NET)",
        Version = "1.0.0",
        Description = "SQL Server-backed result store. Receives training, " +
                      "calibration, backtest and scoring events from the ML " +
                      "platform over HTTP. Append-only: results are never updated."
    });
});

var app = builder.Build();

var receiverOptions = app.Services.GetRequiredService<IOptions<ReceiverOptions>>().Value;

// --------------------------------------------------------------------------
// Startup: verify (or create) the schema before accepting traffic.
// Fail fast and loudly - serving traffic without a usable store would turn
// every delivery into a 500 and stall the ML server's outbox.
// --------------------------------------------------------------------------
try
{
    await app.Services.GetRequiredService<DatabaseInitializer>().InitialiseAsync();
}
catch (Exception ex)
{
    app.Logger.LogCritical(ex, "Receiver cannot start: {Message}", ex.Message);
    return 1;
}

app.UseSwagger();
app.UseSwaggerUI(options =>
{
    options.SwaggerEndpoint("/swagger/v1/swagger.json", "Cart Result Receiver v1");
    options.RoutePrefix = "docs";
});

// SQL errors surface as 503 rather than a bare 500: the ML server's outbox
// retries on 5xx, so a transient database outage is recoverable.
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
                detail = "result store unavailable",
                error = ex.Message
            });
        }
    }
});

app.MapHealthEndpoint();
app.MapResultEndpoints(receiverOptions);

app.Logger.LogInformation("Cart Result Receiver (.NET) ready; database {Database}",
    app.Services.GetRequiredService<SqlConnectionFactory>().DatabaseDescription);

app.Run();
return 0;
