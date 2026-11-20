using System.Text.Json;
using System.Text.Json.Serialization;

namespace CartEvents;

/// <summary>
/// The four event types and the receiver route each one is delivered to.
/// Port of ROUTES in cart_platform/integrations/result_events.py.
/// </summary>
public static class EventRoutes
{
    public const string TrainingCompleted = "training.completed";
    public const string CalibrationCompleted = "calibration.completed";
    public const string BacktestCompleted = "backtest.completed";
    public const string ScoringRecorded = "scoring.recorded";

    private static readonly Dictionary<string, string> Map = new(StringComparer.Ordinal)
    {
        [TrainingCompleted] = "/v1/results/training",
        [CalibrationCompleted] = "/v1/results/calibration",
        [BacktestCompleted] = "/v1/results/backtest",
        [ScoringRecorded] = "/v1/results/scoring"
    };

    public static IReadOnlyCollection<string> KnownTypes => Map.Keys;

    /// <summary>Receiver path for an event type, or null when unknown.</summary>
    public static string? For(string eventType) =>
        Map.TryGetValue(eventType, out var route) ? route : null;
}

/// <summary>
/// Versioned result-event envelope shared by the ML server and the receiver.
/// Port of the ResultEvent dataclass; the JSON names match the wire contract.
/// </summary>
public sealed class ResultEvent
{
    public const int CurrentSchemaVersion = 1;

    [JsonPropertyName("event_id")]
    public string EventId { get; set; } = string.Empty;

    [JsonPropertyName("event_type")]
    public string EventType { get; set; } = string.Empty;

    [JsonPropertyName("payload")]
    public JsonElement Payload { get; set; }

    [JsonPropertyName("entity_id")]
    public string? EntityId { get; set; }

    [JsonPropertyName("site_key")]
    public string? SiteKey { get; set; }

    [JsonPropertyName("upload_id")]
    public string? UploadId { get; set; }

    [JsonPropertyName("job_id")]
    public string? JobId { get; set; }

    [JsonPropertyName("model_id")]
    public string? ModelId { get; set; }

    [JsonPropertyName("schema_version")]
    public int SchemaVersion { get; set; } = CurrentSchemaVersion;

    [JsonPropertyName("occurred_at")]
    public string OccurredAt { get; set; } = UtcIso();

    public static string UtcIso() =>
        DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ",
                                 System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>Validation mirroring the receiver's own contract checks.</summary>
    public string? Validate()
    {
        if (string.IsNullOrWhiteSpace(EventId))
            return "event_id is required";
        if (EventId.Length > 450)
            return "event_id exceeds 450 characters, the storage key limit";
        if (EventRoutes.For(EventType) is null)
            return $"unknown event_type '{EventType}'";
        if (Payload.ValueKind != JsonValueKind.Object)
            return "payload must be a JSON object";
        return null;
    }
}

/// <summary>
/// Deterministic event_id builders - the idempotency keys.
/// Port of the *_event_id helpers in result_events.py; the formats must match
/// exactly or a resend would be stored as a new event instead of a replay.
/// </summary>
public static class EventIds
{
    private static string Slug(object? value) =>
        value is null ? "none" : value.ToString()!.Replace(" ", "_");

    private const int V = ResultEvent.CurrentSchemaVersion;

    public static string Training(object? jobId, object? entityId, object? status) =>
        $"{Slug(jobId)}:{Slug(entityId)}:training:{Slug(status)}:v{V}";

    public static string Calibration(object? jobId, object? entityId,
                                     object? calibrationId) =>
        $"{Slug(jobId)}:{Slug(entityId)}:calibration:{Slug(calibrationId)}:v{V}";

    public static string Backtest(object? jobId, object? entityId,
                                  object? backtestType, object? status) =>
        $"{Slug(jobId)}:{Slug(entityId)}:{Slug(backtestType)}:{Slug(status)}:v{V}";

    public static string Scoring(object? entityId, object? decisionTime,
                                 object? modelId) =>
        $"score:{Slug(entityId)}:{Slug(decisionTime)}:{Slug(modelId)}:v{V}";
}
