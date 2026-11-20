using System.Text.Json.Serialization;

namespace CartPublisher;

/// <summary>Lifecycle states of an outbox row.</summary>
public static class OutboxStatus
{
    public const string Pending = "pending";
    public const string Sending = "sending";
    public const string Sent = "sent";
    public const string DeadLetter = "dead_letter";
}

/// <summary>One queued event. Column names match cart.result_outbox.</summary>
public sealed class OutboxRow
{
    [JsonPropertyName("id")] public long Id { get; set; }
    [JsonPropertyName("event_id")] public string EventId { get; set; } = string.Empty;
    [JsonPropertyName("event_type")] public string EventType { get; set; } = string.Empty;
    [JsonPropertyName("destination")] public string Destination { get; set; } = string.Empty;
    [JsonPropertyName("entity_id")] public string? EntityId { get; set; }
    [JsonPropertyName("job_id")] public string? JobId { get; set; }
    [JsonPropertyName("model_id")] public string? ModelId { get; set; }
    [JsonPropertyName("payload_json")] public string PayloadJson { get; set; } = "{}";
    [JsonPropertyName("status")] public string Status { get; set; } = OutboxStatus.Pending;
    [JsonPropertyName("attempts")] public int Attempts { get; set; }
    [JsonPropertyName("next_attempt_at")] public DateTime? NextAttemptAt { get; set; }
    [JsonPropertyName("claimed_at")] public DateTime? ClaimedAt { get; set; }
    [JsonPropertyName("last_error")] public string? LastError { get; set; }
    [JsonPropertyName("response_code")] public int? ResponseCode { get; set; }
    [JsonPropertyName("created_at")] public DateTime CreatedAt { get; set; }
    [JsonPropertyName("sent_at")] public DateTime? SentAt { get; set; }
}

/// <summary>Queue depth by status.</summary>
public sealed class OutboxStats
{
    [JsonPropertyName("pending")] public int Pending { get; set; }
    [JsonPropertyName("sending")] public int Sending { get; set; }
    [JsonPropertyName("sent")] public int Sent { get; set; }
    [JsonPropertyName("dead_letter")] public int DeadLetter { get; set; }
    [JsonPropertyName("total")] public int Total => Pending + Sending + Sent + DeadLetter;
}

/// <summary>Counts from one delivery batch.</summary>
public sealed class FlushResult
{
    [JsonPropertyName("skipped")] public bool Skipped { get; set; }
    [JsonPropertyName("sent")] public int Sent { get; set; }
    [JsonPropertyName("retried")] public int Retried { get; set; }
    [JsonPropertyName("dead")] public int Dead { get; set; }
}
