namespace CartPublisher;

/// <summary>
/// Sender-side settings. Mirrors the CART_RESULT_PUBLISH_* values in
/// cart_platform/settings.py.
/// </summary>
public sealed class PublisherOptions
{
    public const string SectionName = "Publisher";

    /// <summary>SQL Server holding cart.result_outbox (the sender's database).</summary>
    public string ConnectionString { get; set; } = string.Empty;

    public string Schema { get; set; } = "cart";

    /// <summary>Receiver base URL. CART_RESULT_RECEIVER_URL.</summary>
    public string ReceiverUrl { get; set; } = "http://127.0.0.1:8200";

    /// <summary>Bearer token the receiver expects. CART_RESULT_RECEIVER_TOKEN.</summary>
    public string? Token { get; set; }

    /// <summary>CART_RESULT_RECEIVER_ENABLED. When false nothing is delivered.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>CART_RESULT_PUBLISH_TIMEOUT_SEC.</summary>
    public double TimeoutSeconds { get; set; } = 10;

    /// <summary>CART_RESULT_PUBLISH_MAX_ATTEMPTS, then the row is dead-lettered.</summary>
    public int MaxAttempts { get; set; } = 8;

    /// <summary>CART_RESULT_PUBLISH_BACKOFF_SEC - first retry delay.</summary>
    public double BaseBackoffSeconds { get; set; } = 2;

    /// <summary>CART_RESULT_PUBLISH_MAX_BACKOFF_SEC - backoff ceiling.</summary>
    public double MaxBackoffSeconds { get; set; } = 300;

    /// <summary>CART_RESULT_WORKER_INTERVAL_SEC - delay between flushes.</summary>
    public double WorkerIntervalSeconds { get; set; } = 5;

    /// <summary>Rows delivered per flush.</summary>
    public int BatchSize { get; set; } = 20;

    /// <summary>A claim older than this is treated as abandoned by a dead worker.</summary>
    public double StuckAfterSeconds { get; set; } = 120;

    /// <summary>Run the background delivery loop in this process.</summary>
    public bool RunWorker { get; set; } = true;
}
