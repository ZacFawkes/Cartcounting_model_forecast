namespace ReceiverNet.Configuration;

/// <summary>
/// Receiver settings, bound from the "Receiver" configuration section.
/// Mirrors the RECEIVER_* values in cart_platform/settings.py.
/// </summary>
public sealed class ReceiverOptions
{
    public const string SectionName = "Receiver";

    /// <summary>
    /// SQL Server connection string. Prefer the CART_RECEIVER_SQL_CONNECTION
    /// environment variable over appsettings.json so no credential is committed.
    /// </summary>
    public string ConnectionString { get; set; } = string.Empty;

    /// <summary>Schema holding the receiver tables.</summary>
    public string Schema { get; set; } = "cart";

    /// <summary>
    /// Bearer token required on write routes. Empty disables authentication,
    /// matching the Python receiver's behaviour when CART_RESULT_RECEIVER_TOKEN
    /// is unset.
    /// </summary>
    public string? Token { get; set; }

    /// <summary>Run sql/02_create_tables.sql at startup. Off in production.</summary>
    public bool AutoMigrate { get; set; }

    /// <summary>Default page size for list routes.</summary>
    public int DefaultLimit { get; set; } = 50;

    /// <summary>Hard ceiling on page size.</summary>
    public int MaxLimit { get; set; } = 500;
}
