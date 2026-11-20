using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Options;

namespace CartPublisher;

/// <summary>Pooled connections to the sender's database.</summary>
public sealed class OutboxConnectionFactory
{
    private readonly PublisherOptions _options;

    public OutboxConnectionFactory(IOptions<PublisherOptions> options)
    {
        _options = options.Value;
        if (string.IsNullOrWhiteSpace(_options.ConnectionString))
        {
            throw new InvalidOperationException(
                "No outbox connection string. Set Publisher:ConnectionString or the " +
                "CART_OUTBOX_SQL_CONNECTION environment variable.");
        }
    }

    public string Schema => _options.Schema;

    /// <summary>Server/database, for diagnostics. Never includes credentials.</summary>
    public string DatabaseDescription
    {
        get
        {
            var builder = new SqlConnectionStringBuilder(_options.ConnectionString);
            return $"{builder.DataSource}/{builder.InitialCatalog}";
        }
    }

    public async Task<SqlConnection> OpenAsync(CancellationToken ct = default)
    {
        var connection = new SqlConnection(_options.ConnectionString);
        await connection.OpenAsync(ct);
        return connection;
    }
}
