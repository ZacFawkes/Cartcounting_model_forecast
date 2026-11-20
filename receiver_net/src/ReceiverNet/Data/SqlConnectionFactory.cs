using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Options;
using ReceiverNet.Configuration;

namespace ReceiverNet.Data;

/// <summary>
/// Hands out pooled SQL Server connections. Replaces the thread-local SQLite
/// handle in cart_platform/receiver/database.py - ADO.NET pools for us.
/// </summary>
public sealed class SqlConnectionFactory
{
    private readonly ReceiverOptions _options;

    public SqlConnectionFactory(IOptions<ReceiverOptions> options)
    {
        _options = options.Value;
        if (string.IsNullOrWhiteSpace(_options.ConnectionString))
        {
            throw new InvalidOperationException(
                "No connection string. Set Receiver:ConnectionString or the " +
                "CART_RECEIVER_SQL_CONNECTION environment variable.");
        }
    }

    public string Schema => _options.Schema;

    /// <summary>Database name, for /health. Never includes credentials.</summary>
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
