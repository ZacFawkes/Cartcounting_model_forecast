using System.Text.RegularExpressions;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Options;
using ReceiverNet.Configuration;

namespace ReceiverNet.Data;

/// <summary>
/// Optional startup migration: executes sql/02_create_tables.sql, which is
/// idempotent. Equivalent to database.init_db() in the Python receiver.
/// Disabled unless Receiver:AutoMigrate is true, because production deploys
/// should run the script through SSMS with elevated rights.
/// </summary>
public sealed class DatabaseInitializer
{
    private static readonly Regex BatchSeparator =
        new(@"^\s*GO\s*(--.*)?$", RegexOptions.IgnoreCase | RegexOptions.Multiline);

    private readonly SqlConnectionFactory _factory;
    private readonly ReceiverOptions _options;
    private readonly ILogger<DatabaseInitializer> _logger;

    public DatabaseInitializer(SqlConnectionFactory factory,
                               IOptions<ReceiverOptions> options,
                               ILogger<DatabaseInitializer> logger)
    {
        _factory = factory;
        _options = options.Value;
        _logger = logger;
    }

    public async Task InitialiseAsync(CancellationToken ct = default)
    {
        await using var connection = await _factory.OpenAsync(ct);

        if (!_options.AutoMigrate)
        {
            await VerifyTablesAsync(connection, ct);
            _logger.LogInformation("Receiver database ready at {Database}",
                _factory.DatabaseDescription);
            return;
        }

        var scriptPath = Path.Combine(AppContext.BaseDirectory, "sql",
                                      "02_create_tables.sql");
        if (!File.Exists(scriptPath))
        {
            _logger.LogWarning("AutoMigrate is on but {Path} is missing; skipping",
                scriptPath);
            return;
        }

        var script = await File.ReadAllTextAsync(scriptPath, ct);
        foreach (var batch in BatchSeparator.Split(script))
        {
            if (string.IsNullOrWhiteSpace(batch))
                continue;
            await using var command = connection.CreateCommand();
            command.CommandText = batch;
            command.CommandTimeout = 120;
            await command.ExecuteNonQueryAsync(ct);
        }

        _logger.LogInformation("Applied 02_create_tables.sql to {Database}",
            _factory.DatabaseDescription);
        await VerifyTablesAsync(connection, ct);
    }

    private async Task VerifyTablesAsync(SqlConnection connection,
                                         CancellationToken ct)
    {
        string[] required =
        {
            "received_events", "training_results", "calibration_results",
            "backtest_results", "scoring_results"
        };

        await using var command = connection.CreateCommand();
        command.CommandText =
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = @schema";
        command.Parameters.AddWithValue("@schema", _factory.Schema);

        var present = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        await using (var reader = await command.ExecuteReaderAsync(ct))
        {
            while (await reader.ReadAsync(ct))
                present.Add(reader.GetString(0));
        }

        var missing = required.Where(t => !present.Contains(t)).ToArray();
        if (missing.Length > 0)
        {
            throw new InvalidOperationException(
                $"Schema [{_factory.Schema}] is missing: {string.Join(", ", missing)}. " +
                "Run sql/01_create_database.sql then sql/02_create_tables.sql.");
        }
    }
}
