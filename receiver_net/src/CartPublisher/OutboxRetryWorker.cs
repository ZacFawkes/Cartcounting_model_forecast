using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CartPublisher;

/// <summary>
/// Background delivery loop, independent of API response latency.
/// Port of the RetryWorker thread in publisher.py.
///
/// The loop must never die: one bad iteration is logged and the next one runs.
/// </summary>
public sealed class OutboxRetryWorker : BackgroundService
{
    private readonly IServiceProvider _services;
    private readonly PublisherOptions _options;
    private readonly ILogger<OutboxRetryWorker> _logger;

    public OutboxRetryWorker(IServiceProvider services,
                             IOptions<PublisherOptions> options,
                             ILogger<OutboxRetryWorker> logger)
    {
        _services = services;
        _options = options.Value;
        _logger = logger;
    }

    /// <summary>True once the loop is running; surfaced by the admin API.</summary>
    public bool Running { get; private set; }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!_options.Enabled)
        {
            _logger.LogInformation("Result publishing disabled; retry worker not started");
            return;
        }
        if (!_options.RunWorker)
        {
            _logger.LogInformation("Publisher:RunWorker is false; this process only enqueues");
            return;
        }

        Running = true;
        var interval = TimeSpan.FromSeconds(_options.WorkerIntervalSeconds);
        _logger.LogInformation("Outbox retry worker started (target {Url}, every {Interval}s)",
            _options.ReceiverUrl, _options.WorkerIntervalSeconds);

        // Rows left "sending" by a previous run are ours to reclaim. If the
        // database is briefly unavailable at startup we retry on a later
        // iteration rather than giving up.
        var resetPending = true;

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                using var scope = _services.CreateScope();
                var outbox = scope.ServiceProvider.GetRequiredService<Outbox>();
                var publisher = scope.ServiceProvider.GetRequiredService<ResultPublisher>();

                if (resetPending)
                {
                    await outbox.ResetSendingAsync(stoppingToken);
                    resetPending = false;
                }

                await outbox.RecoverStuckAsync(stoppingToken);
                await publisher.FlushOnceAsync(ct: stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                // The loop must never die: a database outage or a bad row is
                // logged and retried on the next tick. Letting this escape
                // would stop the whole host under the default
                // BackgroundServiceExceptionBehavior.StopHost.
                _logger.LogError(ex, "Outbox worker iteration failed");
            }

            try
            {
                await Task.Delay(interval, stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        Running = false;
        _logger.LogInformation("Outbox retry worker stopped");
    }
}
