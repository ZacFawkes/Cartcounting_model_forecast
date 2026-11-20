using System.Net;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CartPublisher;

/// <summary>Result of one delivery attempt. Never thrown, always returned.</summary>
public sealed record DeliveryOutcome(bool Ok, int? StatusCode = null,
                                     string? Error = null, bool Retryable = false,
                                     double? RetryAfterSeconds = null);

/// <summary>
/// Delivers outbox rows to the receiver over HTTP.
/// Port of cart_platform/integrations/publisher.py.
/// </summary>
public sealed class ResultPublisher
{
    /// <summary>Transient conditions worth another attempt.</summary>
    private static readonly HashSet<int> RetryableStatus = new()
    {
        408, 425, 429, 500, 502, 503, 504
    };

    private readonly HttpClient _http;
    private readonly Outbox _outbox;
    private readonly PublisherOptions _options;
    private readonly ILogger<ResultPublisher> _logger;

    public ResultPublisher(HttpClient http, Outbox outbox,
                           IOptions<PublisherOptions> options,
                           ILogger<ResultPublisher> logger)
    {
        _http = http;
        _outbox = outbox;
        _options = options.Value;
        _logger = logger;
    }

    /// <summary>POST one row to the receiver. Never raises.</summary>
    public async Task<DeliveryOutcome> DeliverAsync(OutboxRow row,
                                                    CancellationToken ct = default)
    {
        var route = CartEvents.EventRoutes.For(row.EventType);
        if (route is null)
            return new DeliveryOutcome(false, null, $"unknown event_type '{row.EventType}'");

        var url = $"{row.Destination.TrimEnd('/')}{route}";

        using var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = new StringContent(row.PayloadJson, Encoding.UTF8, "application/json")
        };
        if (!string.IsNullOrWhiteSpace(_options.Token))
            request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {_options.Token}");

        HttpResponseMessage response;
        try
        {
            response = await _http.SendAsync(request, ct);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException
                                      && !ct.IsCancellationRequested)
        {
            return new DeliveryOutcome(false, null, $"network error: {ex.Message}",
                                       Retryable: true);
        }

        using (response)
        {
            var status = (int)response.StatusCode;

            if (status is 200 or 201)
                return new DeliveryOutcome(true, status);

            var body = await SafeReadAsync(response, ct);

            // Same event_id with a different payload: a real contract violation,
            // and the receiver will never accept it. Retrying is pointless.
            if (status == (int)HttpStatusCode.Conflict)
            {
                return new DeliveryOutcome(false, status,
                    $"receiver reported a conflicting payload: {body}");
            }

            if (RetryableStatus.Contains(status))
            {
                double? retryAfter = null;
                var header = response.Headers.RetryAfter;
                if (header?.Delta is not null)
                    retryAfter = header.Delta.Value.TotalSeconds;
                else if (header?.Date is not null)
                    retryAfter = Math.Max(0, (header.Date.Value - DateTimeOffset.UtcNow)
                                             .TotalSeconds);

                return new DeliveryOutcome(false, status,
                    $"receiver returned {status}: {body}",
                    Retryable: true, RetryAfterSeconds: retryAfter);
            }

            return new DeliveryOutcome(false, status, $"receiver returned {status}: {body}");
        }
    }

    private static async Task<string> SafeReadAsync(HttpResponseMessage response,
                                                    CancellationToken ct)
    {
        try
        {
            var body = await response.Content.ReadAsStringAsync(ct);
            return body.Length > 300 ? body[..300] : body;
        }
        catch
        {
            return "<unreadable response body>";
        }
    }

    /// <summary>Deliver one batch. Used by the worker and by the admin API.</summary>
    public async Task<FlushResult> FlushOnceAsync(int? limit = null,
                                                  CancellationToken ct = default)
    {
        if (!_options.Enabled)
            return new FlushResult { Skipped = true };

        var result = new FlushResult();
        foreach (var row in await _outbox.ClaimBatchAsync(limit, ct))
        {
            var outcome = await DeliverAsync(row, ct);
            if (outcome.Ok)
            {
                await _outbox.MarkSentAsync(row.Id, outcome.StatusCode, ct);
                result.Sent++;
            }
            else if (outcome.Retryable)
            {
                await _outbox.MarkRetryAsync(row.Id, row.Attempts,
                    outcome.Error ?? "unknown error", outcome.StatusCode,
                    outcome.RetryAfterSeconds, ct);
                result.Retried++;
            }
            else
            {
                await _outbox.MarkDeadAsync(row.Id, row.Attempts,
                    outcome.Error ?? "unknown error", outcome.StatusCode ?? 0, ct);
                result.Dead++;
            }
        }

        if (result.Sent > 0 || result.Retried > 0 || result.Dead > 0)
        {
            _logger.LogInformation(
                "Outbox flush: {Sent} sent, {Retried} retried, {Dead} dead-lettered",
                result.Sent, result.Retried, result.Dead);
        }

        return result;
    }
}
