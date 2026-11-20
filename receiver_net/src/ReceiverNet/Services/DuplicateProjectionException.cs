namespace ReceiverNet.Services;

/// <summary>
/// A new event_id collided with a uniqueness constraint on a projection table
/// rather than on received_events - in practice ux_scoring_decision, which
/// allows only one decision per (entity_id, decision_time, model_id).
///
/// Surfaced as HTTP 409, not 5xx: the ML server's outbox retries 5xx, and no
/// number of retries can resolve this.
/// </summary>
public sealed class DuplicateProjectionException : Exception
{
    public DuplicateProjectionException(string message, Exception inner)
        : base(message, inner)
    {
    }
}
