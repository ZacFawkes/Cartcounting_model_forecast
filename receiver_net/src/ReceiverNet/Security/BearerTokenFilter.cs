using Microsoft.Extensions.Options;
using ReceiverNet.Configuration;

namespace ReceiverNet.Security;

/// <summary>
/// Bearer-token check applied to write routes only.
/// Port of require_token() in cart_platform/receiver/app.py: when no token is
/// configured the check is a no-op, exactly as the Python receiver behaves
/// when CART_RESULT_RECEIVER_TOKEN is unset.
/// </summary>
public sealed class BearerTokenFilter : IEndpointFilter
{
    private readonly string? _expected;

    public BearerTokenFilter(IOptions<ReceiverOptions> options)
    {
        var token = options.Value.Token;
        _expected = string.IsNullOrWhiteSpace(token) ? null : $"Bearer {token}";
    }

    public ValueTask<object?> InvokeAsync(EndpointFilterInvocationContext context,
                                          EndpointFilterDelegate next)
    {
        if (_expected is null)
            return next(context);

        var header = context.HttpContext.Request.Headers.Authorization.ToString();
        if (!CryptographicEquals(header, _expected))
        {
            return ValueTask.FromResult<object?>(Results.Problem(
                detail: "invalid or missing bearer token",
                statusCode: StatusCodes.Status401Unauthorized));
        }

        return next(context);
    }

    /// <summary>Length-independent, constant-time comparison.</summary>
    private static bool CryptographicEquals(string left, string right)
    {
        var difference = (uint)left.Length ^ (uint)right.Length;
        for (var i = 0; i < left.Length && i < right.Length; i++)
            difference |= (uint)(left[i] ^ right[i]);
        return difference == 0;
    }
}
