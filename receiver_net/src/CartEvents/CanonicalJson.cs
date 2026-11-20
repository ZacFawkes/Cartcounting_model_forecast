using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace CartEvents;

/// <summary>
/// Canonical JSON serialisation and SHA-256 digests, used as the idempotency
/// fingerprint of an event payload.
///
/// Reproduces Python's
/// <c>json.dumps(payload, sort_keys=True, separators=(",", ":"))</c>:
/// keys sorted by ordinal (code point), no whitespace, and non-ASCII escaped
/// as \uXXXX. A payload resent from the Python ML server therefore hashes to
/// the same digest here, so an existing results_receiver.db can be migrated to
/// SQL Server without turning replays into 409 conflicts.
///
/// Caveat: exotic float literals (for example 1e+30) can repr differently in
/// Python and .NET. Payloads in this system carry plain decimals, so this does
/// not arise in practice.
/// </summary>
public static class CanonicalJson
{
    public static string Serialize(JsonElement value)
    {
        var builder = new StringBuilder();
        Write(value, builder);
        return builder.ToString();
    }

    public static string Hash(JsonElement value)
    {
        var bytes = Encoding.UTF8.GetBytes(Serialize(value));
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static void Write(JsonElement value, StringBuilder output)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                output.Append('{');
                var first = true;
                var properties = value.EnumerateObject()
                    .OrderBy(p => p.Name, StringComparer.Ordinal);
                foreach (var property in properties)
                {
                    if (!first) output.Append(',');
                    first = false;
                    WriteString(property.Name, output);
                    output.Append(':');
                    Write(property.Value, output);
                }
                output.Append('}');
                break;

            case JsonValueKind.Array:
                output.Append('[');
                var firstItem = true;
                foreach (var item in value.EnumerateArray())
                {
                    if (!firstItem) output.Append(',');
                    firstItem = false;
                    Write(item, output);
                }
                output.Append(']');
                break;

            case JsonValueKind.String:
                WriteString(value.GetString() ?? string.Empty, output);
                break;

            case JsonValueKind.Number:
                output.Append(FormatNumber(value.GetRawText()));
                break;

            case JsonValueKind.True:
                output.Append("true");
                break;

            case JsonValueKind.False:
                output.Append("false");
                break;

            default:
                output.Append("null");
                break;
        }
    }

    /// <summary>
    /// Integers are emitted verbatim; anything with a fraction or exponent is
    /// re-rendered the way Python's repr() renders a float, so the digest of a
    /// payload matches byte for byte across the two implementations.
    /// </summary>
    private static string FormatNumber(string raw)
    {
        if (raw.IndexOfAny(new[] { '.', 'e', 'E' }) < 0)
            return raw;
        return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture,
                               out var parsed)
            ? FormatDouble(parsed)
            : raw;
    }

    /// <summary>
    /// Shortest round-trip digits, laid out by CPython's float_repr rules:
    /// scientific notation when the decimal point sits beyond digit 16 or at or
    /// before position -4, otherwise fixed notation with a trailing ".0" when
    /// there is no fractional part.
    /// </summary>
    internal static string FormatDouble(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            return value.ToString("R", CultureInfo.InvariantCulture);

        var shortest = value.ToString("R", CultureInfo.InvariantCulture);

        var negative = shortest.StartsWith('-');
        if (negative)
            shortest = shortest[1..];

        var exponent = 0;
        var exponentAt = shortest.IndexOfAny(new[] { 'e', 'E' });
        if (exponentAt >= 0)
        {
            exponent = int.Parse(shortest[(exponentAt + 1)..], CultureInfo.InvariantCulture);
            shortest = shortest[..exponentAt];
        }

        var pointAt = shortest.IndexOf('.');
        var digits = pointAt >= 0 ? shortest.Remove(pointAt, 1) : shortest;

        // decpt: how many digits sit to the left of the decimal point.
        var decpt = (pointAt >= 0 ? pointAt : shortest.Length) + exponent;

        var lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0')
        {
            lead++;
            decpt--;
        }
        digits = digits[lead..];
        digits = digits.TrimEnd('0');

        var sign = negative ? "-" : string.Empty;
        if (digits.Length == 0)
            return sign + "0.0";

        if (decpt > 16 || decpt <= -4)
        {
            var mantissa = digits.Length > 1
                ? $"{digits[0]}.{digits[1..]}"
                : digits[..1];
            var power = decpt - 1;
            var powerSign = power < 0 ? "-" : "+";
            return $"{sign}{mantissa}e{powerSign}{Math.Abs(power):00}";
        }

        if (decpt <= 0)
            return $"{sign}0.{new string('0', -decpt)}{digits}";
        if (decpt >= digits.Length)
            return $"{sign}{digits}{new string('0', decpt - digits.Length)}.0";
        return $"{sign}{digits[..decpt]}.{digits[decpt..]}";
    }

    private static void WriteString(string value, StringBuilder output)
    {
        output.Append('"');
        foreach (var ch in value)
        {
            switch (ch)
            {
                case '"': output.Append("\\\""); break;
                case '\\': output.Append("\\\\"); break;
                case '\b': output.Append("\\b"); break;
                case '\f': output.Append("\\f"); break;
                case '\n': output.Append("\\n"); break;
                case '\r': output.Append("\\r"); break;
                case '\t': output.Append("\\t"); break;
                default:
                    if (ch < 0x20 || ch > 0x7E)
                        output.Append("\\u").Append(((int)ch).ToString("x4"));
                    else
                        output.Append(ch);
                    break;
            }
        }
        output.Append('"');
    }
}
