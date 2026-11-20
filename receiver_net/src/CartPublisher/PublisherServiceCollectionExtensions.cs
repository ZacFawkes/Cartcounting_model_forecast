using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace CartPublisher;

/// <summary>Registers the outbox, the HTTP publisher and the retry worker.</summary>
public static class PublisherServiceCollectionExtensions
{
    public static IServiceCollection AddCartPublisher(this IServiceCollection services,
                                                      IConfiguration configuration)
    {
        services.AddOptions<PublisherOptions>()
            .Bind(configuration.GetSection(PublisherOptions.SectionName))
            .PostConfigure(options =>
            {
                // Environment variables win, so no secret needs to be committed.
                Apply(configuration["CART_OUTBOX_SQL_CONNECTION"],
                      v => options.ConnectionString = v);
                Apply(configuration["CART_RESULT_RECEIVER_URL"],
                      v => options.ReceiverUrl = v);
                Apply(configuration["CART_RESULT_RECEIVER_TOKEN"],
                      v => options.Token = v);
                Apply(configuration["CART_RESULT_RECEIVER_ENABLED"],
                      v => options.Enabled = v == "1" || v.Equals("true",
                          StringComparison.OrdinalIgnoreCase));
            });

        services.AddSingleton<OutboxConnectionFactory>();
        services.AddScoped<Outbox>();

        services.AddHttpClient<ResultPublisher>((provider, client) =>
        {
            var options = provider
                .GetRequiredService<Microsoft.Extensions.Options.IOptions<PublisherOptions>>()
                .Value;
            client.Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds);
        });

        // One instance, registered twice: the host runs it, the admin API reads
        // its Running flag.
        services.AddSingleton<OutboxRetryWorker>();
        services.AddHostedService(sp => sp.GetRequiredService<OutboxRetryWorker>());

        return services;
    }

    private static void Apply(string? value, Action<string> set)
    {
        if (!string.IsNullOrWhiteSpace(value))
            set(value);
    }
}
