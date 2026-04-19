namespace AnomalyDetection.Api.Models.Configuration
{
    public class JwtSettings
    {
        public string Secret { get; set; } = string.Empty;
        public string Issuer { get; set; } = "AnomalyFactoryApi";
        public string Audience { get; set; } = "AnomalyFactoryFrontend";
        public int ExpirationHours { get; set; } = 8;
    }
}
