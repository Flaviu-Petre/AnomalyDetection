namespace AnomalyDetection.Api.Models.DTOs
{
    public class LoginRequest
    {
        public string? Username { get; set; }
        public string? Email { get; set; }
        public string Password { get; set; } = string.Empty;
    }
}