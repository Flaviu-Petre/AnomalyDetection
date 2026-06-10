namespace AnomalyDetection.Api.Models.DTOs
{
    public class ForgotPasswordResponse
    {
        public string Message { get; set; } = string.Empty;
        public string? ResetToken { get; set; }
    }
}