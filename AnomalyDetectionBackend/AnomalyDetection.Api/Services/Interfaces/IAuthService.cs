using AnomalyDetection.Api.Models.DTOs;

namespace AnomalyDetection.Api.Services.Interfaces
{
    public interface IAuthService
    {
        bool IsUsernameTaken(string username);
        void RegisterUser(string username, string rawPassword, string email);
        LoginResponse? Login(string? username, string? email, string rawPassword);
    }
}