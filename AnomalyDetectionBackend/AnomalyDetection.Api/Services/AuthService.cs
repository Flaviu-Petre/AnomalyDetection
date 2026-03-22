using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories;

namespace AnomalyDetection.Api.Services
{
    public class AuthService
    {
        #region Fields
        private readonly UserRepository _userRepo;
        #endregion

        #region Constructor
        public AuthService(UserRepository userRepo)
        {
            _userRepo = userRepo;
        }
        #endregion

        #region Methods
        public bool IsUsernameTaken(string username)
        {
            return _userRepo.UserExists(username);
        }

        public void RegisterUser(string username, string rawPassword, string? providedSecretCode)
        {
            string hashedPassword = BCrypt.Net.BCrypt.HashPassword(rawPassword);

            string actualSecretCode = Environment.GetEnvironmentVariable("ADMIN_SECRET_CODE")
                ?? "FALLBACK_SECRET_DO_NOT_USE";

            string assignedRole = "User";
            if (!string.IsNullOrWhiteSpace(providedSecretCode) && providedSecretCode == actualSecretCode)
            {
                assignedRole = "Admin";
            }

            var newUser = new User
            {
                Username = username,
                PasswordHash = hashedPassword,
                Role = assignedRole
            };

            _userRepo.AddUser(newUser);
        }

        public User? ValidateUserCredentials(string username, string rawPassword)
        {
            var user = _userRepo.GetUserByUsername(username);

            if (user == null || !BCrypt.Net.BCrypt.Verify(rawPassword, user.PasswordHash))
            {
                return null;
            }

            return user;
        }

        #endregion
    }
}
