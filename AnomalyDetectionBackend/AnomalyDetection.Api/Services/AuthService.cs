using AnomalyDetection.Api.Models.Configuration;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories;
using AnomalyDetection.Api.Repositories.Interfaces;
using AnomalyDetection.Api.Services.Interfaces;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;
using System.ComponentModel.DataAnnotations;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;

namespace AnomalyDetection.Api.Services
{
    public class AuthService : IAuthService
    {
        #region Fields
        private readonly IUserRepository _userRepo;
        private readonly JwtSettings _jwtSettings;
        #endregion

        #region Constructor
        public AuthService(IUserRepository userRepo, IOptions<JwtSettings> jwtSettings)
        {
            _userRepo = userRepo;
            _jwtSettings = jwtSettings.Value;
        }
        #endregion

        #region Public methods
        public bool IsUsernameTaken(string username)
        {
            return _userRepo.UserExists(username);
        }

        public void RegisterUser(string username, string rawPassword, string email)
        {
            if (string.IsNullOrWhiteSpace(email))
            {
                throw new ArgumentException("Email address is required.");
            }

            var emailValidator = new EmailAddressAttribute();
            if (!emailValidator.IsValid(email))
            {
                throw new ArgumentException("Invalid email address format.");
            }

            string hashedPassword = BCrypt.Net.BCrypt.HashPassword(rawPassword);

            var newUser = new User
            {
                Username = username,
                Email = email,
                PasswordHash = hashedPassword,
                Role = "User"
            };

            _userRepo.AddUser(newUser);
        }

        public LoginResponse? Login(string? username, string? email, string rawPassword)
        {
            if (string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(email))
                return null;

            var user = _userRepo.GetUserByUsername(username);

            if (user == null || user.Email != email)
                return null;

            if (!BCrypt.Net.BCrypt.Verify(rawPassword, user.PasswordHash))
                return null;

            var token = GenerateJwtToken(user.Id, user.Username, user.Role);

            return new LoginResponse
            {
                Token = new JwtSecurityTokenHandler().WriteToken(token),
                Role = user.Role,
                Expiration = token.ValidTo
            };
        }

        public ForgotPasswordResponse ForgotPassword(string email)
        {
            var user = _userRepo.GetUserByEmail(email);

            if (user == null)
            {
                return new ForgotPasswordResponse
                {
                    Message = "If this email is registered, a reset token has been generated."
                };
            }

            var token = Convert.ToHexString(RandomNumberGenerator.GetBytes(32));

            user.PasswordResetToken = token;
            user.PasswordResetTokenExpiry = DateTime.UtcNow.AddHours(1);
            _userRepo.UpdateUser(user);

            return new ForgotPasswordResponse
            {
                Message = "Password reset token generated successfully.",
                ResetToken = token
            };
        }

        public bool ResetPassword(string token, string newPassword)
        {
            var user = _userRepo.GetUserByResetToken(token);

            if (user == null)
                return false;

            if (user.PasswordResetTokenExpiry == null || user.PasswordResetTokenExpiry < DateTime.UtcNow)
                return false;

            user.PasswordHash = BCrypt.Net.BCrypt.HashPassword(newPassword);
            user.PasswordResetToken = null;
            user.PasswordResetTokenExpiry = null;
            _userRepo.UpdateUser(user);

            return true;
        }

        #endregion

        #region Private methods
        private JwtSecurityToken GenerateJwtToken(int userId, string username, string role)
        {
            var claims = new[]
            {
                new Claim(JwtRegisteredClaimNames.Sub, username),
                new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString()),
                new Claim(ClaimTypes.Role, role),
                new Claim("id", userId.ToString())
            };

            var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtSettings.Secret));
            var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

            return new JwtSecurityToken(
                issuer: _jwtSettings.Issuer,
                audience: _jwtSettings.Audience,
                claims: claims,
                expires: DateTime.UtcNow.AddHours(_jwtSettings.ExpirationHours),
                signingCredentials: creds
            );
        }
        #endregion
    }
}
