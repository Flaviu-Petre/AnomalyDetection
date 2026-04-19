using AnomalyDetection.Api.Models.Configuration;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;
using System.ComponentModel.DataAnnotations;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;

namespace AnomalyDetection.Api.Services
{
    public class AuthService
    {
        #region Fields
        private readonly UserRepository _userRepo;
        private readonly JwtSettings _jwtSettings;
        #endregion

        #region Constructor
        public AuthService(UserRepository userRepo, IOptions<JwtSettings> jwtSettings)
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

        public LoginResponse? Login(string username, string rawPassword)
        {
            var user = _userRepo.GetUserByUsername(username);

            if (user == null || !BCrypt.Net.BCrypt.Verify(rawPassword, user.PasswordHash))
            {
                return null;
            }

            var token = GenerateJwtToken(user.Id, user.Username, user.Role);

            return new LoginResponse
            {
                Token = new JwtSecurityTokenHandler().WriteToken(token),
                Role = user.Role,
                Expiration = token.ValidTo
            };
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
