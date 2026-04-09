using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    public class AuthController : ControllerBase
    {
        #region Fields
        private readonly AuthService _authService;
        private readonly ILogger<AuthController> _logger;
        #endregion

        #region Constructor
        public AuthController(AuthService authService, ILogger<AuthController> logger)
        {
            _authService = authService;
            _logger = logger;
        }
        #endregion

        #region Endpoints
        [HttpPost("register")]
        public IActionResult Register([FromBody] RegisterRequest request)
        {
            if (_authService.IsUsernameTaken(request.Username))
            {
                _logger.LogWarning("[AUTH] Failed registration attempt. Username '{Username}' is already taken.", request.Username);
                return BadRequest("Username already exists.");
            }
            _authService.RegisterUser(request.Username, request.Password);

            _logger.LogInformation("[AUTH] New user registered successfully: '{Username}'", request.Username);
            return Ok("User registered successfully!");
        }

        [HttpPost("login")]
        public IActionResult Login([FromBody] LoginRequest request)
        {
            var user = _authService.ValidateUserCredentials(request.Username, request.Password);

            if (user == null)
            {
                _logger.LogWarning("[AUTH] Failed login attempt for username: '{Username}'", request.Username);
                return Unauthorized("Invalid username or password.");
            }

            string jwtSecret = Environment.GetEnvironmentVariable("JWT_SECRET")
                ?? throw new Exception("JWT_SECRET missing.");
            string jwtIssuer = Environment.GetEnvironmentVariable("JWT_ISSUER") ?? "AnomalyFactoryApi";
            string jwtAudience = Environment.GetEnvironmentVariable("JWT_AUDIENCE") ?? "AnomalyFactoryFrontend";

            var claims = new[]
            {
                new Claim(JwtRegisteredClaimNames.Sub, user.Username),
                new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString()),
                new Claim(ClaimTypes.Role, user.Role),
                new Claim("id", user.Id.ToString())
            };

            var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecret));
            var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

            var token = new JwtSecurityToken(
                issuer: jwtIssuer,
                audience: jwtAudience,
                claims: claims,
                expires: DateTime.UtcNow.AddHours(8),
                signingCredentials: creds
            );

            _logger.LogInformation("[AUTH] User logged in successfully: '{Username}' (Role: {Role})", user.Username, user.Role);

            return Ok(new
            {
                Token = new JwtSecurityTokenHandler().WriteToken(token),
                Role = user.Role,
                Expiration = token.ValidTo
            });
        }
        #endregion
    }
}