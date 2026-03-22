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
        #endregion

        #region Constructor
        public AuthController(AuthService authService)
        {
            _authService = authService;
        }
        #endregion

        #region Endpoints
        [HttpPost("register")]
        public IActionResult Register([FromBody] LoginRequest request)
        {
            if (_authService.IsUsernameTaken(request.Username))
            {
                return BadRequest("Username already exists.");
            }
            _authService.RegisterUser(request.Username, request.Password);

            return Ok("User registered successfully!");
        }

        [HttpPost("login")]
        public IActionResult Login([FromBody] LoginRequest request)
        {
            var user = _authService.ValidateUserCredentials(request.Username, request.Password);

            if (user == null)
            {
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
                new Claim(ClaimTypes.Role, user.Role)
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