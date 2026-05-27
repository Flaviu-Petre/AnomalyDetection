using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using AnomalyDetection.Api.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    public class AuthController : ControllerBase
    {
        #region Fields
        private readonly IAuthService _authService;
        private readonly ILogger<AuthController> _logger;
        #endregion

        #region Constructor
        public AuthController(IAuthService authService, ILogger<AuthController> logger)
        {
            _authService = authService;
            _logger = logger;
        }
        #endregion

        #region Endpoints
        [HttpPost("register")]
        public IActionResult Register([FromBody] RegisterRequest request)
        {
            try
            {
                if (_authService.IsUsernameTaken(request.Username))
                {
                    _logger.LogWarning("[AUTH] Failed registration attempt. Username '{Username}' is already taken.", request.Username);
                    return BadRequest("Username already exists.");
                }

                _authService.RegisterUser(request.Username, request.Password, request.Email);

                _logger.LogInformation("[AUTH] New user registered successfully: '{Username}'", request.Username);
                return Ok("User registered successfully!");
            }
            catch (ArgumentException ex)
            {
                _logger.LogWarning("[AUTH] Failed registration attempt for username '{Username}': {Reason}", request.Username, ex.Message);
                return BadRequest(ex.Message);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] An unexpected error occurred during registration for username '{Username}'.", request.Username);
                return StatusCode(500, "An unexpected internal server error occurred during registration.");
            }
        }

        [HttpPost("login")]
        public IActionResult Login([FromBody] LoginRequest request)
        {
            var identifier = request.Email ?? request.Username ?? "unknown";

            try
            {
                var response = _authService.Login(request.Username, request.Email, request.Password);

                if (response == null)
                {
                    _logger.LogWarning("[AUTH] Failed login attempt for: '{Identifier}'", identifier);
                    return Unauthorized("Invalid username or password.");
                }

                _logger.LogInformation("[AUTH] User logged in successfully: '{Identifier}' (Role: {Role})", identifier, response.Role);

                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] An unexpected error occurred during login for: '{Identifier}'.", identifier);
                return StatusCode(500, "An unexpected internal server error occurred during login.");
            }
        }
        #endregion
    }
}