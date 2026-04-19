using AnomalyDetection.Api.Extensions;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    [Authorize(Roles = "Admin")]
    public class UsersController : ControllerBase
    {
        #region Fields
        private readonly UserService _userService;
        private readonly ILogger<UsersController> _logger;
        #endregion

        #region Constructor
        public UsersController(UserService userService, ILogger<UsersController> logger)
        {
            _userService = userService;
            _logger = logger;
        }
        #endregion

        #region Endpoints
        [HttpGet]
        public IActionResult GetAllUsers()
        {
            try
            {
                var users = _userService.GetAllUsers();

                return Ok(users);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Failed to fetch the list of users from the database.");
                return StatusCode(500, "An unexpected internal server error occurred while fetching users.");
            }
        }

        [HttpPut("{id}/role")]
        public IActionResult UpdateRole(int id, [FromBody] UpdateRoleRequest request)
        {
            try
            {
                var currentUserIdStr = User.GetUserIdString();
                if (string.IsNullOrEmpty(currentUserIdStr))
                {
                    _logger.LogWarning("[SECURITY] Role update blocked: Could not identify the user making the request.");
                    return Unauthorized("Security Error: Could not identify the user making this request.");
                }

                _userService.UpdateUserRole(currentUserIdStr, id, request.Role);

                _logger.LogInformation("[SECURITY AUDIT] Admin ID '{AdminId}' successfully changed User ID '{TargetId}' to role '{NewRole}'.",
                    currentUserIdStr, id, request.Role);

                return Ok(new { Message = $"User {id} is now an {request.Role}." });
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogWarning(ex, "[SECURITY] Role update blocked: Invalid operation by Admin ID '{AdminId}'.", User.GetUserIdString());
                return BadRequest(ex.Message);
            }
            catch (ArgumentException ex)
            {
                _logger.LogWarning(ex, "[SECURITY] Role update blocked: Invalid argument provided.");
                return BadRequest(ex.Message);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] An unexpected error occurred while updating the role for User ID '{TargetId}'.", id);
                return StatusCode(500, "An unexpected internal server error occurred while updating the user's role.");
            }
        }

        [HttpDelete("{id}")]
        public IActionResult DeleteUser(int id)
        {
            try
            {
                var currentUserIdStr = User.GetUserIdString();
                if (string.IsNullOrEmpty(currentUserIdStr))
                {
                    _logger.LogWarning("[SECURITY] User deletion blocked: Could not identify the user making the request.");
                    return Unauthorized("Security Error: Could not identify the user making this request.");
                }

                _userService.DeleteUser(currentUserIdStr, id);

                _logger.LogInformation("[SECURITY AUDIT] Admin ID '{AdminId}' successfully deleted User ID '{TargetId}'.",
                    currentUserIdStr, id);

                return Ok(new { Message = $"Successfully deleted user {id}." });
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogWarning(ex, "[SECURITY] User deletion blocked: Admin ID '{AdminId}' attempted to delete themselves.", User.GetUserIdString());
                return BadRequest(ex.Message);
            }
            catch (KeyNotFoundException ex)
            {
                _logger.LogWarning("[USERS] Deletion failed: User ID '{TargetId}' does not exist.", id);
                return NotFound(ex.Message);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] An unexpected error occurred while deleting User ID '{TargetId}'.", id);
                return StatusCode(500, "An unexpected internal server error occurred while deleting the user.");
            }
        }
        #endregion
    }
}
