using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Repositories;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    [Authorize(Roles = "Admin")]
    public class UsersController : ControllerBase
    {
        #region Fields
        private readonly UserService _userService;
        #endregion

        #region Constructor
        public UsersController(UserService userService)
        {
            _userService = userService;
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
                return StatusCode(500, $"Internal server error while fetching users: {ex.Message}");
            }
        }

        [HttpPut("{id}/role")]
        public IActionResult UpdateRole(int id, [FromBody] UpdateRoleRequest request)
        {
            try
            {
                var currentUserIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
                _userService.UpdateUserRole(currentUserIdStr, id, request.Role);

                return Ok(new { Message = $"User {id} is now an {request.Role}." });
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(ex.Message);
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error while updating user role: {ex.Message}");
            }
        }
        #endregion
    }
}
