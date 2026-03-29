using AnomalyDetection.Api.Repositories;
using AnomalyDetection.Api.Models.DTOs;
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
        private readonly UserRepository _userRepo;
        #endregion

        #region Constructor
        public UsersController(UserRepository userRepo)
        {
            _userRepo = userRepo;
        }
        #endregion

        #region Endpoints
        [HttpGet]
        public IActionResult GetAllUsers()
        {
            try
            {
                var users = _userRepo.GetAllUsers();
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
                if (request.Role != "Admin" && request.Role != "User")
                {
                    return BadRequest("Invalid role. Must be 'Admin' or 'User'.");
                }

                _userRepo.UpdateUserRole(id, request.Role);
                return Ok(new { Message = $"User {id} is now an {request.Role}." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error while updating user role: {ex.Message}");
            }
        }
        #endregion
    }
}
