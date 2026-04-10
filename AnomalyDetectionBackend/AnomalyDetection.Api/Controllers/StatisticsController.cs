using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    [Authorize]
    public class StatisticsController : ControllerBase
    {
        #region Fields
        private readonly StatisticsService _statisticsService;
        private readonly ILogger<StatisticsController> _logger;
        #endregion

        #region Constructor
        public StatisticsController(StatisticsService statisticsService, ILogger<StatisticsController> logger)
        {
            _statisticsService = statisticsService;
            _logger = logger;
        }
        #endregion

        #region Endpoints
        [HttpGet]
        public IActionResult GetDashboardStats()
        {
            try
            {
                string role = User.FindFirstValue(ClaimTypes.Role) ?? "User";

                int userId = 0;
                var userIdStr = User.FindFirstValue("id");
                if (!string.IsNullOrEmpty(userIdStr)) int.TryParse(userIdStr, out userId);

                var stats = _statisticsService.GetWeeklyStatistics(userId, role);

                _logger.LogInformation("[STATISTICS] User '{Username}' (Role: {Role}) successfully fetched dashboard stats.", userId, role);

                return Ok(stats);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Database query failed while fetching dashboard statistics.");
                return StatusCode(500, "An unexpected internal server error occurred while fetching your dashboard statistics.");
            }
        }

        [HttpGet("history")]
        public IActionResult GetHistory(
            [FromQuery] int page = 1,
            [FromQuery] int pageSize = 10,
            [FromQuery] string sortBy = "timestamp",
            [FromQuery] bool sortDesc = true)
        {
            try
            {
                string role = User.FindFirstValue(ClaimTypes.Role) ?? "User";

                int userId = 0;
                var userIdStr = User.FindFirstValue("id");
                if (!string.IsNullOrEmpty(userIdStr)) int.TryParse(userIdStr, out userId);

                var pagedHistory = _statisticsService.GetInferenceHistory(userId, role, page, pageSize, sortBy, sortDesc);

                _logger.LogInformation("[STATISTICS] User '{Username}' (Role: {Role}) successfully fetched inference history page {Page}.", userId, role, page);

                return Ok(pagedHistory);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[CRITICAL ERROR] Database query failed while fetching inference history.");
                return StatusCode(500, "An unexpected internal server error occurred while fetching your history.");
            }
        }
        #endregion
    }
}
