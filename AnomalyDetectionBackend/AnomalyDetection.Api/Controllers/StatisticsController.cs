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
        #endregion

        #region Constructor
        public StatisticsController(StatisticsService statisticsService)
        {
            _statisticsService = statisticsService;
        }
        #endregion

        #region Endpoints
        [HttpGet]
        public IActionResult GetDashboardStats()
        {
            try
            {
                string username = User.FindFirstValue(ClaimTypes.NameIdentifier) ?? "Unknown";
                string role = User.FindFirstValue(ClaimTypes.Role) ?? "User";

                var stats = _statisticsService.GetWeeklyStatistics(username, role);

                return Ok(stats);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error while fetching statistics: {ex.Message}");
            }
        }

        [HttpGet("history")]
        public IActionResult GetHistory()
        {
            try
            {
                string username = User.FindFirstValue(ClaimTypes.NameIdentifier) ?? "Unknown";
                string role = User.FindFirstValue(ClaimTypes.Role) ?? "User";

                var history = _statisticsService.GetInferenceHistory(username, role);

                return Ok(history);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error while fetching history: {ex.Message}");
            }
        }
        #endregion
    }
}
