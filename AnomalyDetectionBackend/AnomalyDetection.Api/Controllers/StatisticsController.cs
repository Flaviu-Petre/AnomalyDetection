using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AnomalyDetection.Api.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    [Authorize(Roles = "Admin")]
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
                var stats = _statisticsService.GetWeeklyStatistics();
                return Ok(stats);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error while fetching statistics: {ex.Message}");
            }
        }
        #endregion
    }
}
