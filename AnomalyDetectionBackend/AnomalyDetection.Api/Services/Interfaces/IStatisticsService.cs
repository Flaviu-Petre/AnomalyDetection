using AnomalyDetection.Api.Models.DTOs;

namespace AnomalyDetection.Api.Services.Interfaces
{
    public interface IStatisticsService
    {
        void SaveInferenceResult(string category, bool isAnomaly, float score, float threshold, int userId, string imageName);
        DashboardStatsResponse GetWeeklyStatistics(int userId, string role);
        PagedResult<InferenceHistoryDto> GetInferenceHistory(
            int userId,
            string role,
            int pageNumber,
            int pageSize,
            string sortBy,
            bool sortDescending,
            bool? isAnomaly = null,
            string? category = null,
            string? filterUsername = null,
            DateTime? dateFrom = null,
            DateTime? dateTo = null
        );
    }
}