using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories;
using AnomalyDetection.Api.Repositories.Interfaces;
using AnomalyDetection.Api.Services.Interfaces;
using System.Data;

namespace AnomalyDetection.Api.Services
{
    public class StatisticsService : IStatisticsService
    {
        #region Fields
        private readonly IStatisticsRepository _statisticsRepo;
        #endregion

        #region Constructor
        public StatisticsService(IStatisticsRepository statisticsRepo)
        {
            _statisticsRepo = statisticsRepo ?? throw new ArgumentNullException(nameof(statisticsRepo));
        }
        #endregion

        #region Methods
        public void SaveInferenceResult(string category, bool isAnomaly, float score, float threshold, int userId, string imageName)
        {
            var record = new InferenceRecord
            {
                Category = category,
                IsAnomaly = isAnomaly,
                Score = score,
                ThresholdUsed = threshold,
                Timestamp = DateTime.UtcNow,
                UserId = userId,
                ImageName = imageName
            };

            _statisticsRepo.AddInferenceRecord(record);
        }

        public DashboardStatsResponse GetWeeklyStatistics(int userId, string role)
        {
            var sevenDaysAgo = DateTime.UtcNow.AddDays(-7);

            var records = _statisticsRepo.GetRecordsSince(sevenDaysAgo);

            if (role != "Admin")
            {
                records = records.Where(r => r.UserId == userId).ToList();
            }

            int totalInferences = records.Count;
            int totalAnomalies = records.Count(r => r.IsAnomaly);
            double anomalyRate = totalInferences == 0 ? 0 : Math.Round((double)totalAnomalies / totalInferences * 100, 2);

            var anomaliesByCategory = records
                .Where(r => r.IsAnomaly)
                .GroupBy(r => r.Category)
                .ToDictionary(g => g.Key, g => g.Count());

            var inferencesByDay = records
                .GroupBy(r => r.Timestamp.ToString("yyyy-MM-dd"))
                .ToDictionary(g => g.Key, g => g.Count());

            return new DashboardStatsResponse
            {
                TotalInferencesThisWeek = totalInferences,
                TotalAnomaliesThisWeek = totalAnomalies,
                OverallAnomalyRatePercentage = anomalyRate,
                AnomaliesByCategory = anomaliesByCategory,
                InferencesByDay = inferencesByDay
            };
        }

        public PagedResult<InferenceHistoryDto> GetInferenceHistory(
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
            DateTime? dateTo = null)
        {
            var thirtyDaysAgo = DateTime.UtcNow.AddDays(-30);

            int? resolvedUserId = role == "Admin" ? null : userId;

            return _statisticsRepo.GetPagedHistory(
                thirtyDaysAgo,
                resolvedUserId,
                pageNumber,
                pageSize,
                sortBy,
                sortDescending,
                isAnomaly,
                category,
                role == "Admin" ? filterUsername : null,
                dateFrom,
                dateTo);
        }
        #endregion
    }
}