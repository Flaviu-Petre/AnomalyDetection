using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories;
using System.Data;

namespace AnomalyDetection.Api.Services
{
    public class StatisticsService
    {
        #region Fields
        private readonly StatisticsRepository _statisticsRepo;
        #endregion

        #region Constructor
        public StatisticsService(StatisticsRepository statisticsRepo)
        {
            _statisticsRepo = statisticsRepo ?? throw new ArgumentNullException(nameof(statisticsRepo));
        }
        #endregion

        #region Methods
        public void SaveInferenceResult(string category, bool isAnomaly, float score, float threshold, int userId)
        {
            var record = new InferenceRecord
            {
                Category = category,
                IsAnomaly = isAnomaly,
                Score = score,
                ThresholdUsed = threshold,
                Timestamp = DateTime.UtcNow,
                UserId = userId
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

        public List<InferenceRecord> GetInferenceHistory(int userId, string role)
        {
            var records = _statisticsRepo.GetRecordsSince(DateTime.UtcNow.AddDays(-30));

            if (role != "Admin")
            {
                records = records.Where(r => r.UserId == userId).ToList();
            }

            return records.OrderByDescending(r => r.Timestamp).ToList();
        }
        #endregion
    }
}