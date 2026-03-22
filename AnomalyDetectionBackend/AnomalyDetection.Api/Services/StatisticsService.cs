using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models;
using AnomalyDetection.Api.Repositories;

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
        public void SaveInferenceResult(string category, bool isAnomaly, float score, float threshold)
        {
            var record = new InferenceRecord
            {
                Category = category,
                IsAnomaly = isAnomaly,
                Score = score,
                ThresholdUsed = threshold,
                Timestamp = DateTime.UtcNow
            };

            _statisticsRepo.AddInferenceRecord(record);
        }
        #endregion
    }
}