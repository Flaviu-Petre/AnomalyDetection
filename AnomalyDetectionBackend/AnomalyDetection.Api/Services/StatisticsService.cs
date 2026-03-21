using System;
using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models;

namespace AnomalyDetection.Api.Services
{
    public class StatisticsService
    {
        #region Fields
        private readonly AppDbContext _db;
        #endregion

        #region Constructor
        public StatisticsService(AppDbContext db)
        {
            _db = db ?? throw new ArgumentNullException(nameof(db));
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

            _db.InferenceRecords.Add(record);
            _db.SaveChanges();
        }
        #endregion
    }
}