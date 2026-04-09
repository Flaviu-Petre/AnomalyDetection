using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Entities;

namespace AnomalyDetection.Api.Repositories
{
    public class StatisticsRepository
    {
        #region Fields
        private readonly AppDbContext _db;
        #endregion

        #region Constructor
        public StatisticsRepository(AppDbContext db)
        {
            _db = db;
        }
        #endregion

        #region Methods
        public void AddInferenceRecord(InferenceRecord record)
        {
            _db.InferenceRecords.Add(record);
            _db.SaveChanges();
        }

        public List<InferenceRecord> GetRecordsSince(DateTime startDate)
        {
            return _db.InferenceRecords
                      .Where(r => r.Timestamp >= startDate)
                      .ToList();
        }

        public List<InferenceHistoryDto> GetHistoryWithUsernames(DateTime startDate)
        {
            var query = from record in _db.InferenceRecords
                        join user in _db.Users on record.UserId equals user.Id
                        where record.Timestamp >= startDate
                        select new InferenceHistoryDto
                        {
                            Id = record.Id,
                            Timestamp = record.Timestamp,
                            Category = record.Category,
                            IsAnomaly = record.IsAnomaly,
                            Score = record.Score,
                            ThresholdUsed = record.ThresholdUsed,
                            UserId = record.UserId,
                            Username = user.Username
                        };

            return query.ToList();
        }
        #endregion
    }
}