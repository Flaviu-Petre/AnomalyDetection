using AnomalyDetection.Api.Data;
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
        #endregion
    }
}