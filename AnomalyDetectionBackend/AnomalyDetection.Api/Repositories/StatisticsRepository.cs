using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models;

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
        #endregion
    }
}