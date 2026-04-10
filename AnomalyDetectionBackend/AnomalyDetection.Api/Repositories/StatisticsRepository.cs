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

        public PagedResult<InferenceHistoryDto> GetPagedHistory(DateTime startDate, int? userId, int pageNumber, int pageSize, string sortBy, bool sortDescending)
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
                            Username = user.Username,
                            ImageName = record.ImageName
                        };

            if (userId.HasValue)
            {
                query = query.Where(r => r.UserId == userId.Value);
            }

            query = sortBy?.ToLower() switch
            {
                "category" => sortDescending ? query.OrderByDescending(r => r.Category) : query.OrderBy(r => r.Category),
                "score" => sortDescending ? query.OrderByDescending(r => r.Score) : query.OrderBy(r => r.Score),
                "isanomaly" => sortDescending ? query.OrderByDescending(r => r.IsAnomaly) : query.OrderBy(r => r.IsAnomaly),
                "operator" => sortDescending ? query.OrderByDescending(r => r.Username) : query.OrderBy(r => r.Username),
                _ => sortDescending ? query.OrderByDescending(r => r.Timestamp) : query.OrderBy(r => r.Timestamp)
            };

            int totalCount = query.Count();

            var items = query.Skip((pageNumber - 1) * pageSize)
                             .Take(pageSize)
                             .ToList();

            return new PagedResult<InferenceHistoryDto>
            {
                Items = items,
                TotalCount = totalCount,
                PageNumber = pageNumber,
                PageSize = pageSize
            };
        }
        #endregion
        }
}