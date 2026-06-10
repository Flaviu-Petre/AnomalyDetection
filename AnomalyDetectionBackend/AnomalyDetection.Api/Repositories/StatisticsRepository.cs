using AnomalyDetection.Api.Data;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories.Interfaces;

namespace AnomalyDetection.Api.Repositories
{
    public class StatisticsRepository : IStatisticsRepository
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

        public PagedResult<InferenceHistoryDto> GetPagedHistory(
            DateTime startDate,
            int? userId,
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
                query = query.Where(r => r.UserId == userId.Value);

            if (isAnomaly.HasValue)
                query = query.Where(r => r.IsAnomaly == isAnomaly.Value);

            if (!string.IsNullOrWhiteSpace(category))
                query = query.Where(r => r.Category == category.ToLower());

            if (!string.IsNullOrWhiteSpace(filterUsername))
                query = query.Where(r => r.Username.Contains(filterUsername));

            if (dateFrom.HasValue)
                query = query.Where(r => r.Timestamp >= dateFrom.Value);

            if (dateTo.HasValue)
                query = query.Where(r => r.Timestamp <= dateTo.Value);

            query = sortBy?.ToLower() switch
            {
                "category" => sortDescending ? query.OrderByDescending(r => r.Category) : query.OrderBy(r => r.Category),
                "score" => sortDescending ? query.OrderByDescending(r => r.Score) : query.OrderBy(r => r.Score),
                "isanomaly" => sortDescending ? query.OrderByDescending(r => r.IsAnomaly) : query.OrderBy(r => r.IsAnomaly),
                "operator" => sortDescending ? query.OrderByDescending(r => r.Username) : query.OrderBy(r => r.Username),
                _ => sortDescending ? query.OrderByDescending(r => r.Timestamp) : query.OrderBy(r => r.Timestamp)
            };

            int totalCount = query.Count();

            var items = query
                .Skip((pageNumber - 1) * pageSize)
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