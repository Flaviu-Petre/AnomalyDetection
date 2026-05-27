using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Models.Entities;

namespace AnomalyDetection.Api.Repositories.Interfaces
{
    public interface IStatisticsRepository
    {
        void AddInferenceRecord(InferenceRecord record);
        List<InferenceRecord> GetRecordsSince(DateTime startDate);
        PagedResult<InferenceHistoryDto> GetPagedHistory(
            DateTime startDate,
            int? userId,
            int pageNumber,
            int pageSize,
            string sortBy,
            bool sortDescending);
    }
}