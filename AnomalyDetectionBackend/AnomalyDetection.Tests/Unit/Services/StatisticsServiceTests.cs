using AnomalyDetection.Api.Models.Entities;
using AnomalyDetection.Api.Repositories.Interfaces;
using AnomalyDetection.Api.Services;
using AnomalyDetection.Api.Models.DTOs;
using FluentAssertions;
using Moq;

namespace AnomalyDetection.Tests.Unit.Services
{
    public class StatisticsServiceTests
    {
        #region Setup
        private readonly Mock<IStatisticsRepository> _mockStatsRepo;
        private readonly StatisticsService _statisticsService;

        public StatisticsServiceTests()
        {
            _mockStatsRepo = new Mock<IStatisticsRepository>();
            _statisticsService = new StatisticsService(_mockStatsRepo.Object);
        }
        #endregion

        #region SaveInferenceResult Tests
        [Fact]
        public void SaveInferenceResult_CallsRepository_WithCorrectData()
        {
            // Arrange
            _mockStatsRepo.Setup(r => r.AddInferenceRecord(It.IsAny<InferenceRecord>()));

            // Act
            _statisticsService.SaveInferenceResult("bottle", true, 0.85f, 0.5f, 1, "test.png");

            // Assert
            _mockStatsRepo.Verify(r => r.AddInferenceRecord(It.Is<InferenceRecord>(rec =>
                rec.Category == "bottle" &&
                rec.IsAnomaly == true &&
                rec.Score == 0.85f &&
                rec.ThresholdUsed == 0.5f &&
                rec.UserId == 1 &&
                rec.ImageName == "test.png"
            )), Times.Once);
        }

        [Fact]
        public void SaveInferenceResult_SetsTimestamp_ToUtcNow()
        {
            // Arrange
            InferenceRecord? savedRecord = null;
            _mockStatsRepo.Setup(r => r.AddInferenceRecord(It.IsAny<InferenceRecord>()))
                          .Callback<InferenceRecord>(r => savedRecord = r);

            var before = DateTime.UtcNow;

            // Act
            _statisticsService.SaveInferenceResult("bottle", false, 0.2f, 0.5f, 1, "test.png");

            var after = DateTime.UtcNow;

            // Assert
            savedRecord.Should().NotBeNull();
            savedRecord!.Timestamp.Should().BeOnOrAfter(before).And.BeOnOrBefore(after);
        }

        [Fact]
        public void SaveInferenceResult_CallsRepository_ForNonAnomaly()
        {
            // Act
            _statisticsService.SaveInferenceResult("capsule", false, 0.1f, 0.5f, 2, "good.png");

            // Assert
            _mockStatsRepo.Verify(r => r.AddInferenceRecord(It.Is<InferenceRecord>(rec =>
                rec.IsAnomaly == false &&
                rec.Category == "capsule"
            )), Times.Once);
        }
        #endregion

        #region GetWeeklyStatistics Tests
        [Fact]
        public void GetWeeklyStatistics_ReturnsCorrectTotals_ForAdmin()
        {
            // Arrange
            var records = new List<InferenceRecord>
            {
                new InferenceRecord { UserId = 1, Category = "bottle", IsAnomaly = true,  Timestamp = DateTime.UtcNow.AddDays(-1) },
                new InferenceRecord { UserId = 2, Category = "bottle", IsAnomaly = false, Timestamp = DateTime.UtcNow.AddDays(-2) },
                new InferenceRecord { UserId = 1, Category = "capsule", IsAnomaly = true, Timestamp = DateTime.UtcNow.AddDays(-3) },
            };
            _mockStatsRepo.Setup(r => r.GetRecordsSince(It.IsAny<DateTime>())).Returns(records);

            // Act
            var result = _statisticsService.GetWeeklyStatistics(1, "Admin");

            // Assert
            result.TotalInferencesThisWeek.Should().Be(3);
            result.TotalAnomaliesThisWeek.Should().Be(2);
            result.OverallAnomalyRatePercentage.Should().BeApproximately(66.67, 0.01);
        }

        [Fact]
        public void GetWeeklyStatistics_FiltersRecordsByUserId_ForNonAdmin()
        {
            // Arrange
            var records = new List<InferenceRecord>
            {
                new InferenceRecord { UserId = 1, Category = "bottle",  IsAnomaly = true,  Timestamp = DateTime.UtcNow.AddDays(-1) },
                new InferenceRecord { UserId = 2, Category = "capsule", IsAnomaly = false, Timestamp = DateTime.UtcNow.AddDays(-2) },
                new InferenceRecord { UserId = 1, Category = "cable",   IsAnomaly = false, Timestamp = DateTime.UtcNow.AddDays(-3) },
            };
            _mockStatsRepo.Setup(r => r.GetRecordsSince(It.IsAny<DateTime>())).Returns(records);

            // Act
            var result = _statisticsService.GetWeeklyStatistics(1, "User");

            // Assert
            result.TotalInferencesThisWeek.Should().Be(2);
            result.TotalAnomaliesThisWeek.Should().Be(1);
        }

        [Fact]
        public void GetWeeklyStatistics_ReturnsZeroAnomalyRate_WhenNoInferences()
        {
            // Arrange
            _mockStatsRepo.Setup(r => r.GetRecordsSince(It.IsAny<DateTime>()))
                          .Returns(new List<InferenceRecord>());

            // Act
            var result = _statisticsService.GetWeeklyStatistics(1, "User");

            // Assert
            result.TotalInferencesThisWeek.Should().Be(0);
            result.TotalAnomaliesThisWeek.Should().Be(0);
            result.OverallAnomalyRatePercentage.Should().Be(0);
        }

        [Fact]
        public void GetWeeklyStatistics_GroupsAnomaliesByCategory_Correctly()
        {
            // Arrange
            var records = new List<InferenceRecord>
            {
                new InferenceRecord { UserId = 1, Category = "bottle",  IsAnomaly = true,  Timestamp = DateTime.UtcNow.AddDays(-1) },
                new InferenceRecord { UserId = 1, Category = "bottle",  IsAnomaly = true,  Timestamp = DateTime.UtcNow.AddDays(-2) },
                new InferenceRecord { UserId = 1, Category = "capsule", IsAnomaly = true,  Timestamp = DateTime.UtcNow.AddDays(-3) },
                new InferenceRecord { UserId = 1, Category = "capsule", IsAnomaly = false, Timestamp = DateTime.UtcNow.AddDays(-4) },
            };
            _mockStatsRepo.Setup(r => r.GetRecordsSince(It.IsAny<DateTime>())).Returns(records);

            // Act
            var result = _statisticsService.GetWeeklyStatistics(1, "Admin");

            // Assert
            result.AnomaliesByCategory.Should().ContainKey("bottle").WhoseValue.Should().Be(2);
            result.AnomaliesByCategory.Should().ContainKey("capsule").WhoseValue.Should().Be(1);
        }

        [Fact]
        public void GetWeeklyStatistics_GroupsInferencesByDay_Correctly()
        {
            // Arrange
            var today = DateTime.UtcNow.Date;
            var records = new List<InferenceRecord>
            {
                new InferenceRecord { UserId = 1, Category = "bottle", IsAnomaly = false, Timestamp = today },
                new InferenceRecord { UserId = 1, Category = "bottle", IsAnomaly = true,  Timestamp = today },
                new InferenceRecord { UserId = 1, Category = "capsule", IsAnomaly = false, Timestamp = today.AddDays(-1) },
            };
            _mockStatsRepo.Setup(r => r.GetRecordsSince(It.IsAny<DateTime>())).Returns(records);

            // Act
            var result = _statisticsService.GetWeeklyStatistics(1, "Admin");

            // Assert
            result.InferencesByDay.Should().ContainKey(today.ToString("yyyy-MM-dd"))
                  .WhoseValue.Should().Be(2);
            result.InferencesByDay.Should().ContainKey(today.AddDays(-1).ToString("yyyy-MM-dd"))
                  .WhoseValue.Should().Be(1);
        }

        [Fact]
        public void GetWeeklyStatistics_ReturnsAllRecords_ForAdmin_RegardlessOfUserId()
        {
            // Arrange
            var records = new List<InferenceRecord>
            {
                new InferenceRecord { UserId = 1, Category = "bottle",  IsAnomaly = true,  Timestamp = DateTime.UtcNow.AddDays(-1) },
                new InferenceRecord { UserId = 2, Category = "capsule", IsAnomaly = false, Timestamp = DateTime.UtcNow.AddDays(-2) },
                new InferenceRecord { UserId = 3, Category = "cable",   IsAnomaly = true,  Timestamp = DateTime.UtcNow.AddDays(-3) },
            };
            _mockStatsRepo.Setup(r => r.GetRecordsSince(It.IsAny<DateTime>())).Returns(records);

            // Act
            var result = _statisticsService.GetWeeklyStatistics(99, "Admin");

            // Assert
            result.TotalInferencesThisWeek.Should().Be(3);
        }
        #endregion

        #region GetInferenceHistory Tests
        [Fact]
        public void GetInferenceHistory_CallsRepository_WithCorrectParameters_ForAdmin()
        {
            // Arrange
            _mockStatsRepo.Setup(r => r.GetPagedHistory(
                It.IsAny<DateTime>(), null, 1, 10, "timestamp", true,
                null, null, null, null, null))
                .Returns(new PagedResult<InferenceHistoryDto>
                {
                    Items = new List<InferenceHistoryDto>(),
                    TotalCount = 0,
                    PageNumber = 1,
                    PageSize = 10
                });

            // Act
            _statisticsService.GetInferenceHistory(1, "Admin", 1, 10, "timestamp", true);

            // Assert
            _mockStatsRepo.Verify(r => r.GetPagedHistory(
                It.IsAny<DateTime>(),
                null,
                1, 10, "timestamp", true,
                null, null, null, null, null
            ), Times.Once);
        }

        [Fact]
        public void GetInferenceHistory_CallsRepository_WithUserId_ForNonAdmin()
        {
            // Arrange
            _mockStatsRepo.Setup(r => r.GetPagedHistory(
                It.IsAny<DateTime>(), 1, 1, 10, "timestamp", true,
                null, null, null, null, null))
                .Returns(new PagedResult<InferenceHistoryDto>
                {
                    Items = new List<InferenceHistoryDto>(),
                    TotalCount = 0,
                    PageNumber = 1,
                    PageSize = 10
                });

            // Act
            _statisticsService.GetInferenceHistory(1, "User", 1, 10, "timestamp", true);

            // Assert
            _mockStatsRepo.Verify(r => r.GetPagedHistory(
                It.IsAny<DateTime>(),
                1,
                1, 10, "timestamp", true,
                null, null, null, null, null
            ), Times.Once);
        }

        [Fact]
        public void GetInferenceHistory_PassesIsAnomaly_ToRepository()
        {
            // Arrange
            _mockStatsRepo.Setup(r => r.GetPagedHistory(
                It.IsAny<DateTime>(), null, 1, 10, "timestamp", true,
                true, null, null, null, null))
                .Returns(new PagedResult<InferenceHistoryDto>
                {
                    Items = new List<InferenceHistoryDto>(),
                    TotalCount = 0,
                    PageNumber = 1,
                    PageSize = 10
                });

            // Act
            _statisticsService.GetInferenceHistory(1, "Admin", 1, 10, "timestamp", true,
                isAnomaly: true);

            // Assert
            _mockStatsRepo.Verify(r => r.GetPagedHistory(
                It.IsAny<DateTime>(), null, 1, 10, "timestamp", true,
                true, null, null, null, null
            ), Times.Once);
        }

        [Fact]
        public void GetInferenceHistory_PassesCategory_ToRepository()
        {
            // Arrange
            _mockStatsRepo.Setup(r => r.GetPagedHistory(
                It.IsAny<DateTime>(), null, 1, 10, "timestamp", true,
                null, "bottle", null, null, null))
                .Returns(new PagedResult<InferenceHistoryDto>
                {
                    Items = new List<InferenceHistoryDto>(),
                    TotalCount = 0,
                    PageNumber = 1,
                    PageSize = 10
                });

            // Act
            _statisticsService.GetInferenceHistory(1, "Admin", 1, 10, "timestamp", true,
                category: "bottle");

            // Assert
            _mockStatsRepo.Verify(r => r.GetPagedHistory(
                It.IsAny<DateTime>(), null, 1, 10, "timestamp", true,
                null, "bottle", null, null, null
            ), Times.Once);
        }

        [Fact]
        public void GetInferenceHistory_IgnoresFilterUsername_ForNonAdmin()
        {
            // Arrange
            _mockStatsRepo.Setup(r => r.GetPagedHistory(
                It.IsAny<DateTime>(), 1, 1, 10, "timestamp", true,
                null, null, null, null, null))
                .Returns(new PagedResult<InferenceHistoryDto>
                {
                    Items = new List<InferenceHistoryDto>(),
                    TotalCount = 0,
                    PageNumber = 1,
                    PageSize = 10
                });

            // Act
            _statisticsService.GetInferenceHistory(1, "User", 1, 10, "timestamp", true,
                filterUsername: "some.operator");

            // Assert
            _mockStatsRepo.Verify(r => r.GetPagedHistory(
                It.IsAny<DateTime>(), 1, 1, 10, "timestamp", true,
                null, null, null, null, null
            ), Times.Once);
        }
        #endregion
    }
}