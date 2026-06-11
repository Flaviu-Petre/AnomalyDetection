using AnomalyDetection.Api.Controllers;
using AnomalyDetection.Api.Models.DTOs;
using AnomalyDetection.Api.Services.Interfaces;
using FluentAssertions;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Moq;
using System.Security.Claims;

namespace AnomalyDetection.Tests.Unit.Controllers
{
    public class StatisticsControllerTests
    {
        #region Setup
        private readonly Mock<IStatisticsService> _mockStatisticsService;
        private readonly Mock<ILogger<StatisticsController>> _mockLogger;
        private readonly StatisticsController _controller;

        public StatisticsControllerTests()
        {
            _mockStatisticsService = new Mock<IStatisticsService>();
            _mockLogger = new Mock<ILogger<StatisticsController>>();
            _controller = new StatisticsController(_mockStatisticsService.Object, _mockLogger.Object);
        }

        private void SetupAuthenticatedUser(string userId, string role = "User")
        {
            var claims = new List<Claim>
            {
                new Claim("id", userId),
                new Claim(ClaimTypes.Role, role)
            };
            var identity = new ClaimsIdentity(claims, "TestAuth");
            var principal = new ClaimsPrincipal(identity);

            _controller.ControllerContext = new ControllerContext
            {
                HttpContext = new DefaultHttpContext { User = principal }
            };
        }

        private static DashboardStatsResponse CreateFakeDashboardStats() => new DashboardStatsResponse
        {
            TotalInferencesThisWeek = 10,
            TotalAnomaliesThisWeek = 3,
            OverallAnomalyRatePercentage = 30.0,
            AnomaliesByCategory = new Dictionary<string, int> { { "bottle", 2 }, { "capsule", 1 } },
            InferencesByDay = new Dictionary<string, int> { { "2025-01-01", 5 }, { "2025-01-02", 5 } }
        };

        private static PagedResult<InferenceHistoryDto> CreateFakePagedHistory() =>
            new PagedResult<InferenceHistoryDto>
            {
                Items = new List<InferenceHistoryDto>
                {
                    new InferenceHistoryDto { Id = 1, Category = "bottle", IsAnomaly = true,  Score = 0.8f },
                    new InferenceHistoryDto { Id = 2, Category = "capsule", IsAnomaly = false, Score = 0.2f }
                },
                TotalCount = 2,
                PageNumber = 1,
                PageSize = 10
            };
        #endregion

        #region GetDashboardStats Tests
        [Fact]
        public void GetDashboardStats_ReturnsOk_WithStats_ForUser()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            var stats = CreateFakeDashboardStats();
            _mockStatisticsService.Setup(s => s.GetWeeklyStatistics(1, "User")).Returns(stats);

            // Act
            var result = _controller.GetDashboardStats();

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(stats);
        }

        [Fact]
        public void GetDashboardStats_ReturnsOk_WithStats_ForAdmin()
        {
            // Arrange
            SetupAuthenticatedUser("2", "Admin");
            var stats = CreateFakeDashboardStats();
            _mockStatisticsService.Setup(s => s.GetWeeklyStatistics(2, "Admin")).Returns(stats);

            // Act
            var result = _controller.GetDashboardStats();

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(stats);
        }

        [Fact]
        public void GetDashboardStats_CallsService_WithCorrectUserIdAndRole()
        {
            // Arrange
            SetupAuthenticatedUser("5", "Admin");
            _mockStatisticsService.Setup(s => s.GetWeeklyStatistics(5, "Admin"))
                                  .Returns(CreateFakeDashboardStats());

            // Act
            _controller.GetDashboardStats();

            // Assert
            _mockStatisticsService.Verify(s => s.GetWeeklyStatistics(5, "Admin"), Times.Once);
        }

        [Fact]
        public void GetDashboardStats_Returns500_WhenExceptionIsThrown()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            _mockStatisticsService.Setup(s => s.GetWeeklyStatistics(It.IsAny<int>(), It.IsAny<string>()))
                                  .Throws(new Exception("DB error"));

            // Act
            var result = _controller.GetDashboardStats();

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }

        [Fact]
        public void GetDashboardStats_ReturnsCorrectAnomalyRate()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            var stats = CreateFakeDashboardStats();
            _mockStatisticsService.Setup(s => s.GetWeeklyStatistics(1, "User")).Returns(stats);

            // Act
            var result = _controller.GetDashboardStats() as OkObjectResult;

            // Assert
            var response = result!.Value as DashboardStatsResponse;
            response!.OverallAnomalyRatePercentage.Should().Be(30.0);
            response.TotalInferencesThisWeek.Should().Be(10);
            response.TotalAnomaliesThisWeek.Should().Be(3);
        }
        #endregion

        #region GetHistory Tests
        [Fact]
        public void GetHistory_ReturnsOk_WithPagedHistory()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            var history = CreateFakePagedHistory();
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                1, "User", 1, 10, "timestamp", true,
                null, null, null, null, null))
                .Returns(history);

            // Act
            var result = _controller.GetHistory(1, 10, "timestamp", true);

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(history);
        }

        [Fact]
        public void GetHistory_CallsService_WithCorrectParameters()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                It.IsAny<int>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<int>(),
                It.IsAny<string>(), It.IsAny<bool>(),
                null, null, null, null, null))
                .Returns(CreateFakePagedHistory());

            // Act
            _controller.GetHistory(page: 2, pageSize: 5, sortBy: "score", sortDesc: false);

            // Assert
            _mockStatisticsService.Verify(s => s.GetInferenceHistory(
                1, "Admin", 2, 5, "score", false,
                null, null, null, null, null
            ), Times.Once);
        }

        [Fact]
        public void GetHistory_UsesDefaultParameters_WhenNotProvided()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                It.IsAny<int>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<int>(),
                It.IsAny<string>(), It.IsAny<bool>(),
                null, null, null, null, null))
                .Returns(CreateFakePagedHistory());

            // Act
            _controller.GetHistory();

            // Assert
            _mockStatisticsService.Verify(s => s.GetInferenceHistory(
                1, "User", 1, 10, "timestamp", true,
                null, null, null, null, null
            ), Times.Once);
        }

        [Fact]
        public void GetHistory_Returns500_WhenExceptionIsThrown()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                It.IsAny<int>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<int>(),
                It.IsAny<string>(), It.IsAny<bool>(),
                null, null, null, null, null))
                .Throws(new Exception("DB error"));

            // Act
            var result = _controller.GetHistory();

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }

        [Fact]
        public void GetHistory_ReturnsCorrectPaginationData()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            var history = CreateFakePagedHistory();
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                It.IsAny<int>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<int>(),
                It.IsAny<string>(), It.IsAny<bool>(),
                null, null, null, null, null))
                .Returns(history);

            // Act
            var result = _controller.GetHistory() as OkObjectResult;

            // Assert
            var response = result!.Value as PagedResult<InferenceHistoryDto>;
            response!.TotalCount.Should().Be(2);
            response.PageNumber.Should().Be(1);
            response.PageSize.Should().Be(10);
            response.Items.Should().HaveCount(2);
        }

        [Fact]
        public void GetHistory_CallsService_WithIsAnomalyFilter()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                It.IsAny<int>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<int>(),
                It.IsAny<string>(), It.IsAny<bool>(),
                true, null, null, null, null))
                .Returns(CreateFakePagedHistory());

            // Act
            _controller.GetHistory(isAnomaly: true);

            // Assert
            _mockStatisticsService.Verify(s => s.GetInferenceHistory(
                1, "User", 1, 10, "timestamp", true,
                true, null, null, null, null
            ), Times.Once);
        }

        [Fact]
        public void GetHistory_CallsService_WithCategoryFilter()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                It.IsAny<int>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<int>(),
                It.IsAny<string>(), It.IsAny<bool>(),
                null, "bottle", null, null, null))
                .Returns(CreateFakePagedHistory());

            // Act
            _controller.GetHistory(category: "bottle");

            // Assert
            _mockStatisticsService.Verify(s => s.GetInferenceHistory(
                1, "User", 1, 10, "timestamp", true,
                null, "bottle", null, null, null
            ), Times.Once);
        }

        [Fact]
        public void GetHistory_CallsService_WithUsernameFilter_ForAdmin()
        {
            // Arrange
            SetupAuthenticatedUser("1", "Admin");
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                It.IsAny<int>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<int>(),
                It.IsAny<string>(), It.IsAny<bool>(),
                null, null, "some.operator", null, null))
                .Returns(CreateFakePagedHistory());

            // Act
            _controller.GetHistory(filterUsername: "some.operator");

            // Assert
            _mockStatisticsService.Verify(s => s.GetInferenceHistory(
                1, "Admin", 1, 10, "timestamp", true,
                null, null, "some.operator", null, null
            ), Times.Once);
        }

        [Fact]
        public void GetHistory_PassesUsernameFilter_ToService_ForNonAdmin()
        {
            // Arrange
            SetupAuthenticatedUser("1", "User");
            _mockStatisticsService.Setup(s => s.GetInferenceHistory(
                It.IsAny<int>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<int>(),
                It.IsAny<string>(), It.IsAny<bool>(),
                null, null, "some.operator", null, null))
                .Returns(CreateFakePagedHistory());

            // Act
            _controller.GetHistory(filterUsername: "some.operator");

            // Assert
            _mockStatisticsService.Verify(s => s.GetInferenceHistory(
                1, "User", 1, 10, "timestamp", true,
                null, null, "some.operator", null, null
            ), Times.Once);
        }
        #endregion
    }
}