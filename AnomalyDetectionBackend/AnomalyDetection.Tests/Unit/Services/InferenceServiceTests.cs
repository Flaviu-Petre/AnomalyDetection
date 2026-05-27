using AnomalyDetection.Api.Models.Domain;
using AnomalyDetection.Api.Services;
using AnomalyDetection.Api.Services.Interfaces;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Moq;

namespace AnomalyDetection.Tests.Unit.Services
{
    public class InferenceServiceTests
    {
        #region Setup
        private readonly Mock<IModelManagerService> _mockModelManager;
        private readonly Mock<IStatisticsService> _mockStatisticsService;
        private readonly Mock<IRouterService> _mockRouterService;
        private readonly Mock<ILogger<InferenceService>> _mockLogger;
        private readonly InferenceService _inferenceService;

        public InferenceServiceTests()
        {
            _mockModelManager = new Mock<IModelManagerService>();
            _mockStatisticsService = new Mock<IStatisticsService>();
            _mockRouterService = new Mock<IRouterService>();
            _mockLogger = new Mock<ILogger<InferenceService>>();

            _inferenceService = new InferenceService(
                _mockModelManager.Object,
                _mockStatisticsService.Object,
                _mockRouterService.Object,
                _mockLogger.Object);
        }

        private static Stream CreateFakeImageStream()
        {
            return new MemoryStream(new byte[] { 0x89, 0x50, 0x4E, 0x47 });
        }

        private void SetupSuccessfulPipeline(
            string category = "bottle",
            float confidence = 0.95f,
            bool isAnomaly = false,
            float score = 0.3f,
            float threshold = 0.5f)
        {
            var mockAnomalyService = new Mock<IAnomalyDetectionService>();
            var metadata = new ModelMetadata
            {
                ClassName = category,
                OptimalThreshold = threshold,
                ScoreMin = 0f,
                ScoreMax = 1f,
                ApplyMask = true,
                HeatmapUseGlobalMax = false
            };
            var anomalyResult = new AnomalyResult
            {
                IsAnomaly = isAnomaly,
                Score = score,
                UsedThreshold = threshold,
                HeatmapBase64 = null
            };

            _mockRouterService
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync((category, confidence));

            _mockModelManager
                .Setup(m => m.GetModelForCategory(category))
                .Returns((mockAnomalyService.Object, metadata));

            mockAnomalyService
                .Setup(s => s.PredictAnomalyScore(
                    It.IsAny<Stream>(), threshold, 0f, 1f,
                    true, false, It.IsAny<bool>()))
                .Returns(anomalyResult);
        }
        #endregion

        #region ProcessImageAsync Router Tests
        [Fact]
        public async Task ProcessImageAsync_ThrowsInvalidOperationException_WhenCategoryIsUnknown()
        {
            // Arrange
            _mockRouterService
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync(("unknown", 0.1f));

            // Act
            Func<Task> act = () => _inferenceService.ProcessImageAsync(
                CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            await act.Should().ThrowAsync<InvalidOperationException>()
                     .WithMessage("*Image not recognized*");
        }

        [Fact]
        public async Task ProcessImageAsync_CallsRouter_WithImageStream()
        {
            // Arrange
            SetupSuccessfulPipeline();

            // Act
            await _inferenceService.ProcessImageAsync(CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            _mockRouterService.Verify(r => r.ClassifyAsync(It.IsAny<Stream>()), Times.Once);
        }

        [Fact]
        public async Task ProcessImageAsync_DoesNotCallModelManager_WhenRouterRejectsImage()
        {
            // Arrange
            _mockRouterService
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync(("unknown", 0.05f));

            // Act
            try { await _inferenceService.ProcessImageAsync(CreateFakeImageStream(), "test.png", 1, false); } catch { }

            // Assert
            _mockModelManager.Verify(m => m.GetModelForCategory(It.IsAny<string>()), Times.Never);
        }
        #endregion

        #region ProcessImageAsync — Model Pipeline Tests
        [Fact]
        public async Task ProcessImageAsync_CallsModelManager_WithPredictedCategory()
        {
            // Arrange
            SetupSuccessfulPipeline(category: "capsule", confidence: 0.98f);

            // Act
            await _inferenceService.ProcessImageAsync(CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            _mockModelManager.Verify(m => m.GetModelForCategory("capsule"), Times.Once);
        }

        [Fact]
        public async Task ProcessImageAsync_ReturnsCorrectCategory_InResponse()
        {
            // Arrange
            SetupSuccessfulPipeline(category: "bottle", confidence: 0.95f);

            // Act
            var result = await _inferenceService.ProcessImageAsync(
                CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            result.PredictedCategory.Should().Be("bottle");
        }

        [Fact]
        public async Task ProcessImageAsync_ReturnsIsAnomaly_WhenAnomalyDetected()
        {
            // Arrange
            SetupSuccessfulPipeline(isAnomaly: true, score: 0.9f);

            // Act
            var result = await _inferenceService.ProcessImageAsync(
                CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            result.IsAnomaly.Should().BeTrue();
            result.Score.Should().Be(0.9f);
        }

        [Fact]
        public async Task ProcessImageAsync_ReturnsIsNotAnomaly_WhenNormalImage()
        {
            // Arrange
            SetupSuccessfulPipeline(isAnomaly: false, score: 0.2f);

            // Act
            var result = await _inferenceService.ProcessImageAsync(
                CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            result.IsAnomaly.Should().BeFalse();
            result.Score.Should().Be(0.2f);
        }

        [Fact]
        public async Task ProcessImageAsync_ReturnsUsedThreshold_InResponse()
        {
            // Arrange
            SetupSuccessfulPipeline(threshold: 0.5f);

            // Act
            var result = await _inferenceService.ProcessImageAsync(
                CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            result.UsedThreshold.Should().Be(0.5f);
        }
        #endregion

        #region ProcessImageAsync — Statistics Tests
        [Fact]
        public async Task ProcessImageAsync_CallsSaveInferenceResult_AfterSuccessfulInference()
        {
            // Arrange
            SetupSuccessfulPipeline(category: "bottle", isAnomaly: true, score: 0.8f, threshold: 0.5f);

            // Act
            await _inferenceService.ProcessImageAsync(CreateFakeImageStream(), "photo.png", 42, false);

            // Assert
            _mockStatisticsService.Verify(s => s.SaveInferenceResult(
                "bottle", true, 0.8f, 0.5f, 42, "photo.png"
            ), Times.Once);
        }

        [Fact]
        public async Task ProcessImageAsync_DoesNotSaveStatistics_WhenRouterRejectsImage()
        {
            // Arrange
            _mockRouterService
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync(("unknown", 0.05f));

            // Act
            try { await _inferenceService.ProcessImageAsync(CreateFakeImageStream(), "test.png", 1, false); } catch { }

            // Assert
            _mockStatisticsService.Verify(s => s.SaveInferenceResult(
                It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<float>(),
                It.IsAny<float>(), It.IsAny<int>(), It.IsAny<string>()
            ), Times.Never);
        }

        [Fact]
        public async Task ProcessImageAsync_SavesStatistics_WithCorrectUserId()
        {
            // Arrange
            SetupSuccessfulPipeline();

            // Act
            await _inferenceService.ProcessImageAsync(CreateFakeImageStream(), "test.png", 99, false);

            // Assert
            _mockStatisticsService.Verify(s => s.SaveInferenceResult(
                It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<float>(),
                It.IsAny<float>(), 99, It.IsAny<string>()
            ), Times.Once);
        }
        #endregion

        #region ProcessImageAsync — Heatmap Tests
        [Fact]
        public async Task ProcessImageAsync_ReturnsNullHeatmap_WhenReturnHeatmapIsFalse()
        {
            // Arrange
            SetupSuccessfulPipeline();

            // Act
            var result = await _inferenceService.ProcessImageAsync(
                CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            result.HeatmapBase64.Should().BeNull();
        }

        [Fact]
        public async Task ProcessImageAsync_PropagatesFileNotFoundException_WhenModelNotFound()
        {
            // Arrange
            _mockRouterService
                .Setup(r => r.ClassifyAsync(It.IsAny<Stream>()))
                .ReturnsAsync(("bottle", 0.95f));

            _mockModelManager
                .Setup(m => m.GetModelForCategory("bottle"))
                .Throws(new FileNotFoundException("Memory bank not found"));

            // Act
            Func<Task> act = () => _inferenceService.ProcessImageAsync(
                CreateFakeImageStream(), "test.png", 1, false);

            // Assert
            await act.Should().ThrowAsync<FileNotFoundException>();
        }
        #endregion
    }
}