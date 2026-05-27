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
    public class InferenceControllerTests
    {
        #region Setup
        private readonly Mock<IInferenceService> _mockInferenceService;
        private readonly Mock<ILogger<InferenceController>> _mockLogger;
        private readonly InferenceController _controller;

        public InferenceControllerTests()
        {
            _mockInferenceService = new Mock<IInferenceService>();
            _mockLogger = new Mock<ILogger<InferenceController>>();
            _controller = new InferenceController(_mockInferenceService.Object, _mockLogger.Object);

            SetupAuthenticatedUser("1", "User");
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

        private static IFormFile CreateMockFormFile(string fileName = "test.png", long length = 100)
        {
            var mockFile = new Mock<IFormFile>();
            var stream = new MemoryStream(new byte[length]);

            mockFile.Setup(f => f.FileName).Returns(fileName);
            mockFile.Setup(f => f.Length).Returns(length);
            mockFile.Setup(f => f.OpenReadStream()).Returns(stream);

            return mockFile.Object;
        }

        private static AnomalyResponse CreateFakeAnomalyResponse(
            string category = "bottle",
            bool isAnomaly = false,
            float score = 0.3f,
            float threshold = 0.5f) => new AnomalyResponse
            {
                PredictedCategory = category,
                IsAnomaly = isAnomaly,
                Score = score,
                UsedThreshold = threshold,
                HeatmapBase64 = null
            };
        #endregion

        #region DetectAnomaly — Validation Tests
        [Fact]
        public async Task DetectAnomaly_ReturnsBadRequest_WhenImageIsNull()
        {
            // Act
            var result = await _controller.DetectAnomaly(null!, false);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("No image file was uploaded.");
        }

        [Fact]
        public async Task DetectAnomaly_ReturnsBadRequest_WhenImageIsEmpty()
        {
            // Arrange
            var emptyFile = CreateMockFormFile(length: 0);

            // Act
            var result = await _controller.DetectAnomaly(emptyFile, false);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("No image file was uploaded.");
        }

        [Fact]
        public async Task DetectAnomaly_DoesNotCallService_WhenImageIsNull()
        {
            // Act
            await _controller.DetectAnomaly(null!, false);

            // Assert
            _mockInferenceService.Verify(s => s.ProcessImageAsync(
                It.IsAny<Stream>(), It.IsAny<string>(),
                It.IsAny<int>(), It.IsAny<bool>()
            ), Times.Never);
        }
        #endregion

        #region DetectAnomaly — Success Tests
        [Fact]
        public async Task DetectAnomaly_ReturnsOk_WhenImageIsValid()
        {
            // Arrange
            var mockFile = CreateMockFormFile("part.png");
            var response = CreateFakeAnomalyResponse();
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), "part.png", 1, false))
                .ReturnsAsync(response);

            // Act
            var result = await _controller.DetectAnomaly(mockFile, false);

            // Assert
            result.Should().BeOfType<OkObjectResult>()
                  .Which.Value.Should().Be(response);
        }

        [Fact]
        public async Task DetectAnomaly_ReturnsIsAnomaly_WhenDefectDetected()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            var response = CreateFakeAnomalyResponse(isAnomaly: true, score: 0.9f);
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<bool>()))
                .ReturnsAsync(response);

            // Act
            var result = await _controller.DetectAnomaly(mockFile, false) as OkObjectResult;

            // Assert
            var anomalyResponse = result!.Value as AnomalyResponse;
            anomalyResponse!.IsAnomaly.Should().BeTrue();
            anomalyResponse.Score.Should().Be(0.9f);
        }

        [Fact]
        public async Task DetectAnomaly_ReturnsNormalResult_WhenNoDefectDetected()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            var response = CreateFakeAnomalyResponse(isAnomaly: false, score: 0.1f);
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<bool>()))
                .ReturnsAsync(response);

            // Act
            var result = await _controller.DetectAnomaly(mockFile, false) as OkObjectResult;

            // Assert
            var anomalyResponse = result!.Value as AnomalyResponse;
            anomalyResponse!.IsAnomaly.Should().BeFalse();
            anomalyResponse.Score.Should().Be(0.1f);
        }

        [Fact]
        public async Task DetectAnomaly_CallsService_WithCorrectImageName()
        {
            // Arrange
            var mockFile = CreateMockFormFile("industrial_part.png");
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), "industrial_part.png", It.IsAny<int>(), It.IsAny<bool>()))
                .ReturnsAsync(CreateFakeAnomalyResponse());

            // Act
            await _controller.DetectAnomaly(mockFile, false);

            // Assert
            _mockInferenceService.Verify(s => s.ProcessImageAsync(
                It.IsAny<Stream>(), "industrial_part.png", It.IsAny<int>(), It.IsAny<bool>()
            ), Times.Once);
        }

        [Fact]
        public async Task DetectAnomaly_CallsService_WithCorrectUserId()
        {
            // Arrange
            SetupAuthenticatedUser("42");
            var mockFile = CreateMockFormFile();
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), It.IsAny<string>(), 42, It.IsAny<bool>()))
                .ReturnsAsync(CreateFakeAnomalyResponse());

            // Act
            await _controller.DetectAnomaly(mockFile, false);

            // Assert
            _mockInferenceService.Verify(s => s.ProcessImageAsync(
                It.IsAny<Stream>(), It.IsAny<string>(), 42, It.IsAny<bool>()
            ), Times.Once);
        }

        [Fact]
        public async Task DetectAnomaly_CallsService_WithReturnHeatmapTrue()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), It.IsAny<string>(), It.IsAny<int>(), true))
                .ReturnsAsync(CreateFakeAnomalyResponse());

            // Act
            await _controller.DetectAnomaly(mockFile, returnHeatmap: true);

            // Assert
            _mockInferenceService.Verify(s => s.ProcessImageAsync(
                It.IsAny<Stream>(), It.IsAny<string>(), It.IsAny<int>(), true
            ), Times.Once);
        }

        [Fact]
        public async Task DetectAnomaly_UsesUnknownImageName_WhenFileNameIsNull()
        {
            // Arrange
            var mockFile = new Mock<IFormFile>();
            mockFile.Setup(f => f.FileName).Returns((string)null!);
            mockFile.Setup(f => f.Length).Returns(100);
            mockFile.Setup(f => f.OpenReadStream()).Returns(new MemoryStream(new byte[100]));

            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), "Unknown Image", It.IsAny<int>(), It.IsAny<bool>()))
                .ReturnsAsync(CreateFakeAnomalyResponse());

            // Act
            await _controller.DetectAnomaly(mockFile.Object, false);

            // Assert
            _mockInferenceService.Verify(s => s.ProcessImageAsync(
                It.IsAny<Stream>(), "Unknown Image", It.IsAny<int>(), It.IsAny<bool>()
            ), Times.Once);
        }
        #endregion

        #region DetectAnomaly — Error Handling Tests
        [Fact]
        public async Task DetectAnomaly_ReturnsBadRequest_WhenRouterRejectsImage()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<bool>()))
                .ThrowsAsync(new InvalidOperationException("Image not recognized. Please upload a valid factory part."));

            // Act
            var result = await _controller.DetectAnomaly(mockFile, false);

            // Assert
            result.Should().BeOfType<BadRequestObjectResult>()
                  .Which.Value.Should().Be("Image not recognized. Please upload a valid factory part.");
        }

        [Fact]
        public async Task DetectAnomaly_ReturnsNotFound_WhenModelIsMissing()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<bool>()))
                .ThrowsAsync(new FileNotFoundException("Model not found"));

            // Act
            var result = await _controller.DetectAnomaly(mockFile, false);

            // Assert
            result.Should().BeOfType<NotFoundObjectResult>()
                  .Which.Value.Should().Be("The AI model for this category is currently unavailable.");
        }

        [Fact]
        public async Task DetectAnomaly_Returns500_WhenUnexpectedExceptionIsThrown()
        {
            // Arrange
            var mockFile = CreateMockFormFile();
            _mockInferenceService
                .Setup(s => s.ProcessImageAsync(It.IsAny<Stream>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<bool>()))
                .ThrowsAsync(new Exception("Unexpected error"));

            // Act
            var result = await _controller.DetectAnomaly(mockFile, false);

            // Assert
            result.Should().BeOfType<ObjectResult>()
                  .Which.StatusCode.Should().Be(500);
        }
        #endregion
    }
}